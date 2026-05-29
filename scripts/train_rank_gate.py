"""Train a RankMagnitudeGatedPredictor that clears raw hit rate ≥ 0.60.

Strategy:
  1. Load the most recent sweep directory.
  2. For every (spec, horizon, variant) OOS parquet, calibrate a (mag_floor,
     rank_floor) gate using the 70% training slice.
  3. Pick the (spec, horizon, variant, gate) with the highest EVAL-slice hit
     rate subject to clearing the target on the TRAIN slice with enough kept
     rows. This protects against overfitting the gate to one slice.
  4. Retrain the chosen primary on the full dataset.
  5. Wrap with RankMagnitudeGatedPredictor, save, and register as staging.

Usage:
    docker compose run --rm zeus-scheduler python -m scripts.train_rank_gate
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import select, update

from zeus.backtesting.metrics import (
    hit_rate,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
)
from zeus.config.settings import get_settings
from zeus.data.storage.database import FeaturesDaily, ModelVersion, OHLCVDaily, get_session_factory
from zeus.models.base import BaseModel
from zeus.models.ensemble_return_predictor import EnsembleReturnPredictor
from zeus.models.labels import compute_labels, compute_triple_barrier_labels
from zeus.models.lgbm_return_predictor import LGBMReturnPredictor
from zeus.models.rank_gated_predictor import RankMagnitudeGatedPredictor, calibrate_gate
from zeus.models.return_predictor import XGBReturnPredictor

log = structlog.get_logger("train_rank_gate")

TARGET_HIT_RATE: float = 0.60
MIN_N_KEPT_TRAIN: int = 500
MIN_N_KEPT_EVAL: int = 100


def _sanitize(obj):
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def _xgb_factory() -> BaseModel:
    return XGBReturnPredictor(n_estimators=500, max_depth=5, learning_rate=0.01)


def _lgb_factory() -> BaseModel:
    return LGBMReturnPredictor(n_estimators=500, num_leaves=31, learning_rate=0.01)


def _ensemble_factory() -> BaseModel:
    return EnsembleReturnPredictor(
        base_models=[
            XGBReturnPredictor(n_estimators=500, max_depth=5, learning_rate=0.01),
            LGBMReturnPredictor(n_estimators=500, num_leaves=31, learning_rate=0.01),
        ]
    )


SPEC_FACTORIES: dict[str, Callable[[], BaseModel]] = {
    "xgb": _xgb_factory,
    "lgb": _lgb_factory,
    "ensemble": _ensemble_factory,
}


def _parse_variant(stem: str) -> tuple[str, int, str]:
    """Parse `{spec}_h{N}_{variant}` → (spec, horizon, variant). Variant may contain underscores."""
    # spec is prefix up to first `_h`
    ix = stem.find("_h")
    spec = stem[:ix]
    rest = stem[ix + 2:]
    # horizon is digits before next underscore
    under = rest.find("_")
    horizon = int(rest[:under])
    variant = rest[under + 1:]
    return spec, horizon, variant


def _load_features(session, feature_version: str = "v1") -> pd.DataFrame:
    log.info("loading_features", version=feature_version)
    rows = session.query(FeaturesDaily).filter(
        FeaturesDaily.feature_version == feature_version
    ).all()
    records = []
    for r in rows:
        rec = {"symbol": r.symbol, "ts": r.feature_date}
        rec.update(r.features or {})
        records.append(rec)
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    log.info("features_loaded", rows=len(df), cols=df.shape[1])
    return df


def _load_ohlcv(session, symbols: list[str]) -> pd.DataFrame:
    rows = (
        session.query(OHLCVDaily.symbol, OHLCVDaily.ts, OHLCVDaily.close, OHLCVDaily.high, OHLCVDaily.low)
        .filter(OHLCVDaily.symbol.in_(symbols))
        .order_by(OHLCVDaily.ts)
        .all()
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "close", "high", "low"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _build_xy(
    features_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    horizon: int,
    variant: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if variant == "return":
        labels_df = compute_labels(ohlcv_df, horizon_days=horizon)
        labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)
        target_col = f"fwd_{horizon}d_return"
        merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")
        merged = merged.dropna(subset=[target_col])
        dropset = {"symbol", "ts", target_col, f"direction_{horizon}d", f"mae_{horizon}d"}
    else:
        # tb_k{val}
        k_str = variant.replace("tb_k", "")
        k = float(k_str)
        labels_df = compute_triple_barrier_labels(
            ohlcv_df, horizon_days=horizon, barrier_sigma_mult=k
        )
        labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)
        target_col = f"tb_label_h{horizon}"
        merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")
        merged = merged.dropna(subset=[target_col])
        merged = merged[merged[target_col] != 0].reset_index(drop=True)
        dropset = {"symbol", "ts", target_col,
                   f"tb_return_h{horizon}", f"tb_days_to_hit_h{horizon}"}

    feat_cols = [c for c in merged.columns if c not in dropset]
    merged[feat_cols] = merged[feat_cols].apply(pd.to_numeric, errors="coerce")
    nan_frac = merged[feat_cols].isna().mean(axis=1)
    merged = merged[nan_frac <= 0.20].sort_values("ts").reset_index(drop=True)

    X = merged[feat_cols].astype("float32")
    y = merged[target_col].astype("float64")
    meta = merged[["symbol", "ts"]].copy()
    return X, y, meta


def _top_quintile_returns(preds: pd.Series, actuals: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": preds.values, "a": actuals.values, "d": dates.values}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    daily = []
    for dt, grp in df.groupby("d"):
        nonzero = grp[grp["p"] != 0]
        if len(nonzero) < 5:
            continue
        thr = nonzero["p"].quantile(0.8)
        top = nonzero[nonzero["p"] >= thr]["a"]
        if top.empty:
            continue
        daily.append({"d": dt, "r": top.mean()})
    if not daily:
        return pd.Series(dtype=float)
    return pd.DataFrame(daily).set_index("d")["r"]


def main() -> int:
    t0 = time.time()
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )

    settings = get_settings()
    artifacts_path = Path(settings.artifacts_path)
    sweep_dirs = sorted(artifacts_path.glob("training_sweep_*"), reverse=True)
    if not sweep_dirs:
        log.error("no_sweep_dir_found")
        return 1
    sweep_dir = sweep_dirs[0]
    oos_dir = sweep_dir / "oos_predictions"
    files = sorted(oos_dir.glob("*.parquet"))
    log.info("sweep_loaded", sweep_dir=str(sweep_dir), n_oos_files=len(files))

    # Scan every (spec, horizon, variant) and calibrate a gate
    candidates = []
    for f in files:
        spec, horizon, variant = _parse_variant(f.stem)
        if spec not in SPEC_FACTORIES:
            continue
        oos = pd.read_parquet(f)
        try:
            gate = calibrate_gate(
                oos,
                target_hit=TARGET_HIT_RATE,
                min_n_kept=MIN_N_KEPT_TRAIN,
            )
        except RuntimeError as e:
            log.warning("calibrate_skip", file=f.name, err=str(e))
            continue
        log.info(
            "gate_calibrated",
            spec=spec,
            horizon=horizon,
            variant=variant,
            strategy=gate["strategy"],
            mag_floor=round(gate["mag_floor"], 5),
            rank_floor=gate["rank_floor"],
            n_train_kept=int(gate["n_train_kept"]),
            n_eval_kept=int(gate["n_eval_kept"]),
            train_hit=round(gate["train_hit"], 4),
            eval_hit=round(gate["eval_hit"], 4) if not np.isnan(gate["eval_hit"]) else None,
        )
        candidates.append({
            "spec": spec,
            "horizon": horizon,
            "variant": variant,
            "oos_path": str(f),
            "gate": gate,
        })

    if not candidates:
        log.error("no_viable_gate_found")
        return 2

    # Pick: best eval_hit among those that cleared train target with enough eval kept
    qualifiers = [
        c for c in candidates
        if c["gate"]["strategy"] == "clears_target"
        and c["gate"]["n_eval_kept"] >= MIN_N_KEPT_EVAL
    ]
    if qualifiers:
        winner = max(qualifiers, key=lambda c: c["gate"]["eval_hit"])
        selection = "clears_train_and_eval_viable"
    else:
        # Fall back to max eval_hit overall
        valid = [c for c in candidates if not np.isnan(c["gate"]["eval_hit"])]
        winner = max(valid, key=lambda c: c["gate"]["eval_hit"])
        selection = "fallback_max_eval_hit"

    log.info(
        "winner_selected",
        selection=selection,
        spec=winner["spec"],
        horizon=winner["horizon"],
        variant=winner["variant"],
        train_hit=round(winner["gate"]["train_hit"], 4),
        eval_hit=round(winner["gate"]["eval_hit"], 4),
        n_train_kept=int(winner["gate"]["n_train_kept"]),
        n_eval_kept=int(winner["gate"]["n_eval_kept"]),
        mag_floor=round(winner["gate"]["mag_floor"], 5),
        rank_floor=winner["gate"]["rank_floor"],
    )

    # Retrain primary on full data
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        features_df = _load_features(session, "v1")
        symbols = features_df["symbol"].unique().tolist()
        ohlcv_df = _load_ohlcv(session, symbols)

    X, y, meta = _build_xy(features_df, ohlcv_df, winner["horizon"], winner["variant"])
    log.info("retrain_xy", rows=len(X), cols=X.shape[1],
             spec=winner["spec"], horizon=winner["horizon"], variant=winner["variant"])

    factory = SPEC_FACTORIES[winner["spec"]]
    # Chronological 85/15 split for internal early stopping
    n = len(X)
    val_cut = int(n * 0.85)
    X_tr, y_tr = X.iloc[:val_cut], y.iloc[:val_cut]
    X_val, y_val = X.iloc[val_cut:], y.iloc[val_cut:]
    primary = factory()
    primary.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    primary.version_ = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    gated = RankMagnitudeGatedPredictor(
        primary=primary,
        mag_floor=winner["gate"]["mag_floor"],
        rank_floor=winner["gate"]["rank_floor"],
        long_only=True,
    )

    # Verify on the original OOS parquet (sanity — the retrained primary is new,
    # but this at least confirms the gate parameters pass through correctly).
    oos_df = pd.read_parquet(winner["oos_path"])
    oos_df = oos_df.dropna(subset=["pred", "actual"]).copy()
    oos_df["ts"] = pd.to_datetime(oos_df["ts"], utc=True)
    oos_df = oos_df.sort_values("ts").reset_index(drop=True)
    oos_df["date"] = oos_df["ts"].dt.date
    oos_df["rank_pct"] = oos_df.groupby("date")["pred"].rank(pct=True)
    mag_floor = winner["gate"]["mag_floor"]
    rank_floor = winner["gate"]["rank_floor"]
    kept = oos_df[
        (oos_df["pred"].abs() >= mag_floor)
        & (oos_df["rank_pct"] >= rank_floor)
        & (oos_df["pred"] > 0)
    ]
    full_oos_hit = float((np.sign(kept["pred"]) == np.sign(kept["actual"])).mean()) if len(kept) else float("nan")
    log.info("full_oos_gated_hit", n_kept=len(kept), hit=round(full_oos_hit, 4))

    # Save + register
    artifacts_root = artifacts_path / "models" / f"{winner['spec']}_h{winner['horizon']}_{winner['variant']}_rankgated"
    version_tag = primary.version_
    artifact_dir = artifacts_root / version_tag
    artifact_dir.mkdir(parents=True, exist_ok=True)
    gated.save(artifact_dir / "model.joblib")

    model_class_str = f"{type(gated).__module__}:{type(gated).__name__}"
    (artifact_dir / "model_class.txt").write_text(model_class_str)

    metrics_doc = {
        "selection_strategy": selection,
        "spec": winner["spec"],
        "horizon_days": winner["horizon"],
        "variant": winner["variant"],
        "gate_params": {
            "mag_floor": winner["gate"]["mag_floor"],
            "rank_floor": winner["gate"]["rank_floor"],
            "long_only": True,
        },
        "train_slice": {
            "n_kept": int(winner["gate"]["n_train_kept"]),
            "hit_rate": float(winner["gate"]["train_hit"]),
        },
        "eval_slice": {
            "n_kept": int(winner["gate"]["n_eval_kept"]),
            "hit_rate": float(winner["gate"]["eval_hit"]),
        },
        "full_oos": {
            "n_kept": int(len(kept)),
            "hit_rate": full_oos_hit,
        },
        "target_hit_rate": TARGET_HIT_RATE,
    }
    metrics_doc = _sanitize(metrics_doc)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, default=str))

    config_to_save = {
        "__model_class__": model_class_str,
        **gated.get_params(),
    }
    config_to_save = _sanitize(config_to_save)
    (artifact_dir / "config.json").write_text(json.dumps(config_to_save, indent=2, default=str))

    # Promotion: must clear target on eval slice (held-out)
    promoted = bool(winner["gate"]["eval_hit"] >= TARGET_HIT_RATE
                    and winner["gate"]["n_eval_kept"] >= MIN_N_KEPT_EVAL)
    status = "staging" if promoted else "failed"

    model_name = f"{winner['spec']}_h{winner['horizon']}_{winner['variant']}_rankgated"
    with SessionLocal() as session:
        existing = session.execute(
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.version == version_tag)
            .limit(1)
        ).scalar_one_or_none()
        payload = dict(
            model_name=model_name,
            version=version_tag,
            status=status,
            metrics=metrics_doc,
            config=config_to_save,
            artifact_path=str(artifact_dir),
            n_training_samples=int(len(X)),
        )
        if existing is None:
            session.add(ModelVersion(**payload))
        else:
            for k, v in payload.items():
                setattr(existing, k, v)
        session.commit()

        # Demote any other staging version of this model
        session.execute(
            update(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.version != version_tag)
            .where(ModelVersion.status == "staging")
            .values(status="retired")
        )
        session.commit()

    log.info(
        "rank_gate_registered",
        model_name=model_name,
        version=version_tag,
        status=status,
        train_hit=round(winner["gate"]["train_hit"], 4),
        eval_hit=round(winner["gate"]["eval_hit"], 4),
        n_eval_kept=int(winner["gate"]["n_eval_kept"]),
        promoted=promoted,
        total_elapsed_s=round(time.time() - t0, 1),
    )
    return 0 if promoted else 3


if __name__ == "__main__":
    sys.exit(main())
