"""Training sweep: Option A (shorter horizon) + B (walk-forward) + D (ensemble).

For every (horizon, model_spec) pair, runs a walk-forward CV over all available
feature/OHLCV history. Reports mean, std, and information-ratio-style scores for
IC, hit rate, Sharpe, and max drawdown across folds. Picks a winner by a
composite stability-weighted score, retrains it on the full history, saves the
artifact, and registers it as `status='staging'` in `model_versions`.

Side output: per-fold OOS predictions are stored in
`artifacts/training_sweep_<ts>/oos_predictions/<spec>_<horizon>.parquet`, which
feeds Option G (meta-labeling) in a second script — no need to re-run folds.

Usage:
    docker compose run --rm zeus-scheduler python -m scripts.train_sweep
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import structlog
from dateutil.relativedelta import relativedelta
from sqlalchemy import select

from zeus.backtesting.metrics import (
    hit_rate,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
)
from zeus.config.settings import get_settings
from zeus.data.storage.database import FeaturesDaily, OHLCVDaily, get_session_factory
from zeus.models.base import BaseModel
from zeus.models.ensemble_return_predictor import EnsembleReturnPredictor
from zeus.models.labels import compute_labels, compute_triple_barrier_labels
from zeus.models.lgbm_return_predictor import LGBMReturnPredictor
from zeus.models.return_predictor import XGBReturnPredictor
from zeus.models.trainer import ModelTrainer

log = structlog.get_logger("train_sweep")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────

HORIZONS: list[int] = [5, 10]
# Label variants. "return" = raw fwd log-return (regressor target).
# "tb" = triple-barrier label ∈ {-1, 0, +1} with non-directional rows dropped;
#        this is the lever that pushes hit rate past the 0.55 ceiling we
#        saw with continuous returns.
LABEL_VARIANTS: list[tuple[str, dict]] = [
    ("return", {}),
    ("tb", {"barrier_sigma_mult": 1.0}),
    ("tb", {"barrier_sigma_mult": 1.5}),
]
FEATURE_VERSION: str = "v1"

TRAIN_WINDOW_MONTHS: int = 18
TEST_WINDOW_MONTHS: int = 3
STEP_MONTHS: int = 3           # 3-month step → ~17 folds over 6y, keeps compute reasonable
EMBARGO_TRADING_DAYS: int = 20

MAX_NAN_FRACTION: float = 0.20
EARLY_STOPPING_VAL_FRACTION: float = 0.15  # tail of each train fold used for early stopping

# Promotion gates we want the final winner to satisfy on the aggregated walk-forward
PROMOTION_GATES: dict[str, float] = {
    "mean_ic": 0.03,
    "ic_stability": 0.30,
    "mean_hit_rate": 0.52,
    "mean_sharpe_on_signals": 0.80,
    "mean_max_drawdown_on_signals": -0.25,  # >= -0.25
}


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


MODEL_SPECS: dict[str, Callable[[], BaseModel]] = {
    "xgb": _xgb_factory,
    "lgb": _lgb_factory,
    "ensemble": _ensemble_factory,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (once, shared across all sweeps)
# ─────────────────────────────────────────────────────────────────────────────

def _load_features(session, feature_version: str) -> pd.DataFrame:
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
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    log.info("features_loaded", rows=len(df), cols=df.shape[1])
    return df


def _load_ohlcv(session, symbols: list[str]) -> pd.DataFrame:
    log.info("loading_ohlcv", n_symbols=len(symbols))
    rows = (
        session.query(OHLCVDaily.symbol, OHLCVDaily.ts, OHLCVDaily.close, OHLCVDaily.high, OHLCVDaily.low)
        .filter(OHLCVDaily.symbol.in_(symbols))
        .order_by(OHLCVDaily.ts)
        .all()
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "close", "high", "low"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    log.info("ohlcv_loaded", rows=len(df))
    return df


def _build_xy_for_horizon(
    features_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    horizon: int,
    label_type: str = "return",
    label_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], str]:
    label_kwargs = label_kwargs or {}
    if label_type == "tb":
        labels_df = compute_triple_barrier_labels(
            ohlcv_df, horizon_days=horizon, **label_kwargs
        )
        labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)
        target_col = f"tb_label_h{horizon}"
        aux_cols = [f"tb_return_h{horizon}", f"tb_days_to_hit_h{horizon}"]
        merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")
        merged = merged.dropna(subset=[target_col])
        n_before = len(merged)
        merged = merged[merged[target_col] != 0].reset_index(drop=True)
        log.info(
            "tb_filter",
            horizon=horizon,
            n_before=n_before,
            n_after=len(merged),
            pct_kept=round(len(merged) / max(n_before, 1), 3),
            barrier_sigma_mult=label_kwargs.get("barrier_sigma_mult"),
        )
        non_feat = set(["symbol", "ts", target_col, *aux_cols])
    else:
        labels_df = compute_labels(ohlcv_df, horizon_days=horizon)
        labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)
        target_col = f"fwd_{horizon}d_return"
        dir_col = f"direction_{horizon}d"
        mae_col = f"mae_{horizon}d"
        merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")
        merged = merged.dropna(subset=[target_col])
        non_feat = set(["symbol", "ts", target_col, dir_col, mae_col])

    feature_cols = [c for c in merged.columns if c not in non_feat]
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce")

    nan_fractions = merged[feature_cols].isna().mean(axis=1)
    merged = merged[nan_fractions <= MAX_NAN_FRACTION]

    merged = merged.sort_values("ts").reset_index(drop=True)

    X = merged[feature_cols].astype("float32")
    y = merged[target_col].astype("float64")
    meta = merged[["symbol", "ts"]].copy()

    log.info(
        "xy_built",
        horizon=horizon,
        label_type=label_type,
        rows=len(X),
        cols=len(feature_cols),
        target=target_col,
    )
    return X, y, meta, feature_cols, target_col


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train: int
    n_test: int
    ic: float
    hit_rate: float
    sharpe_on_signals: float
    max_drawdown_on_signals: float
    per_date_ic_mean: float
    per_date_ic_std: float


def _generate_folds(
    ts_index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    step_months: int,
    embargo_days: int,
) -> list[tuple[date, date, date, date]]:
    unique_dates = sorted(ts_index.normalize().unique())
    if not unique_dates:
        return []
    first = unique_dates[0].date()
    last = unique_dates[-1].date()

    folds = []
    train_start = first
    while True:
        train_end = train_start + relativedelta(months=train_months) - relativedelta(days=1)
        post_train = [d.date() for d in unique_dates if d.date() > train_end]
        if len(post_train) < embargo_days + 1:
            break
        test_start = post_train[embargo_days]
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)
        if test_end > last:
            break
        folds.append((train_start, train_end, test_start, test_end))
        train_start = train_start + relativedelta(months=step_months)
    return folds


def _per_date_spearman(preds: pd.Series, actuals: pd.Series, dates: pd.Series) -> pd.Series:
    from scipy import stats

    df = pd.DataFrame({"p": preds.values, "a": actuals.values, "d": dates.values})
    out = {}
    for dt, grp in df.groupby("d"):
        g = grp.dropna()
        if len(g) < 2:
            continue
        corr, _ = stats.spearmanr(g["p"], g["a"])
        if not np.isnan(corr):
            out[dt] = corr
    return pd.Series(out)


def _signal_returns_top_quintile(preds: pd.Series, actuals: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": preds.values, "a": actuals.values, "d": dates.values}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    daily = []
    for dt, grp in df.groupby("d"):
        if len(grp) < 5:
            continue
        thr = grp["p"].quantile(0.8)
        top = grp[grp["p"] >= thr]["a"]
        if top.empty:
            continue
        daily.append({"d": dt, "r": top.mean()})
    if not daily:
        return pd.Series(dtype=float)
    return pd.DataFrame(daily).set_index("d")["r"]


def _run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    factory: Callable[[], BaseModel],
    spec_name: str,
    horizon: int,
    oos_pred_path: Path,
) -> tuple[list[FoldResult], pd.DataFrame]:
    ts_index = pd.DatetimeIndex(meta["ts"])
    folds = _generate_folds(
        ts_index,
        train_months=TRAIN_WINDOW_MONTHS,
        test_months=TEST_WINDOW_MONTHS,
        step_months=STEP_MONTHS,
        embargo_days=EMBARGO_TRADING_DAYS,
    )
    log.info("walk_forward_folds", spec=spec_name, horizon=horizon, n_folds=len(folds))
    if not folds:
        return [], pd.DataFrame()

    fold_results: list[FoldResult] = []
    oos_chunks: list[pd.DataFrame] = []

    ts_dates = meta["ts"].dt.date.values

    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        tr_mask = (ts_dates >= tr_s) & (ts_dates <= tr_e)
        te_mask = (ts_dates >= te_s) & (ts_dates <= te_e)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            log.warning("empty_fold_skip", fold=fold_idx)
            continue

        X_train_full = X[tr_mask]
        y_train_full = y[tr_mask]
        X_test = X[te_mask]
        y_test = y[te_mask]
        meta_test = meta[te_mask]

        # Use last 15% of train (chronologically) as val for early stopping
        n_tr = len(X_train_full)
        val_cut = int(n_tr * (1.0 - EARLY_STOPPING_VAL_FRACTION))
        X_tr, y_tr = X_train_full.iloc[:val_cut], y_train_full.iloc[:val_cut]
        X_val, y_val = X_train_full.iloc[val_cut:], y_train_full.iloc[val_cut:]

        t0 = time.time()
        model = factory()
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        preds = model.predict(X_test).astype("float64")
        elapsed = time.time() - t0

        te_dates = meta_test["ts"]
        fold_ic = information_coefficient(preds, y_test, group=te_dates.dt.date)
        fold_hit = hit_rate(preds, y_test)
        fold_sig_ret = _signal_returns_top_quintile(preds, y_test, te_dates)
        fold_sharpe = sharpe_ratio(fold_sig_ret) if not fold_sig_ret.empty else 0.0
        fold_equity = (1 + fold_sig_ret).cumprod() if not fold_sig_ret.empty else pd.Series(dtype=float)
        fold_mdd = max_drawdown(fold_equity) if not fold_equity.empty else 0.0

        per_date_ic = _per_date_spearman(preds, y_test, te_dates)
        per_ic_mean = float(per_date_ic.mean()) if not per_date_ic.empty else 0.0
        per_ic_std = float(per_date_ic.std()) if not per_date_ic.empty else 0.0

        fr = FoldResult(
            fold=fold_idx,
            train_start=tr_s,
            train_end=tr_e,
            test_start=te_s,
            test_end=te_e,
            n_train=n_tr,
            n_test=int(te_mask.sum()),
            ic=float(fold_ic),
            hit_rate=float(fold_hit),
            sharpe_on_signals=float(fold_sharpe),
            max_drawdown_on_signals=float(fold_mdd),
            per_date_ic_mean=per_ic_mean,
            per_date_ic_std=per_ic_std,
        )
        fold_results.append(fr)
        log.info(
            "fold_done",
            spec=spec_name,
            horizon=horizon,
            fold=fold_idx,
            ic=round(fr.ic, 4),
            hit=round(fr.hit_rate, 4),
            sharpe=round(fr.sharpe_on_signals, 3),
            mdd=round(fr.max_drawdown_on_signals, 3),
            elapsed_s=round(elapsed, 1),
        )

        oos_chunk = meta_test.copy()
        oos_chunk["pred"] = preds.values
        oos_chunk["actual"] = y_test.values
        oos_chunk["fold"] = fold_idx
        oos_chunk["horizon"] = horizon
        oos_chunk["spec"] = spec_name
        oos_chunks.append(oos_chunk)

    oos_df = pd.concat(oos_chunks, ignore_index=True) if oos_chunks else pd.DataFrame()
    if not oos_df.empty:
        oos_pred_path.parent.mkdir(parents=True, exist_ok=True)
        oos_df.to_parquet(oos_pred_path, index=False)
        log.info("oos_predictions_saved", path=str(oos_pred_path), rows=len(oos_df))

    return fold_results, oos_df


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + ranking
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(fr: list[FoldResult]) -> dict:
    if not fr:
        return {}
    ic = np.array([f.ic for f in fr])
    hr = np.array([f.hit_rate for f in fr])
    sh = np.array([f.sharpe_on_signals for f in fr])
    mdd = np.array([f.max_drawdown_on_signals for f in fr])

    mean_ic = float(np.mean(ic))
    std_ic = float(np.std(ic, ddof=1)) if len(ic) > 1 else 0.0
    ic_stability = (mean_ic / std_ic) if std_ic > 0 else 0.0

    return {
        "n_folds": len(fr),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_stability": float(ic_stability),
        "median_ic": float(np.median(ic)),
        "mean_hit_rate": float(np.mean(hr)),
        "std_hit_rate": float(np.std(hr, ddof=1)) if len(hr) > 1 else 0.0,
        "mean_sharpe_on_signals": float(np.mean(sh)),
        "std_sharpe_on_signals": float(np.std(sh, ddof=1)) if len(sh) > 1 else 0.0,
        "mean_max_drawdown_on_signals": float(np.mean(mdd)),
        "worst_max_drawdown": float(np.min(mdd)),
        "fraction_ic_positive": float(np.mean(ic > 0)),
        "fraction_hit_above_52": float(np.mean(hr >= 0.52)),
    }


def _composite_score(agg: dict) -> float:
    """Stability-weighted composite.

    We want: consistently positive IC, hit rate above 0.52, and Sharpe that
    survives aggregation. Penalise volatility in IC across folds, and cap
    Sharpe's contribution so a single lucky fold can't dominate.
    """
    if not agg:
        return -math.inf
    ic = agg["mean_ic"]
    hr = agg["mean_hit_rate"]
    sh = agg["mean_sharpe_on_signals"]
    stab = agg["ic_stability"]
    frac_pos = agg["fraction_ic_positive"]
    mdd = agg["mean_max_drawdown_on_signals"]

    ic_component = ic * (1.0 + max(0.0, min(1.0, stab)))
    hr_component = max(0.0, hr - 0.5) * 4.0
    sh_component = max(-0.5, min(1.5, sh)) * 0.25
    mdd_component = max(-0.5, mdd) * 0.10
    pos_component = frac_pos * 0.15

    return ic_component + hr_component + sh_component + mdd_component + pos_component


def _passes_gates(agg: dict) -> tuple[bool, dict[str, bool]]:
    checks: dict[str, bool] = {}
    for k, v in PROMOTION_GATES.items():
        val = agg.get(k)
        if val is None:
            checks[k] = False
            continue
        if k == "mean_max_drawdown_on_signals":
            checks[k] = val >= v
        else:
            checks[k] = val >= v
    return all(checks.values()), checks


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(out_dir: Path, ranked_results: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "report.json").open("w") as f:
        json.dump(ranked_results, f, indent=2, default=str)

    lines: list[str] = []
    lines.append("# Training Sweep Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"Walk-forward: {TRAIN_WINDOW_MONTHS}mo train / {TEST_WINDOW_MONTHS}mo test / "
                 f"{STEP_MONTHS}mo step, {EMBARGO_TRADING_DAYS}-day embargo")
    lines.append("")
    lines.append("## Ranked leaderboard (composite score desc)")
    lines.append("")
    lines.append("| Rank | Spec | Horizon | Label | Composite | Mean IC | IC Stab | Hit% | Sharpe | MeanMDD | Folds | Gates |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(ranked_results, 1):
        a = r["aggregate"]
        variant = r.get("variant_tag", "return")
        lines.append(
            f"| {i} | {r['spec']} | {r['horizon']}d | {variant} | {r['composite']:.4f} | "
            f"{a['mean_ic']:.4f} | {a['ic_stability']:.2f} | {a['mean_hit_rate']:.4f} | "
            f"{a['mean_sharpe_on_signals']:.2f} | {a['mean_max_drawdown_on_signals']:.3f} | "
            f"{a['n_folds']} | {'PASS' if r['passed_gates'] else 'FAIL'} |"
        )
    lines.append("")

    for r in ranked_results:
        a = r["aggregate"]
        variant = r.get("variant_tag", "return")
        lines.append(f"## {r['spec']} @ {r['horizon']}d ({variant})")
        lines.append("")
        lines.append(f"- Composite score: **{r['composite']:.4f}**")
        lines.append(f"- Folds: {a['n_folds']}")
        lines.append(f"- IC: mean {a['mean_ic']:.4f}, std {a['std_ic']:.4f}, "
                     f"stability (mean/std) {a['ic_stability']:.2f}, median {a['median_ic']:.4f}")
        lines.append(f"- Hit rate: mean {a['mean_hit_rate']:.4f}, std {a['std_hit_rate']:.4f}, "
                     f"fraction folds ≥ 0.52: {a['fraction_hit_above_52']:.2f}")
        lines.append(f"- Sharpe on signals: mean {a['mean_sharpe_on_signals']:.3f}, "
                     f"std {a['std_sharpe_on_signals']:.3f}")
        lines.append(f"- Max drawdown on signals: mean {a['mean_max_drawdown_on_signals']:.3f}, "
                     f"worst {a['worst_max_drawdown']:.3f}")
        lines.append(f"- Fraction folds with positive IC: {a['fraction_ic_positive']:.2f}")
        lines.append(f"- Gates: {'PASS' if r['passed_gates'] else 'FAIL'}  {r['gate_checks']}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))
    log.info("report_written", path=str(out_dir / "report.md"))


# ─────────────────────────────────────────────────────────────────────────────
# Final train + registry
# ─────────────────────────────────────────────────────────────────────────────

def _final_train_and_register(
    winner: dict,
    features_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    artifacts_path: str,
) -> tuple[str, str, dict]:
    spec = winner["spec"]
    horizon = winner["horizon"]
    label_type = winner.get("label_type", "return")
    label_kwargs = winner.get("label_kwargs", {}) or {}
    factory = MODEL_SPECS[spec]

    X, y, meta, _, _ = _build_xy_for_horizon(
        features_df, ohlcv_df, horizon,
        label_type=label_type, label_kwargs=label_kwargs,
    )

    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        trainer = ModelTrainer(session=session, artifacts_path=artifacts_path)
        # We've already walk-forward validated — promote directly.
        result = trainer.train_and_evaluate(X, y, meta, model_factory=factory)

        agg = winner["aggregate"]
        metrics_for_db = {
            "split_metrics": result["metrics"],  # final 80/10/10 split, informational
            "walk_forward": agg,                  # the true performance numbers
            "horizon_days": horizon,
            "spec": spec,
            "label_type": label_type,
            "label_kwargs": label_kwargs,
            "composite_score": winner["composite"],
            "gate_checks": winner["gate_checks"],
        }

        passed = winner["passed_gates"]
        variant_tag = winner.get("variant_tag", label_type)
        model_name = f"{spec}_h{horizon}_{variant_tag}"
        artifact_dir = trainer.save_model(
            model=result["model"],
            metrics=metrics_for_db,
            metadata=meta,
            passed_gates=passed,
            version_string=result["version_string"],
            model_name=model_name,
        )
        log.info(
            "winner_registered",
            spec=spec,
            horizon=horizon,
            model_name=model_name,
            version=result["version_string"],
            artifact=artifact_dir,
            passed_gates=passed,
        )
    return result["version_string"], model_name, metrics_for_db


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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
    artifacts_path = settings.artifacts_path
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(artifacts_path) / f"training_sweep_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("sweep_start", out_dir=str(out_dir), horizons=HORIZONS, specs=list(MODEL_SPECS))

    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        features_df = _load_features(session, FEATURE_VERSION)
        if features_df.empty:
            log.error("no_features_found", version=FEATURE_VERSION)
            return 1
        symbols = features_df["symbol"].unique().tolist()
        ohlcv_df = _load_ohlcv(session, symbols)

    results: list[dict] = []

    for horizon in HORIZONS:
        for label_type, label_kwargs in LABEL_VARIANTS:
            variant_tag = label_type
            if label_type == "tb":
                variant_tag = f"tb_k{label_kwargs.get('barrier_sigma_mult', 1.0)}"
            X, y, meta, _, target_col = _build_xy_for_horizon(
                features_df, ohlcv_df, horizon,
                label_type=label_type, label_kwargs=label_kwargs,
            )
            if len(X) < 1000:
                log.warning("insufficient_rows_for_variant",
                            horizon=horizon, variant=variant_tag, rows=len(X))
                continue

            for spec_name, factory in MODEL_SPECS.items():
                spec_t0 = time.time()
                oos_filename = f"{spec_name}_h{horizon}_{variant_tag}.parquet"
                oos_path = out_dir / "oos_predictions" / oos_filename
                fold_results, oos_df = _run_walk_forward(
                    X=X,
                    y=y,
                    meta=meta,
                    factory=factory,
                    spec_name=spec_name,
                    horizon=horizon,
                    oos_pred_path=oos_path,
                )
                if not fold_results:
                    continue

                agg = _aggregate(fold_results)
                passed, gate_checks = _passes_gates(agg)
                composite = _composite_score(agg)

                results.append({
                    "spec": spec_name,
                    "horizon": horizon,
                    "label_type": label_type,
                    "label_kwargs": label_kwargs,
                    "variant_tag": variant_tag,
                    "target_col": target_col,
                    "aggregate": agg,
                    "passed_gates": passed,
                    "gate_checks": gate_checks,
                    "composite": composite,
                    "oos_predictions_path": str(oos_path),
                    "folds": [f.__dict__ for f in fold_results],
                })
                log.info(
                    "spec_complete",
                    spec=spec_name,
                    horizon=horizon,
                    variant=variant_tag,
                    composite=round(composite, 4),
                    mean_ic=round(agg["mean_ic"], 4),
                    mean_hit_rate=round(agg["mean_hit_rate"], 4),
                    passed_gates=passed,
                    elapsed_s=round(time.time() - spec_t0, 1),
                )

    if not results:
        log.error("sweep_produced_no_results")
        return 2

    results.sort(key=lambda r: r["composite"], reverse=True)
    _write_report(out_dir, results)

    winner = results[0]
    log.info(
        "sweep_winner",
        spec=winner["spec"],
        horizon=winner["horizon"],
        composite=round(winner["composite"], 4),
        mean_ic=round(winner["aggregate"]["mean_ic"], 4),
        mean_hit_rate=round(winner["aggregate"]["mean_hit_rate"], 4),
        passed_gates=winner["passed_gates"],
    )

    version, model_name, metrics_for_db = _final_train_and_register(
        winner=winner,
        features_df=features_df,
        ohlcv_df=ohlcv_df,
        artifacts_path=artifacts_path,
    )

    (out_dir / "winner.json").write_text(json.dumps({
        "spec": winner["spec"],
        "horizon": winner["horizon"],
        "label_type": winner.get("label_type", "return"),
        "label_kwargs": winner.get("label_kwargs", {}),
        "variant_tag": winner.get("variant_tag", "return"),
        "model_name": model_name,
        "version": version,
        "composite": winner["composite"],
        "aggregate": winner["aggregate"],
        "passed_gates": winner["passed_gates"],
        "gate_checks": winner["gate_checks"],
        "oos_predictions_path": winner["oos_predictions_path"],
    }, indent=2, default=str))

    log.info(
        "sweep_complete",
        winner_model=model_name,
        winner_version=version,
        total_elapsed_s=round(time.time() - t0, 1),
        out_dir=str(out_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
