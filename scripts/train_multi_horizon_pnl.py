"""Train + backtest rank-gated models across day (1d), swing (5d), long-term (20d).

For each horizon:
  1. Walk-forward CV of ensemble(xgb+lgb) on return-labels → OOS predictions.
  2. Calibrate (mag_floor, rank_floor) gate: maximize eval avg trade PnL
     (after 10bps round-trip cost) subject to raw hit rate ≥ 0.60 on both
     train and eval slices.
  3. Retrain ensemble on full data.
  4. Wrap with RankMagnitudeGatedPredictor. Backtest full-OOS PnL.
  5. Register each as `{spec}_h{N}_return_rankgated_pnl` staging.

Usage:
    docker compose run --rm zeus-scheduler python -m scripts.train_multi_horizon_pnl
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

from zeus.config.settings import get_settings
from zeus.data.storage.database import ModelVersion, get_session_factory
from zeus.models.base import BaseModel
from zeus.models.ensemble_return_predictor import EnsembleReturnPredictor
from zeus.models.lgbm_return_predictor import LGBMReturnPredictor
from zeus.models.rank_gated_predictor import (
    RankMagnitudeGatedPredictor,
    backtest_gate,
    calibrate_gate,
)
from zeus.models.return_predictor import XGBReturnPredictor

# Reuse sweep plumbing for walk-forward CV, data loading, and xy building
from scripts.train_sweep import (
    EARLY_STOPPING_VAL_FRACTION,
    EMBARGO_TRADING_DAYS,
    STEP_MONTHS,
    TEST_WINDOW_MONTHS,
    TRAIN_WINDOW_MONTHS,
    _build_xy_for_horizon,
    _load_features,
    _load_ohlcv,
    _run_walk_forward,
)

log = structlog.get_logger("train_multi_horizon_pnl")

HORIZONS: list[int] = [2, 3, 5, 10, 20]  # day-ish, short-swing, swing, mid, long-term
HORIZON_LABELS: dict[int, str] = {
    2: "day", 3: "day", 5: "swing", 10: "swing", 20: "long_term",
}
TARGET_HIT_RATE: float = 0.60
MIN_N_KEPT_TRAIN: int = 500
MIN_N_KEPT_EVAL: int = 100
COST_BPS: float = 10.0  # round-trip; 10bps is conservative for liquid US equities


def _ensemble_factory() -> BaseModel:
    return EnsembleReturnPredictor(
        base_models=[
            XGBReturnPredictor(n_estimators=500, max_depth=5, learning_rate=0.01),
            LGBMReturnPredictor(n_estimators=500, num_leaves=31, learning_rate=0.01),
        ]
    )


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


def _register_model(
    session_factory,
    model_name: str,
    version: str,
    status: str,
    metrics: dict,
    config: dict,
    artifact_dir: Path,
    n_training_samples: int,
) -> None:
    with session_factory() as session:
        existing = session.execute(
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.version == version)
            .limit(1)
        ).scalar_one_or_none()
        payload = dict(
            model_name=model_name,
            version=version,
            status=status,
            metrics=metrics,
            config=config,
            artifact_path=str(artifact_dir),
            n_training_samples=int(n_training_samples),
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
            .where(ModelVersion.version != version)
            .where(ModelVersion.status == "staging")
            .values(status="retired")
        )
        session.commit()


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
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = artifacts_path / f"multi_horizon_pnl_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    oos_dir = out_dir / "oos_predictions"
    oos_dir.mkdir(parents=True, exist_ok=True)

    log.info("run_start", out_dir=str(out_dir), horizons=HORIZONS, cost_bps=COST_BPS)

    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        features_df = _load_features(session, "v1")
        if features_df.empty:
            log.error("no_features_found")
            return 1
        symbols = features_df["symbol"].unique().tolist()
        ohlcv_df = _load_ohlcv(session, symbols)

    per_horizon_results: list[dict] = []

    for horizon in HORIZONS:
        horizon_t0 = time.time()
        label = HORIZON_LABELS[horizon]
        log.info("horizon_start", horizon=horizon, label=label)

        X, y, meta, _, target_col = _build_xy_for_horizon(
            features_df, ohlcv_df, horizon, label_type="return", label_kwargs={},
        )
        if len(X) < 1000:
            log.warning("insufficient_rows", horizon=horizon, rows=len(X))
            continue

        # Walk-forward CV → OOS predictions
        oos_path = oos_dir / f"ensemble_h{horizon}_return.parquet"
        fold_results, oos_df = _run_walk_forward(
            X=X, y=y, meta=meta,
            factory=_ensemble_factory,
            spec_name="ensemble",
            horizon=horizon,
            oos_pred_path=oos_path,
        )
        if oos_df.empty:
            log.warning("walk_forward_produced_no_oos", horizon=horizon)
            continue

        # Calibrate gate with PnL objective
        try:
            gate = calibrate_gate(
                oos_df,
                target_hit=TARGET_HIT_RATE,
                min_n_kept=MIN_N_KEPT_TRAIN,
                min_eval_n_kept=MIN_N_KEPT_EVAL,
                objective="pnl",
                cost_bps=COST_BPS,
                horizon_days=horizon,
            )
        except RuntimeError as e:
            log.error("calibration_failed", horizon=horizon, err=str(e))
            continue

        log.info(
            "gate_calibrated",
            horizon=horizon, label=label,
            strategy=gate["strategy"],
            mag_floor=round(gate["mag_floor"], 5),
            rank_floor=gate["rank_floor"],
            train_hit=round(gate["train_hit"], 4),
            eval_hit=round(gate["eval_hit"], 4) if not np.isnan(gate["eval_hit"]) else None,
            n_train_kept=int(gate["n_train_kept"]),
            n_eval_kept=int(gate["n_eval_kept"]),
            train_avg_pnl=round(gate["train_avg_pnl_after_cost"], 5),
            eval_avg_pnl=round(gate["eval_avg_pnl_after_cost"], 5) if not np.isnan(gate["eval_avg_pnl_after_cost"]) else None,
        )

        # Retrain ensemble on full data (chronological 85/15 for early stopping)
        n = len(X)
        val_cut = int(n * 0.85)
        X_tr, y_tr = X.iloc[:val_cut], y.iloc[:val_cut]
        X_val, y_val = X.iloc[val_cut:], y.iloc[val_cut:]
        primary = _ensemble_factory()
        primary.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        primary.version_ = f"v{run_ts}_h{horizon}"

        gated = RankMagnitudeGatedPredictor(
            primary=primary,
            mag_floor=gate["mag_floor"],
            rank_floor=gate["rank_floor"],
            long_only=True,
        )

        # Full-OOS backtest (report metric; gate params are what we saved)
        bt = backtest_gate(
            oos_df,
            mag_floor=gate["mag_floor"],
            rank_floor=gate["rank_floor"],
            long_only=True,
            cost_bps=COST_BPS,
            horizon_days=horizon,
        )
        log.info(
            "backtest_full_oos",
            horizon=horizon, label=label,
            n_trades=bt["n_trades"],
            hit_rate=round(bt["hit_rate"], 4),
            avg_trade_return_after_cost=round(bt["avg_trade_return_after_cost"], 5),
            daily_sharpe_annualized=round(bt["daily_sharpe_annualized"], 3),
            annualized_return_pct=round(bt["annualized_return_pct"] * 100, 2),
            max_drawdown_log=round(bt["max_drawdown_log"], 4),
            trades_per_year=round(bt["trades_per_year"], 1),
            win_loss_ratio=round(bt["win_loss_ratio"], 3) if not np.isinf(bt["win_loss_ratio"]) else None,
        )

        # Split-slice backtests for transparency
        oos_sorted = oos_df.sort_values("ts").reset_index(drop=True)
        split_ix = int(len(oos_sorted) * 0.70)
        train_slice_bt = backtest_gate(
            oos_sorted.iloc[:split_ix],
            mag_floor=gate["mag_floor"], rank_floor=gate["rank_floor"],
            long_only=True, cost_bps=COST_BPS, horizon_days=horizon,
        )
        eval_slice_bt = backtest_gate(
            oos_sorted.iloc[split_ix:],
            mag_floor=gate["mag_floor"], rank_floor=gate["rank_floor"],
            long_only=True, cost_bps=COST_BPS, horizon_days=horizon,
        )

        # Save artifact + register
        model_name = f"ensemble_h{horizon}_return_rankgated_pnl"
        version = f"v{run_ts}_h{horizon}"
        artifact_dir = artifacts_path / "models" / model_name / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        gated.save(artifact_dir / "model.joblib")
        model_class_str = f"{type(gated).__module__}:{type(gated).__name__}"
        (artifact_dir / "model_class.txt").write_text(model_class_str)

        metrics_doc = {
            "horizon_days": horizon,
            "horizon_label": label,
            "spec": "ensemble",
            "variant": "return",
            "gate_params": {
                "mag_floor": gate["mag_floor"],
                "rank_floor": gate["rank_floor"],
                "long_only": True,
            },
            "selection_strategy": gate["strategy"],
            "target_hit_rate": TARGET_HIT_RATE,
            "cost_bps": COST_BPS,
            "calibration": {
                "train_slice_hit": gate["train_hit"],
                "eval_slice_hit": gate["eval_hit"],
                "train_slice_avg_pnl_after_cost": gate["train_avg_pnl_after_cost"],
                "eval_slice_avg_pnl_after_cost": gate["eval_avg_pnl_after_cost"],
                "n_train_kept": int(gate["n_train_kept"]),
                "n_eval_kept": int(gate["n_eval_kept"]),
            },
            "backtest_full_oos": bt,
            "backtest_train_slice": train_slice_bt,
            "backtest_eval_slice": eval_slice_bt,
        }
        metrics_doc = _sanitize(metrics_doc)
        (artifact_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, default=str))

        config_to_save = {
            "__model_class__": model_class_str,
            **gated.get_params(),
        }
        config_to_save = _sanitize(config_to_save)
        (artifact_dir / "config.json").write_text(json.dumps(config_to_save, indent=2, default=str))

        # Promotion gates: both train and eval must clear hit target AND eval PnL > 0
        promoted = bool(
            gate["train_hit"] >= TARGET_HIT_RATE
            and gate["eval_hit"] >= TARGET_HIT_RATE
            and gate["eval_avg_pnl_after_cost"] > 0
            and gate["n_eval_kept"] >= MIN_N_KEPT_EVAL
        )
        status = "staging" if promoted else "failed"

        _register_model(
            SessionLocal, model_name, version, status,
            metrics_doc, config_to_save, artifact_dir, n_training_samples=len(X),
        )

        log.info(
            "horizon_done",
            horizon=horizon, label=label,
            model_name=model_name, version=version, status=status,
            promoted=promoted, elapsed_s=round(time.time() - horizon_t0, 1),
        )
        per_horizon_results.append({
            "horizon": horizon,
            "label": label,
            "model_name": model_name,
            "version": version,
            "promoted": promoted,
            "status": status,
            "gate_params": metrics_doc["gate_params"],
            "calibration": metrics_doc["calibration"],
            "backtest_full_oos": bt,
            "backtest_train_slice": train_slice_bt,
            "backtest_eval_slice": eval_slice_bt,
        })

    # Combined report
    report = {
        "run_ts": run_ts,
        "cost_bps": COST_BPS,
        "target_hit_rate": TARGET_HIT_RATE,
        "horizons": per_horizon_results,
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(_sanitize(report), indent=2, default=str))

    # Markdown summary table
    md = ["# Multi-Horizon Rank-Gated PnL Report", ""]
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append(f"Cost assumption: {COST_BPS} bps round-trip")
    md.append(f"Hit-rate target: ≥ {TARGET_HIT_RATE}")
    md.append("")
    md.append("## Summary (full OOS backtest, after costs)")
    md.append("")
    md.append("| Horizon | Label | Hit | n_trades | AvgTradeRet% | AnnRet% | Sharpe | MaxDD% | Trades/yr | W/L | Promoted |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in per_horizon_results:
        bt = r["backtest_full_oos"]
        avg_pct = (np.exp(bt["avg_trade_return_after_cost"]) - 1.0) * 100 if not np.isnan(bt["avg_trade_return_after_cost"]) else float("nan")
        mdd_pct = (np.exp(bt["max_drawdown_log"]) - 1.0) * 100 if bt["max_drawdown_log"] else 0.0
        md.append(
            f"| {r['horizon']}d | {r['label']} | {bt['hit_rate']:.4f} | {bt['n_trades']} | "
            f"{avg_pct:.3f} | {bt['annualized_return_pct']*100:.2f} | "
            f"{bt['daily_sharpe_annualized']:.2f} | {mdd_pct:.2f} | "
            f"{bt['trades_per_year']:.1f} | "
            f"{bt['win_loss_ratio']:.2f} | {'YES' if r['promoted'] else 'no'} |"
        )
    md.append("")
    md.append("## Train vs Eval slice hit rates")
    md.append("")
    md.append("| Horizon | Train hit | Eval hit | Train PnL/trade | Eval PnL/trade | n_eval |")
    md.append("|---|---|---|---|---|---|")
    for r in per_horizon_results:
        c = r["calibration"]
        md.append(
            f"| {r['horizon']}d | {c['train_slice_hit']:.4f} | {c['eval_slice_hit']:.4f} | "
            f"{c['train_slice_avg_pnl_after_cost']:.5f} | {c['eval_slice_avg_pnl_after_cost']:.5f} | "
            f"{c['n_eval_kept']} |"
        )
    md.append("")
    (out_dir / "report.md").write_text("\n".join(md))
    log.info("report_written", path=str(out_dir / "report.md"))

    log.info("run_complete",
             horizons_run=len(per_horizon_results),
             promoted=sum(1 for r in per_horizon_results if r["promoted"]),
             total_elapsed_s=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
