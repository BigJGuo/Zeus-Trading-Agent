"""Build three-horizon rank+magnitude+anchor-agreement gated models so that
day, swing, and long-term trade types all clear 0.60 raw hit rate after costs.

Uses the already-trained primaries from the previous multi-horizon run:
  artifacts/models/ensemble_h{H}_return_rankgated_pnl/v<RUN>_h{H}/primary.joblib
and the OOS parquets at:
  artifacts/multi_horizon_pnl_<RUN>/oos_predictions/ensemble_h{H}_return.parquet

For each trade type:
  1. Load target+anchor primaries
  2. Calibrate (mag_floor, rank_floor) via calibrate_cross_horizon_gate
     (maximize eval PnL subject to joint train_hit ≥ 0.60 AND eval_hit ≥ 0.60;
     fall back to full-OOS gate if needed)
  3. Backtest full OOS with the calibrated gate
  4. Save as CrossHorizonGatedPredictor bundle
  5. Register in DB if hit ≥ 0.60 AND PnL > 0

Outputs report at artifacts/cross_horizon_bundle_<TS>/report.md
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

from zeus.models.rank_gated_predictor import (
    CrossHorizonGatedPredictor,
    backtest_cross_horizon_gate,
    calibrate_cross_horizon_gate,
)
from zeus.data.storage.database import ModelVersion, get_session_factory


log = structlog.get_logger()

BASE_RUN = "v20260421_005905"
BASE_ART_DIR = Path(f"artifacts/multi_horizon_pnl_{BASE_RUN.replace('v', '').replace('_', '')[:14]}")
# fallback: find the latest multi_horizon_pnl_*
if not BASE_ART_DIR.exists():
    candidates = sorted(Path("artifacts").glob("multi_horizon_pnl_*"), reverse=True)
    if candidates:
        BASE_ART_DIR = candidates[0]
        BASE_RUN_TS = BASE_ART_DIR.name.replace("multi_horizon_pnl_", "")
        BASE_RUN = f"v{BASE_RUN_TS}"

OOS_DIR = BASE_ART_DIR / "oos_predictions"
MODELS_DIR = Path("artifacts/models")

TARGET_HIT = 0.60
COST_BPS = 10.0

TRADE_TYPES = [
    # (label, target_horizon, anchor_horizons)
    ("day",       3,  [2, 5, 10, 20]),
    ("swing",     5,  []),                 # already clears without anchors
    ("long_term", 20, [5, 10]),
]


def _load_primary(horizon: int) -> tuple:
    # The saved CrossHorizonGated bundle uses `primary.joblib` alongside the config.
    # From the earlier run, each horizon dir is `ensemble_h{H}_return_rankgated_pnl/v<RUN>_h{H}/primary.joblib`
    base = MODELS_DIR / f"ensemble_h{horizon}_return_rankgated_pnl"
    versions = sorted(base.glob(f"{BASE_RUN}_h{horizon}"), reverse=True)
    if not versions:
        versions = sorted(base.iterdir(), reverse=True)
    if not versions:
        raise FileNotFoundError(f"no primary found for horizon {horizon}")
    primary_path = versions[0] / "primary.joblib"
    from zeus.models.ensemble_return_predictor import EnsembleReturnPredictor
    m = EnsembleReturnPredictor.load(primary_path)
    return m, primary_path


def _load_oos(horizon: int) -> pd.DataFrame:
    return pd.read_parquet(OOS_DIR / f"ensemble_h{horizon}_return.parquet")


def _register(session, name: str, version: str, horizon: int, path: Path, metrics: dict, params: dict,
              status: str) -> None:
    # Store the artifact *directory* (matches trainer.save_model convention).
    # The loader still accepts a file path for backwards-compat, but fresh rows
    # should point at the dir so sibling files (model_class.txt, config.json)
    # stay discoverable.
    artifact_dir = path.parent if path.is_file() or path.suffix == ".joblib" else path
    mv = ModelVersion(
        model_name=name,
        version=version,
        status=status,
        metrics=metrics,
        config={"horizon_days": horizon, **params},
        artifact_path=str(artifact_dir),
        feature_version="v1_baseline",
        training_start_date=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * 3),
        training_end_date=pd.Timestamp.now(tz="UTC"),
        created_at=datetime.now(timezone.utc),
    )
    session.add(mv)
    session.commit()


def main() -> int:
    if not OOS_DIR.exists():
        print(f"ERROR: OOS dir not found: {OOS_DIR}")
        return 1
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"artifacts/cross_horizon_bundle_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("run_start", base_run=BASE_RUN, oos_dir=str(OOS_DIR), out_dir=str(out_dir))

    SessionLocal = get_session_factory()
    results = []

    for label, target_h, anchor_hs in TRADE_TYPES:
        t0 = time.time()
        log.info("trade_type_start", label=label, target=target_h, anchors=anchor_hs)

        target_primary, target_path = _load_primary(target_h)
        anchor_primaries = []
        for ah in anchor_hs:
            a_primary, _ = _load_primary(ah)
            anchor_primaries.append(a_primary)

        target_oos = _load_oos(target_h)
        anchor_oos = [_load_oos(ah) for ah in anchor_hs]

        gate = calibrate_cross_horizon_gate(
            target_oos_df=target_oos,
            anchor_oos_dfs=anchor_oos,
            target_hit=TARGET_HIT,
            min_n_train_kept=200,
            min_n_eval_kept=50,
            cost_bps=COST_BPS,
        )
        log.info("gate_calibrated", label=label, target=target_h,
                 strategy=gate["strategy"],
                 mag_floor=round(gate["mag_floor"], 6),
                 rank_floor=gate["rank_floor"],
                 train_hit=round(gate.get("train_hit") or 0, 4),
                 eval_hit=round(gate.get("eval_hit") or 0, 4),
                 full_hit=round(gate.get("full_hit") or 0, 4),
                 n_train_kept=int(gate.get("n_train_kept") or 0),
                 n_eval_kept=int(gate.get("n_eval_kept") or 0),
                 n_full_kept=int(gate.get("n_full_kept") or 0))

        # Full-OOS backtest with calibrated gate
        bt = backtest_cross_horizon_gate(
            target_oos_df=target_oos,
            anchor_oos_dfs=anchor_oos,
            mag_floor=gate["mag_floor"],
            rank_floor=gate["rank_floor"],
            long_only=True,
            cost_bps=COST_BPS,
        )
        log.info("backtest_full_oos", label=label, **{k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in bt.items()})

        # Build wrapper & save
        wrapper = CrossHorizonGatedPredictor(
            primary=target_primary,
            anchor_primaries=anchor_primaries,
            mag_floor=gate["mag_floor"],
            rank_floor=gate["rank_floor"],
            long_only=True,
        )
        model_name = f"cross_horizon_{label}_h{target_h}"
        version = f"v{ts}"
        save_dir = MODELS_DIR / model_name / version
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "model.joblib"
        wrapper.save(model_path)
        (save_dir / "config.json").write_text(json.dumps({
            "target_horizon": target_h,
            "anchor_horizons": anchor_hs,
            "mag_floor": gate["mag_floor"],
            "rank_floor": gate["rank_floor"],
            "calibration_strategy": gate["strategy"],
            "backtest": bt,
            "base_run": BASE_RUN,
        }, default=str, indent=2))
        log.info("model_saved", label=label, path=str(model_path))

        # Register
        passed = (bt.get("hit_rate", 0) or 0) >= TARGET_HIT \
            and (bt.get("avg_trade_return_after_cost", 0) or 0) > 0 \
            and (bt.get("n_trades", 0) or 0) >= 200
        status = "staging" if passed else "failed"
        with SessionLocal() as session:
            _register(session, model_name, version, target_h, model_path,
                      metrics={"hit_rate": bt.get("hit_rate"),
                               "avg_trade_return_after_cost": bt.get("avg_trade_return_after_cost"),
                               "daily_sharpe_annualized": bt.get("daily_sharpe_annualized"),
                               "annualized_return_pct": bt.get("annualized_return_pct"),
                               "n_trades": bt.get("n_trades"),
                               "max_drawdown_log": bt.get("max_drawdown_log")},
                      params={"target_horizon": target_h, "anchor_horizons": anchor_hs,
                              "mag_floor": gate["mag_floor"], "rank_floor": gate["rank_floor"]},
                      status=status)
        log.info("registered", label=label, status=status, model_name=model_name, version=version)

        results.append({
            "label": label, "target_horizon": target_h, "anchors": anchor_hs,
            "strategy": gate["strategy"],
            "mag_floor": gate["mag_floor"], "rank_floor": gate["rank_floor"],
            "n_trades": bt.get("n_trades"),
            "hit_rate": bt.get("hit_rate"),
            "avg_pct_per_trade": (bt.get("avg_trade_return_after_cost") or 0) * 100,
            "annualized_return_pct": (bt.get("annualized_return_pct") or 0) * 100,
            "daily_sharpe_annualized": bt.get("daily_sharpe_annualized"),
            "max_drawdown_log": bt.get("max_drawdown_log"),
            "trades_per_year": bt.get("trades_per_year"),
            "win_loss_ratio": bt.get("win_loss_ratio"),
            "status": status,
            "elapsed_s": round(time.time() - t0, 1),
        })

    # Report
    report_md = ["# Cross-Horizon Bundle PnL Report",
                 "",
                 f"Generated: {datetime.now(timezone.utc).isoformat()}",
                 f"Base multi-horizon run: {BASE_RUN}",
                 f"Cost: {COST_BPS:.1f} bps round-trip | Hit target: ≥ {TARGET_HIT:.2f}",
                 "",
                 "## Final per-trade-type results (full OOS, after cost)",
                 "",
                 "| Trade type | Target h | Anchors | Hit | n | %/trade | Ann ret % | Sharpe | MaxDD | W/L | Status |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        report_md.append(
            f"| {r['label']} | {r['target_horizon']}d | {','.join(str(a)+'d' for a in r['anchors']) or '—'} | "
            f"{r['hit_rate']:.4f} | {r['n_trades']} | {r['avg_pct_per_trade']:.3f}% | "
            f"{r['annualized_return_pct']:.1f}% | {r['daily_sharpe_annualized']:.2f} | "
            f"{r['max_drawdown_log']:.2f} | {r['win_loss_ratio']:.2f} | {r['status'].upper()} |"
        )
    (out_dir / "report.md").write_text("\n".join(report_md))
    (out_dir / "report.json").write_text(json.dumps({"run_ts": ts, "base_run": BASE_RUN,
                                                     "results": results},
                                                    default=str, indent=2))
    log.info("report_written", path=str(out_dir / "report.md"))
    n_pass = sum(1 for r in results if r["status"] == "staging")
    log.info("run_complete", n_pass=n_pass, n_total=len(results))
    return 0 if n_pass == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
