"""One-shot bootstrap: pull S&P 500 OHLCV, build features, train XGBoost return predictor.

Usage (inside container):
    docker compose run --rm zeus-scheduler python -m scripts.bootstrap_train
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import structlog
from sqlalchemy import select

from zeus.config.settings import get_settings
from zeus.data.ingestion.ohlcv_ingester import OHLCVIngester
from zeus.data.ingestion.yfinance_client import YFinanceClient
from zeus.data.storage.database import OHLCVDaily, Position, get_session_factory
from zeus.features.pipeline import FeaturePipeline
from zeus.models.regime_detector import REGIME_PARAMS, RuleBasedRegimeDetector
from zeus.models.return_predictor import XGBReturnPredictor
from zeus.models.trainer import ModelTrainer
from zeus.portfolio.constructor import PortfolioConstructor
from zeus.risk.engine import RiskEngine
from zeus.risk.limits import RiskLimits
from zeus.risk.stop_logic import compute_initial_stops
from zeus.scheduler.market_schedule import next_trading_day, prev_trading_day
from zeus.signals.filter import SignalFilter
from zeus.signals.generator import SignalGenerator

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
log = structlog.get_logger("bootstrap_train")


# ── Tunables ─────────────────────────────────────────────────────────────────
N_TICKERS = 500           # full S&P 500 universe
HISTORY_YEARS = 6         # 5-year training window + 1y buffer for 200d-MA warmup
FEATURE_DAYS = 1260       # ~5 trading years of feature snapshots → training rows
LABEL_HORIZON = 5         # forward 5d return label (built into trainer)


def get_universe(client: YFinanceClient, n: int) -> list[str]:
    tickers = client.get_sp500_tickers()
    if not tickers:
        # Fallback to a hand-curated liquid mega-cap list if Wikipedia scrape fails.
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "AVGO", "JPM", "V", "WMT", "MA", "UNH", "XOM", "JNJ", "PG", "HD",
            "COST", "ABBV", "CVX", "MRK", "LLY", "ORCL", "BAC", "KO", "PEP",
            "CRM", "ADBE", "ACN", "TMO", "MCD", "NFLX", "CSCO", "AMD", "DIS",
            "ABT", "WFC", "TXN", "INTC", "QCOM", "DHR", "VZ", "PFE", "NKE",
            "PM", "MS", "GS", "RTX", "INTU",
        ]
    out = tickers[:n]
    # SPY is required by the regime detector; ensure it's always present.
    if "SPY" not in out:
        out.append("SPY")
    return out


def trading_days(start: date, end: date) -> list[date]:
    """Weekday list as a market-day proxy. Holidays drop out at feature time."""
    days: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def main() -> int:
    t0 = time.time()
    settings = get_settings()
    log.info("bootstrap_start", env=settings.environment, n_tickers=N_TICKERS, history_years=HISTORY_YEARS)

    yf = YFinanceClient()
    universe = get_universe(yf, N_TICKERS)
    log.info("universe_resolved", n=len(universe), head=universe[:10])

    today = date.today()
    history_start = today - timedelta(days=HISTORY_YEARS * 365 + 30)

    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        existing = session.execute(
            select(OHLCVDaily.symbol).distinct()
        ).scalars().all()
    existing_set = set(existing)
    missing = [s for s in universe if s not in existing_set]
    log.info("ohlcv_inventory", n_existing=len(existing_set), n_missing=len(missing))

    # ── 1. Backfill OHLCV ─────────────────────────────────────────────────────
    ingester = OHLCVIngester(broker=None, yf_client=yf)
    if missing or len(existing_set) < len(universe):
        log.info("backfill_begin", n_symbols=len(universe), start=str(history_start), end=str(today))
        counts = ingester.backfill_daily(universe, history_start, today, source="yfinance")
        log.info("backfill_done", **counts, elapsed_s=round(time.time() - t0, 1))
    else:
        log.info("ohlcv_already_present_skip_backfill")

    with SessionLocal() as session:
        n_rows = session.execute(
            select(OHLCVDaily.symbol).where(OHLCVDaily.symbol.in_(universe))
        ).all()
        log.info("ohlcv_row_count", n=len(n_rows))

    # ── 2. Compute features for the last FEATURE_DAYS sessions ────────────────
    pipeline = FeaturePipeline(feature_version="v1")
    feature_window_end = today - timedelta(days=LABEL_HORIZON + 2)  # leave room for forward labels
    feature_window_start = feature_window_end - timedelta(days=int(FEATURE_DAYS * 1.6))
    days = trading_days(feature_window_start, feature_window_end)
    log.info("feature_window", start=str(days[0]), end=str(days[-1]), n_days=len(days))

    with SessionLocal() as session:
        for d in days:
            try:
                pipeline.compute_features_for_date(session, d, universe)
            except Exception as exc:
                log.warning("feature_compute_failed", as_of=str(d), error=str(exc))

    # ── 3. Train ──────────────────────────────────────────────────────────────
    artifacts_path = settings.artifacts_path
    Path(artifacts_path).mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        trainer = ModelTrainer(session=session, artifacts_path=artifacts_path)
        train_start = days[0]
        train_end = days[-1]
        try:
            X, y, metadata = trainer.build_training_dataset(
                start=train_start, end=train_end, feature_version="v1"
            )
        except ValueError as exc:
            log.error("training_dataset_empty", error=str(exc))
            return 1

        log.info("training_dataset_built", n_rows=len(X), n_features=X.shape[1])
        result = trainer.train_and_evaluate(X, y, metadata)
        artifact_dir = trainer.save_model(
            model=result["model"],
            metrics=result["metrics"],
            metadata=metadata,
            passed_gates=result["passed_gates"],
            version_string=result["version_string"],
        )
        log.info(
            "training_complete",
            metrics=result["metrics"],
            passed_gates=result["passed_gates"],
            artifact=artifact_dir,
            total_elapsed_s=round(time.time() - t0, 1),
        )

    # ── 4. Compute features for the most recent trading day, build Monday plan ─
    plan_for = next_trading_day(today)
    snapshot_day = prev_trading_day(plan_for)
    log.info("planning_inputs", snapshot_day=str(snapshot_day), plan_for=str(plan_for))

    with SessionLocal() as session:
        try:
            pipeline.compute_features_for_date(session, snapshot_day, universe)
        except Exception as exc:
            log.error("snapshot_feature_compute_failed", as_of=str(snapshot_day), error=str(exc))
            return 2

    # Reload the freshly trained model to pick up the saved version_
    model = XGBReturnPredictor.load(Path(artifact_dir) / "model.joblib")
    model.version_ = result["version_string"]

    sig_gen = SignalGenerator(model=model)
    sig_filter = SignalFilter()
    limits = RiskLimits()
    risk_engine = RiskEngine(limits=limits, starting_capital=100_000.0)
    constructor = PortfolioConstructor(risk_engine=risk_engine, limits=limits)
    regime_detector = RuleBasedRegimeDetector()

    features_dir = os.path.join(artifacts_path, "features", snapshot_day.isoformat())
    features_df = pipeline.load_snapshot(features_dir)
    if features_df.empty:
        log.error("no_feature_snapshot_for_planning", path=features_dir)
        return 3

    # Need a feature_date column for SignalGenerator
    if "feature_date" not in features_df.columns and "ts" in features_df.columns:
        features_df["feature_date"] = features_df["ts"]

    # Regime
    with SessionLocal() as session:
        spy_rows = session.execute(
            select(OHLCVDaily.ts, OHLCVDaily.close)
            .where(OHLCVDaily.symbol == "SPY")
            .order_by(OHLCVDaily.ts.desc())
            .limit(220)
        ).all()
    if spy_rows:
        import pandas as pd
        spy = pd.Series({r[0]: r[1] for r in spy_rows}).sort_index()
        vix = pd.Series([15.0] * len(spy), index=spy.index)
        regime = regime_detector.detect(spy, vix)
    else:
        regime = "RANGE"
    log.info("regime_for_plan", regime=regime)

    # Inference + filter
    try:
        signals = sig_gen.generate(
            features_df=features_df, regime=regime, feature_version="v1",
        )
    except KeyError as exc:
        log.error("signal_generation_feature_mismatch", error=str(exc))
        return 4

    signals = sig_filter.apply(signals)
    log.info("signals_after_filter", n=len(signals))
    if signals.empty:
        log.warning("no_signals_after_filter_writing_empty_plan")

    # Enrich signals with last close price + dollar volume from the snapshot
    # so the portfolio constructor sees real prices instead of the $100 placeholder.
    price_cols = {"symbol": "symbol"}
    if "close" in features_df.columns:
        price_cols["close"] = "price"
    if "dollar_volume" in features_df.columns:
        price_cols["dollar_volume"] = "adv_usd_real"
    enrich = features_df[list(price_cols.keys())].rename(columns=price_cols)
    signals = signals.merge(enrich, on="symbol", how="left")
    if "price" in signals.columns:
        signals["price"] = signals["price"].astype(float)

    # Portfolio construction
    with SessionLocal() as session:
        from zeus.execution.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.alpaca_paper,
        )
        account = broker.get_account()
        current_positions = {
            r.symbol: {
                "shares": r.qty,
                "market_value": r.market_value or 0.0,
                "sector": None,
                "avg_entry_price": r.avg_entry_price,
            }
            for r in session.query(Position).all()
        }
        plan_dict = constructor.construct(
            signals=signals,
            current_positions=current_positions,
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            regime=regime,
            regime_params=REGIME_PARAMS[regime],
        )

    entries_list = []
    for pt in plan_dict.get("entries", []):
        stops = compute_initial_stops(
            entry_price=pt.price,
            atr=pt.price * 0.02,
            vol_annual=0.30,
        )
        entries_list.append({
            "symbol": pt.symbol,
            "shares": pt.shares,
            "target_price": pt.price,
            "sector": pt.sector,
            "adv_usd": pt.avg_daily_dollar_volume,
            "stop": stops.hard_stop,
            "signal_score": 0.0,
        })

    plan_doc = {
        "plan_date": plan_for.isoformat(),
        "regime": regime,
        "entries": entries_list,
        "exits": [pt.symbol for pt in plan_dict.get("exits", [])],
        "holds": plan_dict.get("holds", []),
        "model_version": getattr(model, "version_", "unknown"),
        "feature_version": "v1",
    }
    plan_path = os.path.join(
        artifacts_path, "knowledge", "session_plans", f"{plan_for.isoformat()}.json"
    )
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w") as f:
        json.dump(plan_doc, f, indent=2)
    log.info(
        "monday_plan_saved",
        path=plan_path,
        entries=len(entries_list),
        regime=regime,
        total_elapsed_s=round(time.time() - t0, 1),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
