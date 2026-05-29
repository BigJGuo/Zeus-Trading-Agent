"""After-hours research pipeline: data refresh → features → signals → next-day plan.

Invoked by scheduled jobs in zeus.scheduler.jobs. Stitches together all the
independent subsystems (data ingestion, feature pipeline, model inference,
portfolio construction) and persists the next-session plan to disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import structlog

from zeus.config.settings import get_settings
from zeus.data.ingestion.ohlcv_ingester import OHLCVIngester
from zeus.data.ingestion.universe_builder import UniverseBuilder
from zeus.data.ingestion.yfinance_client import YFinanceClient
from zeus.data.storage.database import Position, get_session_factory
from zeus.execution.alpaca_broker import AlpacaBroker
from zeus.features.pipeline import FeaturePipeline
from zeus.live.trading_loop import SessionPlan
from zeus.models.base import BaseModel
from zeus.models.regime_detector import REGIME_PARAMS, RuleBasedRegimeDetector
from zeus.models.return_predictor import XGBReturnPredictor
from zeus.portfolio.constructor import PortfolioConstructor
from zeus.risk.engine import RiskEngine
from zeus.risk.limits import RiskLimits
from zeus.risk.stop_logic import compute_initial_stops
from zeus.scheduler.market_schedule import next_trading_day, prev_trading_day
from zeus.signals.filter import SignalFilter
from zeus.signals.generator import SignalGenerator

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)


@dataclass
class AfterHoursContext:
    broker: AlpacaBroker
    yf_client: YFinanceClient
    ingester: OHLCVIngester
    feature_pipeline: FeaturePipeline
    model: BaseModel
    signal_generator: SignalGenerator
    signal_filter: SignalFilter
    portfolio_constructor: PortfolioConstructor
    regime_detector: RuleBasedRegimeDetector
    risk_engine: RiskEngine


def run_data_refresh(ctx: AfterHoursContext, as_of: Optional[date] = None) -> None:
    """Step 1: pull latest daily bars for universe, upsert into DB."""
    as_of = as_of or date.today()
    log.info("after_hours_data_refresh", as_of=str(as_of))

    builder = UniverseBuilder(ctx.yf_client)
    universe = builder.build_base_universe()

    with get_session_factory()() as session:
        result = ctx.ingester.update_daily(universe, as_of)
        log.info("data_refresh_complete", **result)
        builder.refresh_universe_snapshot(session, as_of)


def run_feature_engineering(ctx: AfterHoursContext, as_of: Optional[date] = None) -> None:
    """Step 2: compute features for universe, save to DB + parquet."""
    as_of = as_of or date.today()
    log.info("after_hours_feature_engineering", as_of=str(as_of))

    with get_session_factory()() as session:
        from zeus.data.storage.database import UniverseSnapshot
        from sqlalchemy import select

        rows = session.execute(
            select(UniverseSnapshot.symbol)
            .where(UniverseSnapshot.snapshot_date == datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc))
            .where(UniverseSnapshot.passes_filter.is_(True))
        ).all()
        symbols = [r[0] for r in rows]
        if not symbols:
            log.warning("no_filtered_symbols", as_of=str(as_of))
            return
        df = ctx.feature_pipeline.compute_features_for_date(session, as_of, symbols)
        log.info("feature_engineering_complete", n_rows=len(df))


def run_next_session_planning_for_strategy(
    ctx,  # zeus.live.strategy.StrategyContext
    budget,  # zeus.live.strategy.StrategyBudget
    broker,
    regime: str,
    as_of: Optional[date] = None,
) -> SessionPlan:
    """Build a next-session plan for one strategy using its own model +
    portfolio constructor + the allocator-provided budget.

    Mirrors the single-strategy `run_next_session_planning` but scopes
    portfolio_value / buying_power / max_positions to this strategy's slice
    via the `budget` argument. Current positions are filtered to only those
    this strategy owns.
    """
    from zeus.data.storage.database import Position as _Position

    as_of = as_of or date.today()
    plan_for = next_trading_day(as_of)
    log.info(
        "strategy_planning",
        strategy_id=ctx.strategy_id, as_of=str(as_of), plan_for=str(plan_for),
    )

    with get_session_factory()() as session:
        features_path = _resolve_features_path(as_of)
        features_df = ctx.feature_pipeline.load_snapshot(features_path)

        signals = ctx.signal_generator.generate(
            features_df=features_df,
            regime=regime,
            feature_version=ctx.feature_pipeline.feature_version,
        )
        signals = ctx.signal_filter.apply(signals)

        # Only this strategy's positions count against its own max_positions cap.
        current_positions = {
            r.symbol: {
                "shares": int(r.strategy_shares or r.qty),
                "market_value": float(r.market_value or 0.0),
                "sector": None,
                "avg_entry_price": float(r.avg_entry_price),
            }
            for r in session.query(_Position)
            .filter(_Position.strategy_id == ctx.strategy_id)
            .all()
        }

        regime_params = {
            "max_positions": budget.max_positions,
            "max_exposure_pct": budget.max_exposure_pct,
        }
        plan_dict = ctx.portfolio_constructor.construct(
            signals=signals,
            current_positions=current_positions,
            portfolio_value=budget.portfolio_value,
            buying_power=budget.buying_power,
            regime=regime,
            regime_params=regime_params,
        )

        from zeus.risk.stop_logic import compute_initial_stops
        signal_scores = {
            row["symbol"]: float(row.get("expected_return") or 0.0)
            for _, row in signals.iterrows()
        }
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
                "signal_score": signal_scores.get(pt.symbol, 0.0),
                "strategy_id": ctx.strategy_id,
            })

        plan = SessionPlan(
            plan_date=plan_for,
            regime=regime,
            entries=entries_list,
            exits=[pt.symbol for pt in plan_dict.get("exits", [])],
            holds=plan_dict.get("holds", []),
            model_version=getattr(ctx.model, "version_", "unknown"),
            feature_version=ctx.feature_pipeline.feature_version,
        )

    _save_plan_for_strategy(plan, ctx.strategy_id)
    log.info(
        "strategy_plan_saved",
        strategy_id=ctx.strategy_id, entries=len(plan.entries),
    )
    return plan


def _resolve_features_path(as_of: date) -> str:
    """Return the newest features/ directory at or before as_of.

    When run_next_session_planning fires pre-market on a trading day, today's
    features haven't been computed yet (feature_engineering runs post-close at
    17:00 ET). Fall back to the most recent prior trading day whose snapshot
    exists on disk so the planner can still emit a tomorrow-plan.
    """
    base = os.path.join(get_settings().artifacts_path, "features")
    candidate = os.path.join(base, as_of.isoformat())
    if os.path.isdir(candidate):
        return candidate
    if not os.path.isdir(base):
        return candidate
    # Pick the latest snapshot directory not exceeding as_of.
    target_iso = as_of.isoformat()
    snapshots = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d <= target_iso
    )
    if snapshots:
        return os.path.join(base, snapshots[-1])
    return candidate


def _save_plan_for_strategy(plan: SessionPlan, strategy_id: str) -> str:
    import json
    path = os.path.join(
        get_settings().artifacts_path,
        "knowledge", "session_plans",
        f"{plan.plan_date.isoformat()}_{strategy_id}.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2)
    return path


def run_next_session_planning(ctx: AfterHoursContext, as_of: Optional[date] = None) -> SessionPlan:
    """Step 3: run model, build next-day portfolio, save plan."""
    as_of = as_of or date.today()
    plan_for = next_trading_day(as_of)
    log.info("after_hours_planning", as_of=str(as_of), plan_for=str(plan_for))

    with get_session_factory()() as session:
        features_path = _resolve_features_path(as_of)
        features_df = ctx.feature_pipeline.load_snapshot(features_path)

        spy_close, vix_close = _load_regime_series(session)
        regime = ctx.regime_detector.detect(spy_close, vix_close)
        log.info("regime_detected", regime=regime)

        signals = ctx.signal_generator.generate(
            features_df=features_df,
            regime=regime,
            feature_version=ctx.feature_pipeline.feature_version,
        )
        signals = ctx.signal_filter.apply(signals)
        if signals.empty:
            log.warning("no_signals_after_filter")

        account = ctx.broker.get_account()
        current_positions = {
            r.symbol: {
                "shares": r.qty,
                "market_value": r.market_value or 0.0,
                "sector": None,
                "avg_entry_price": r.avg_entry_price,
            }
            for r in session.query(Position).all()
        }

        plan_dict = ctx.portfolio_constructor.construct(
            signals=signals,
            current_positions=current_positions,
            portfolio_value=float(account.portfolio_value or 0),
            buying_power=float(account.buying_power or 0),
            regime=regime,
            regime_params=REGIME_PARAMS[regime],
        )

        signal_scores = {
            row["symbol"]: float(row.get("expected_return") or 0.0)
            for _, row in signals.iterrows()
        }
        entries_list = []
        for pt in plan_dict.get("entries", []):
            stops = compute_initial_stops(
                entry_price=pt.price,
                atr=pt.price * 0.02,  # placeholder — real ATR attached by signal gen
                vol_annual=0.30,
            )
            entries_list.append({
                "symbol": pt.symbol,
                "shares": pt.shares,
                "target_price": pt.price,
                "sector": pt.sector,
                "adv_usd": pt.avg_daily_dollar_volume,
                "stop": stops.hard_stop,
                "signal_score": signal_scores.get(pt.symbol, 0.0),
            })

        plan = SessionPlan(
            plan_date=plan_for,
            regime=regime,
            entries=entries_list,
            exits=[pt.symbol for pt in plan_dict.get("exits", [])],
            holds=plan_dict.get("holds", []),
            model_version=getattr(ctx.model, "version_", "unknown"),
            feature_version=ctx.feature_pipeline.feature_version,
        )

    path = _save_plan(plan)
    log.info("plan_saved", path=path, entries=len(plan.entries))
    return plan


def run_nightly_retrain(ctx: AfterHoursContext, as_of: Optional[date] = None) -> dict:
    """Step 4 (Sunday only): retrain model. Delegate to zeus.models.trainer."""
    from zeus.models.trainer import ModelTrainer

    as_of = as_of or date.today()
    training_start = as_of - timedelta(days=730)
    training_end = as_of - timedelta(days=1)
    log.info("nightly_retrain_start", start=str(training_start), end=str(training_end))

    with get_session_factory()() as session:
        trainer = ModelTrainer(session, artifacts_path=get_settings().artifacts_path)
        X, y, meta = trainer.build_training_dataset(
            start=training_start, end=training_end, feature_version=ctx.feature_pipeline.feature_version,
        )
        result = trainer.train_and_evaluate(X, y, meta)
        path = trainer.save_model(result["model"], result["metrics"], meta, result["passed_gates"])
        log.info("nightly_retrain_complete", path=path, passed=result["passed_gates"])
        return result


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _save_plan(plan: SessionPlan) -> str:
    import json
    path = os.path.join(
        get_settings().artifacts_path,
        "knowledge", "session_plans", f"{plan.plan_date.isoformat()}.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2)
    return path


def _load_regime_series(session: "Session"):
    from zeus.data.storage.database import OHLCVDaily
    from sqlalchemy import select
    import pandas as pd

    def _series(symbol: str) -> pd.Series:
        rows = session.execute(
            select(OHLCVDaily.ts, OHLCVDaily.close)
            .where(OHLCVDaily.symbol == symbol)
            .order_by(OHLCVDaily.ts.desc())
            .limit(220)
        ).all()
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series({r[0]: r[1] for r in rows}).sort_index()
        return s

    spy = _series("SPY")
    vix = _series("^VIX")
    if vix.empty:
        # Fallback: a flat VIX of 15 if not ingested
        vix = pd.Series([15.0] * len(spy), index=spy.index) if not spy.empty else pd.Series([15.0])
    return spy, vix
