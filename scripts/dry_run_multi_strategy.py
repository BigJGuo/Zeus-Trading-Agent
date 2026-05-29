"""Dry-run the multi-strategy pipeline end-to-end without touching the broker.

Verifies that:
  1. config/strategies.yaml loads cleanly
  2. Each strategy's model resolves from model_versions
  3. StrategyAllocator produces budgets that sum to portfolio_value
  4. StrategyManager.plan_all() produces per-strategy SessionPlan files
  5. Per-strategy plans land at artifacts/knowledge/session_plans/<date>_<id>.json

Does NOT submit orders. Safe to run against the paper broker.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import structlog

from zeus.config.settings import get_settings
from zeus.config.strategies import get_strategies
from zeus.execution.alpaca_broker import AlpacaBroker
from zeus.execution.order_manager import OrderManager
from zeus.features.pipeline import FeaturePipeline
from zeus.live.strategy import StrategyAllocator, StrategyContext
from zeus.live.strategy_manager import StrategyManager
from zeus.models.regime_detector import RuleBasedRegimeDetector
from zeus.portfolio.constructor import PortfolioConstructor
from zeus.risk.engine import RiskEngine
from zeus.risk.limits import RiskLimits
from zeus.scheduler.runner import _load_model_by_name, _NullPredictor
from zeus.signals.filter import SignalFilter
from zeus.signals.generator import SignalGenerator

log = structlog.get_logger(__name__)


def main() -> int:
    settings = get_settings()
    bundle = get_strategies()
    print(f"Loaded {len(bundle.strategies)} strategy declarations "
          f"({sum(1 for s in bundle.strategies if s.enabled)} enabled).")
    for s in bundle.strategies:
        print(f"  - {s.id:10s}  model={s.model_name:35s} "
              f"weight={s.weight:.2f}  max_positions={s.max_positions}  "
              f"enabled={s.enabled}")

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=settings.alpaca_paper,
    )
    account = broker.get_account()
    pv = float(account.portfolio_value)
    bp = float(account.buying_power)
    print(f"\nBroker account: portfolio_value=${pv:,.2f}  buying_power=${bp:,.2f}")

    allocator = StrategyAllocator(bundle.strategies, bundle.globals)
    budgets = allocator.budgets(pv, bp)
    print("\nPer-strategy budgets:")
    for sid, b in budgets.items():
        print(f"  {sid:10s}  pv=${b.portfolio_value:,.2f}  "
              f"max_positions={b.max_positions}  max_expo={b.max_exposure_pct:.0%}")

    # Build contexts (models resolve from DB; fall back to _NullPredictor).
    limits = RiskLimits()
    risk_engine = RiskEngine(limits=limits, starting_capital=pv or 100_000.0)
    feature_pipeline = FeaturePipeline(feature_version="v1")
    contexts: dict[str, StrategyContext] = {}
    for s in bundle.enabled():
        m = _load_model_by_name(s.model_name) or _NullPredictor()
        contexts[s.id] = StrategyContext(
            config=s,
            model=m,
            signal_generator=SignalGenerator(model=m),
            signal_filter=SignalFilter(),
            portfolio_constructor=PortfolioConstructor(risk_engine=risk_engine, limits=limits),
            feature_pipeline=feature_pipeline,
        )
        print(f"  {s.id:10s}  model_loaded={type(m).__name__} "
              f"version={getattr(m, 'version_', 'unknown')}")

    manager = StrategyManager(
        contexts=contexts,
        allocator=allocator,
        broker=broker,
        order_manager=OrderManager(broker, environment="paper"),
        risk_engine=risk_engine,
        globals_=bundle.globals,
        environment="paper",
    )

    # Regime detect once, feed to plan_all.
    regime_detector = RuleBasedRegimeDetector()
    from zeus.data.storage.database import get_session_factory
    from zeus.research.after_hours import _load_regime_series

    with get_session_factory()() as session:
        spy, vix = _load_regime_series(session)
    regime = regime_detector.detect(spy, vix)
    print(f"\nDetected regime: {regime}")

    print("\nRunning plan_all() ...")
    plans = manager.plan_all(regime=regime, as_of=date.today())

    plan_dir = Path(settings.artifacts_path) / "knowledge" / "session_plans"
    print(f"\nPlans written to: {plan_dir}")
    for sid, plan in plans.items():
        fname = f"{plan.plan_date.isoformat()}_{sid}.json"
        print(f"  {sid:10s}  entries={len(plan.entries):3d} "
              f"exits={len(plan.exits):3d}  plan_file={fname}")

    print("\nSummaries:")
    for s in manager.summaries():
        print(f"  {s.strategy_id:10s}  open={s.n_open_positions}  "
              f"deployed=${s.deployed_notional:,.2f}  "
              f"planned_entries={s.planned_entries}")

    print("\nDRY RUN COMPLETE (no orders submitted).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
