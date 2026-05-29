"""Per-strategy execution context + capital allocator.

A `StrategyContext` bundles everything needed to run one trader agent:
  - its loaded model
  - its signal generator, filter, portfolio constructor
  - its per-strategy config (horizon, weights, max_positions)
  - its own SessionPlan

The `StrategyAllocator` enforces per-strategy budgets on top of the shared
portfolio value: each strategy sees a scaled `portfolio_value * weight` + a
`buying_power * weight` when building its plan. Global caps
(max_total_exposure_pct, global_max_position_pct) are enforced at order-submit
time inside StrategyManager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import structlog

from zeus.config.strategies import GlobalLimits, StrategyConfig
from zeus.live.trading_loop import SessionPlan

if TYPE_CHECKING:
    from zeus.features.pipeline import FeaturePipeline
    from zeus.models.base import BaseModel
    from zeus.portfolio.constructor import PortfolioConstructor
    from zeus.signals.filter import SignalFilter
    from zeus.signals.generator import SignalGenerator

log = structlog.get_logger(__name__)


@dataclass
class StrategyContext:
    """All the per-strategy state the trading loop needs at session planning /
    execution time. One instance per enabled strategy."""
    config: StrategyConfig
    model: "BaseModel"
    signal_generator: "SignalGenerator"
    signal_filter: "SignalFilter"
    portfolio_constructor: "PortfolioConstructor"
    feature_pipeline: "FeaturePipeline"
    current_plan: Optional[SessionPlan] = field(default=None)

    @property
    def strategy_id(self) -> str:
        return self.config.id


@dataclass
class StrategyBudget:
    """Per-strategy slice of the shared portfolio value. Fed into
    PortfolioConstructor.construct() as portfolio_value + buying_power."""
    strategy_id: str
    portfolio_value: float
    buying_power: float
    max_positions: int
    max_position_pct: float
    max_exposure_pct: float


class StrategyAllocator:
    """Splits a shared portfolio across N enabled strategies by static weights.

    v1 is static weights from config. v2 (overseer phase) can subclass this
    and re-weight based on per-strategy Sharpe / drawdown.
    """

    def __init__(self, strategies: list[StrategyConfig], globals_: GlobalLimits):
        self._strategies = [s for s in strategies if s.enabled]
        self._globals = globals_
        total = sum(s.weight for s in self._strategies)
        if total <= 0:
            raise ValueError("no enabled strategies with positive weight")
        # Normalize so allocator is robust to small float drift in the YAML.
        self._weights: Dict[str, float] = {
            s.id: s.weight / total for s in self._strategies
        }

    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def budgets(self, portfolio_value: float, buying_power: float) -> Dict[str, StrategyBudget]:
        """Compute per-strategy budgets. `portfolio_value` and `buying_power`
        come from the broker account once per planning cycle."""
        out: Dict[str, StrategyBudget] = {}
        for s in self._strategies:
            w = self._weights[s.id]
            out[s.id] = StrategyBudget(
                strategy_id=s.id,
                portfolio_value=portfolio_value * w,
                buying_power=buying_power * w,
                max_positions=s.max_positions,
                max_position_pct=s.max_position_pct,
                max_exposure_pct=s.max_exposure_pct,
            )
        log.info(
            "budgets_computed",
            portfolio_value=portfolio_value,
            allocations={k: round(v.portfolio_value, 2) for k, v in out.items()},
        )
        return out

    def admits_global_overlap(
        self,
        symbol: str,
        proposed_notional: float,
        existing_by_strategy: Dict[str, float],
        portfolio_value: float,
    ) -> bool:
        """Check if adding `proposed_notional` of `symbol` under a new strategy
        would breach the aggregate per-symbol cap (global_max_position_pct)."""
        aggregate = sum(existing_by_strategy.values()) + proposed_notional
        cap = portfolio_value * self._globals.global_max_position_pct
        return aggregate <= cap

    def admits_total_exposure(self, projected_deployed: float, portfolio_value: float) -> bool:
        cap = portfolio_value * self._globals.max_total_exposure_pct
        return projected_deployed <= cap
