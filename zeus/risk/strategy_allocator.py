"""Overseer-driven StrategyAllocator.

Phase 1 shipped a static-weight allocator in `zeus/live/strategy.py`. This
module formalizes it with an overridable weight vector the overseer can
nudge on a weekly cadence, subject to hard bounds from `config/strategies.yaml`:

  * Each strategy stays within `weight_floor` / `weight_ceiling` (derived
    from the base weight ± `max_weekly_shift`).
  * Weekly shifts are capped at `max_weekly_shift` (±10pp by default) so
    runaway shifts can't concentrate capital.
  * Halted agents are excluded from the allocation — their weight is
    redistributed proportionally to the remaining agents.

The `StrategyManager` reads `effective_budgets()` each planning cycle; the
overseer calls `reallocate()` after its weekly review.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog

from zeus.config.strategies import GlobalLimits, StrategyBundle, StrategyConfig
from zeus.live.strategy import StrategyAllocator as _StaticAllocator
from zeus.live.strategy import StrategyBudget

log = structlog.get_logger(__name__)


DEFAULT_MAX_WEEKLY_SHIFT = 0.10    # ±10 percentage points per week


@dataclass
class AllocationUpdate:
    """Record of one reallocation event. Journaled as an overseer decision."""
    ts: datetime
    prior_weights: Dict[str, float]
    new_weights: Dict[str, float]
    reason: str
    rejected_reason: Optional[str] = None


class OverseerStrategyAllocator(_StaticAllocator):
    """StrategyAllocator that the overseer can reweight within bounds.

    Keeps the base (config) weight around as the anchor for the shift cap so
    the overseer can't drift monotonically away from the operator's intent
    over many weeks.
    """

    def __init__(
        self,
        strategies: List[StrategyConfig],
        globals_: GlobalLimits,
        *,
        max_weekly_shift: float = DEFAULT_MAX_WEEKLY_SHIFT,
        risk_engine=None,
    ):
        super().__init__(strategies, globals_)
        self._enabled_by_id = {s.id: s for s in strategies if s.enabled}
        self._base_weights: Dict[str, float] = dict(self._weights)
        self._max_weekly_shift = max_weekly_shift
        self._risk_engine = risk_engine
        self._history: List[AllocationUpdate] = []

    # ─── Properties ──────────────────────────────────────────────────────────
    @property
    def history(self) -> List[AllocationUpdate]:
        return list(self._history)

    def base_weights(self) -> Dict[str, float]:
        return dict(self._base_weights)

    def effective_weights(self) -> Dict[str, float]:
        """Current weights with halted agents pulled to 0 and the remainder
        renormalized. This is the right input to `budgets(...)` at plan time."""
        halted = self._halted_agent_ids()
        if not halted:
            return dict(self._weights)
        active = {k: v for k, v in self._weights.items() if k not in halted}
        total = sum(active.values())
        if total <= 0:
            return {k: 0.0 for k in self._weights}
        out = {k: 0.0 for k in self._weights}
        out.update({k: (v / total) for k, v in active.items()})
        return out

    def effective_budgets(
        self, portfolio_value: float, buying_power: float,
    ) -> Dict[str, StrategyBudget]:
        """Same shape as the parent `budgets(...)` but using effective weights."""
        weights = self.effective_weights()
        out: Dict[str, StrategyBudget] = {}
        for sid, cfg in self._enabled_by_id.items():
            w = weights.get(sid, 0.0)
            out[sid] = StrategyBudget(
                strategy_id=sid,
                portfolio_value=portfolio_value * w,
                buying_power=buying_power * w,
                max_positions=cfg.max_positions,
                max_position_pct=cfg.max_position_pct,
                max_exposure_pct=cfg.max_exposure_pct,
            )
        return out

    # ─── Overseer-driven reallocation ────────────────────────────────────────
    def reallocate(
        self, proposed: Dict[str, float], *, reason: str,
    ) -> AllocationUpdate:
        """Apply the overseer's proposed weights, subject to bounds.

        Rejects (no state change) on:
          * unknown strategy_id
          * weights not summing to ~1
          * any per-strategy shift from base exceeds `max_weekly_shift`
        """
        update = self._validate_proposed(proposed, reason=reason)
        if update.rejected_reason is None:
            self._weights = dict(update.new_weights)
            log.info(
                "allocator_reweighted",
                reason=reason, new_weights=update.new_weights,
            )
        else:
            log.warning(
                "allocator_reweight_rejected",
                reason=reason, rejected_reason=update.rejected_reason,
            )
        self._history.append(update)
        return update

    def _validate_proposed(
        self, proposed: Dict[str, float], *, reason: str,
    ) -> AllocationUpdate:
        prior = dict(self._weights)
        now = datetime.now(timezone.utc)

        unknown = set(proposed) - set(self._enabled_by_id)
        if unknown:
            return AllocationUpdate(
                ts=now, prior_weights=prior, new_weights=prior, reason=reason,
                rejected_reason=f"unknown strategy ids: {sorted(unknown)}",
            )

        # Fill in any missing sid with its prior weight so a partial overseer
        # proposal doesn't zero out a strategy it didn't mention — the
        # omission means "leave this one alone."
        new_weights: Dict[str, float] = {
            sid: proposed.get(sid, prior.get(sid, 0.0))
            for sid in self._enabled_by_id
        }

        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.02:
            return AllocationUpdate(
                ts=now, prior_weights=prior, new_weights=prior, reason=reason,
                rejected_reason=f"weights sum to {total:.4f}, expected ~1.0",
            )

        for sid, w in new_weights.items():
            base = self._base_weights.get(sid, 0.0)
            if abs(w - base) > self._max_weekly_shift + 1e-9:
                return AllocationUpdate(
                    ts=now, prior_weights=prior, new_weights=prior, reason=reason,
                    rejected_reason=(
                        f"{sid!r} shift |{w:.3f} - {base:.3f}| exceeds cap "
                        f"{self._max_weekly_shift:.3f}"
                    ),
                )
            if w < 0.0:
                return AllocationUpdate(
                    ts=now, prior_weights=prior, new_weights=prior, reason=reason,
                    rejected_reason=f"{sid!r} weight < 0",
                )

        # Renormalize to handle floating-point slop.
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        return AllocationUpdate(
            ts=now, prior_weights=prior, new_weights=new_weights, reason=reason,
        )

    # ─── Integration with RiskEngine.halt_agent ──────────────────────────────
    def _halted_agent_ids(self) -> List[str]:
        if self._risk_engine is None:
            return []
        try:
            return list(self._risk_engine.halted_agents().keys())
        except AttributeError:
            return []


def from_bundle(
    bundle: StrategyBundle, *,
    max_weekly_shift: float = DEFAULT_MAX_WEEKLY_SHIFT,
    risk_engine=None,
) -> OverseerStrategyAllocator:
    """Convenience constructor from a loaded StrategyBundle."""
    return OverseerStrategyAllocator(
        strategies=bundle.strategies,
        globals_=bundle.globals,
        max_weekly_shift=max_weekly_shift,
        risk_engine=risk_engine,
    )
