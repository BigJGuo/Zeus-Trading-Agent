"""OverseerStrategyAllocator: reweighting, bounds, halt redistribution."""
from __future__ import annotations

from typing import Dict

import pytest

from zeus.config.strategies import GlobalLimits, StrategyConfig
from zeus.risk.strategy_allocator import OverseerStrategyAllocator


def _cfg(id_: str, weight: float) -> StrategyConfig:
    return StrategyConfig(
        id=id_,
        model_name=f"cross_horizon_{id_}",
        horizon_days=3,
        max_hold_days=5,
        weight=weight,
        max_positions=5,
        max_position_pct=0.10,
        max_exposure_pct=0.30,
        enabled=True,
    )


def _alloc(risk_engine=None) -> OverseerStrategyAllocator:
    strategies = [_cfg("day", 0.30), _cfg("swing", 0.40), _cfg("long_term", 0.30)]
    return OverseerStrategyAllocator(
        strategies=strategies,
        globals_=GlobalLimits(),
        max_weekly_shift=0.10,
        risk_engine=risk_engine,
    )


def test_base_weights_and_effective_weights_match_at_construction():
    a = _alloc()
    w = a.effective_weights()
    assert pytest.approx(w["day"], abs=1e-6) == 0.30
    assert pytest.approx(w["swing"], abs=1e-6) == 0.40
    assert pytest.approx(w["long_term"], abs=1e-6) == 0.30


def test_reallocate_within_bounds_succeeds():
    a = _alloc()
    upd = a.reallocate({"day": 0.25, "swing": 0.45, "long_term": 0.30},
                       reason="swing outperforming")
    assert upd.rejected_reason is None
    w = a.effective_weights()
    assert pytest.approx(w["day"], abs=1e-6) == 0.25
    assert pytest.approx(w["swing"], abs=1e-6) == 0.45
    assert pytest.approx(w["long_term"], abs=1e-6) == 0.30


def test_reallocate_shift_exceeding_cap_rejected():
    a = _alloc()
    upd = a.reallocate(
        {"day": 0.10, "swing": 0.60, "long_term": 0.30},  # +20pp swing, -20pp day
        reason="extreme",
    )
    assert upd.rejected_reason is not None
    # Prior weights retained (rejected).
    assert a.effective_weights()["day"] == pytest.approx(0.30)


def test_reallocate_wrong_sum_rejected():
    a = _alloc()
    upd = a.reallocate({"day": 0.30, "swing": 0.40, "long_term": 0.50}, reason="typo")
    assert upd.rejected_reason is not None
    assert "sum" in upd.rejected_reason


def test_reallocate_unknown_strategy_rejected():
    a = _alloc()
    upd = a.reallocate(
        {"day": 0.30, "swing": 0.40, "long_term": 0.30, "ghost": 0.0},
        reason="unknown strat",
    )
    assert upd.rejected_reason is not None
    assert "ghost" in upd.rejected_reason


def test_halt_redistributes_weight_in_effective():
    class _FakeRE:
        def __init__(self, halted: Dict[str, str]):
            self._h = halted
        def halted_agents(self):
            return dict(self._h)

    re = _FakeRE({"day": "drawdown"})
    a = _alloc(risk_engine=re)
    w = a.effective_weights()
    assert w["day"] == 0.0
    # Swing + long_term should renormalize
    assert pytest.approx(sum(w.values()), abs=1e-6) == 1.0
    # 0.40 / 0.70 ≈ 0.571
    assert pytest.approx(w["swing"], abs=1e-3) == 0.40 / 0.70


def test_effective_budgets_halted_strategy_gets_zero():
    class _FakeRE:
        def halted_agents(self):
            return {"long_term": "mandate"}
    a = _alloc(risk_engine=_FakeRE())
    bud = a.effective_budgets(portfolio_value=100_000.0, buying_power=100_000.0)
    assert bud["long_term"].portfolio_value == 0.0
    assert bud["day"].portfolio_value > 0
    assert bud["swing"].portfolio_value > 0


def test_reallocate_history_logged_even_on_reject():
    a = _alloc()
    a.reallocate({"day": 0.25, "swing": 0.45, "long_term": 0.30}, reason="ok")
    a.reallocate({"day": 0.05, "swing": 0.65, "long_term": 0.30}, reason="too big")
    assert len(a.history) == 2
    assert a.history[0].rejected_reason is None
    assert a.history[1].rejected_reason is not None


def test_missing_strategy_in_proposal_retains_prior():
    a = _alloc()
    # Only two of three specified — third should get its prior (base) weight.
    upd = a.reallocate({"day": 0.25, "swing": 0.45}, reason="partial")
    # Accept — effective totals to ~1 and no bound breached.
    assert upd.rejected_reason is None
    w = a.effective_weights()
    assert pytest.approx(w["long_term"], abs=1e-6) == 0.30
