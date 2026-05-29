"""Tests for StrategyAllocator — weight splitting + global cap gates."""
from __future__ import annotations

import pytest

from zeus.config.strategies import GlobalLimits, StrategyConfig
from zeus.live.strategy import StrategyAllocator


def _cfg(sid: str, weight: float, enabled: bool = True) -> StrategyConfig:
    return StrategyConfig(
        id=sid,
        model_name=f"model_{sid}",
        horizon_days=3,
        max_hold_days=5,
        weight=weight,
        max_positions=5,
        max_position_pct=0.08,
        max_exposure_pct=0.60,
        enabled=enabled,
    )


def _globals(
    max_total_exposure_pct: float = 0.95,
    global_max_position_pct: float = 0.18,
) -> GlobalLimits:
    return GlobalLimits(
        max_total_exposure_pct=max_total_exposure_pct,
        global_max_position_pct=global_max_position_pct,
    )


def test_budgets_sum_to_portfolio_value() -> None:
    alloc = StrategyAllocator(
        [_cfg("day", 0.30), _cfg("swing", 0.40), _cfg("long_term", 0.30)],
        _globals(),
    )
    budgets = alloc.budgets(portfolio_value=100_000.0, buying_power=200_000.0)

    total_pv = sum(b.portfolio_value for b in budgets.values())
    total_bp = sum(b.buying_power for b in budgets.values())
    assert total_pv == pytest.approx(100_000.0, rel=1e-9)
    assert total_bp == pytest.approx(200_000.0, rel=1e-9)


def test_budgets_proportional_to_weights() -> None:
    alloc = StrategyAllocator(
        [_cfg("day", 0.30), _cfg("swing", 0.40), _cfg("long_term", 0.30)],
        _globals(),
    )
    b = alloc.budgets(portfolio_value=100_000.0, buying_power=100_000.0)
    assert b["day"].portfolio_value == pytest.approx(30_000.0)
    assert b["swing"].portfolio_value == pytest.approx(40_000.0)
    assert b["long_term"].portfolio_value == pytest.approx(30_000.0)


def test_budgets_normalize_when_weights_drift() -> None:
    # Weights intentionally summed to 0.9 (not 1.0) — allocator should normalize.
    alloc = StrategyAllocator(
        [_cfg("day", 0.27), _cfg("swing", 0.36), _cfg("long_term", 0.27)],
        _globals(),
    )
    b = alloc.budgets(portfolio_value=100_000.0, buying_power=100_000.0)
    total = sum(v.portfolio_value for v in b.values())
    assert total == pytest.approx(100_000.0)


def test_disabled_strategy_excluded_from_budgets() -> None:
    alloc = StrategyAllocator(
        [
            _cfg("day", 0.30),
            _cfg("swing", 0.40),
            _cfg("long_term", 0.30, enabled=False),
        ],
        _globals(),
    )
    b = alloc.budgets(portfolio_value=100_000.0, buying_power=100_000.0)
    assert "long_term" not in b
    assert set(b.keys()) == {"day", "swing"}
    # Only enabled strategies share the full portfolio.
    assert sum(v.portfolio_value for v in b.values()) == pytest.approx(100_000.0)


def test_no_enabled_strategies_raises() -> None:
    with pytest.raises(ValueError, match="no enabled strategies"):
        StrategyAllocator(
            [_cfg("day", 0.30, enabled=False)],
            _globals(),
        )


def test_weights_are_normalized_copy() -> None:
    alloc = StrategyAllocator(
        [_cfg("day", 0.30), _cfg("swing", 0.70)],
        _globals(),
    )
    weights = alloc.weights()
    weights["day"] = 99.0
    # mutating returned dict must not affect allocator state
    assert alloc.weights()["day"] == pytest.approx(0.30)


def test_admits_global_overlap_accepts_below_cap() -> None:
    alloc = StrategyAllocator([_cfg("day", 1.0)], _globals(global_max_position_pct=0.18))
    # PV=100k, cap=18k. Existing=5k across 1 strategy, proposed=10k → aggregate 15k < 18k.
    ok = alloc.admits_global_overlap(
        symbol="AAPL",
        proposed_notional=10_000.0,
        existing_by_strategy={"swing": 5_000.0},
        portfolio_value=100_000.0,
    )
    assert ok is True


def test_admits_global_overlap_rejects_over_cap() -> None:
    alloc = StrategyAllocator([_cfg("day", 1.0)], _globals(global_max_position_pct=0.18))
    # PV=100k, cap=18k. Existing=10k, proposed=10k → aggregate 20k > 18k.
    ok = alloc.admits_global_overlap(
        symbol="AAPL",
        proposed_notional=10_000.0,
        existing_by_strategy={"swing": 10_000.0},
        portfolio_value=100_000.0,
    )
    assert ok is False


def test_admits_global_overlap_empty_existing() -> None:
    alloc = StrategyAllocator([_cfg("day", 1.0)], _globals(global_max_position_pct=0.18))
    # First entry, no overlap — cap applies to single-strategy notional.
    ok = alloc.admits_global_overlap(
        symbol="AAPL",
        proposed_notional=15_000.0,
        existing_by_strategy={},
        portfolio_value=100_000.0,
    )
    assert ok is True
    # And rejects when the first entry already exceeds the cap
    ok = alloc.admits_global_overlap(
        symbol="AAPL",
        proposed_notional=20_000.0,
        existing_by_strategy={},
        portfolio_value=100_000.0,
    )
    assert ok is False


def test_admits_total_exposure_enforces_cap() -> None:
    alloc = StrategyAllocator(
        [_cfg("day", 1.0)], _globals(max_total_exposure_pct=0.95)
    )
    # 95k projected at PV=100k → exactly at cap, admitted
    assert alloc.admits_total_exposure(95_000.0, 100_000.0) is True
    # 95_001 → over the cap, rejected
    assert alloc.admits_total_exposure(95_001.0, 100_000.0) is False
