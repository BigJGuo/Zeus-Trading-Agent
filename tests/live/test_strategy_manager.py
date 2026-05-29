"""Tests for StrategyManager — upsert, time-exits, kill-switch gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from zeus.config.strategies import GlobalLimits, StrategyConfig
from zeus.data.storage.database import Position
from zeus.live.strategy import StrategyAllocator, StrategyContext
from zeus.live.strategy_manager import StrategyManager


# ─── Test doubles ─────────────────────────────────────────────────────────────


@dataclass
class _FakeAccount:
    portfolio_value: float = 100_000.0
    buying_power: float = 200_000.0
    cash: float = 50_000.0


@dataclass
class _FakePosition:
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float = 100.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0


class _FakeBroker:
    def __init__(self, positions: Optional[List[_FakePosition]] = None, pv: float = 100_000.0):
        self._positions = positions or []
        self._account = _FakeAccount(portfolio_value=pv, buying_power=pv * 2)
        self.quotes: Dict[str, Any] = {}

    def get_account(self) -> _FakeAccount:
        return self._account

    def get_positions(self) -> List[_FakePosition]:
        return list(self._positions)

    def get_latest_quote(self, symbol: str):
        return self.quotes.get(symbol, type("Q", (), {"bid_price": 100.0, "ask_price": 100.1})())


class _FakeRiskEngine:
    def __init__(self, kill: bool = False, stop_entries: bool = False):
        self._kill = kill
        self._stop = stop_entries
        self.calls: list = []

    @property
    def kill_switch_active(self) -> bool:
        return self._kill

    def update_portfolio_value(self, pv: float):
        return type("DDState", (), {
            "stop_new_entries": self._stop,
            "level": type("L", (), {"value": 0})(),
        })()

    def pre_trade_check(self, trade, portfolio, dd_state):
        self.calls.append(("pre_trade_check", trade.symbol, trade.shares))
        return type("R", (), {"approved": True, "reason": "", "adjusted_shares": None})()


class _FakeOrderManager:
    def __init__(self):
        self.submitted: list = []


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _cfg(sid: str, weight: float, max_hold: int = 5) -> StrategyConfig:
    return StrategyConfig(
        id=sid,
        model_name=f"model_{sid}",
        horizon_days=3,
        max_hold_days=max_hold,
        weight=weight,
        max_positions=5,
        max_position_pct=0.08,
        max_exposure_pct=0.60,
    )


def _globals() -> GlobalLimits:
    return GlobalLimits(max_total_exposure_pct=0.95, global_max_position_pct=0.18)


def _make_manager(patched_session_factory, contexts_ids: List[str], max_hold_per: Optional[Dict[str, int]] = None):
    max_hold_per = max_hold_per or {}
    cfgs = [_cfg(sid, 1.0 / len(contexts_ids), max_hold=max_hold_per.get(sid, 5)) for sid in contexts_ids]
    alloc = StrategyAllocator(cfgs, _globals())
    contexts: Dict[str, StrategyContext] = {}
    for c in cfgs:
        # Use a thin object with a strategy_id + config — StrategyContext
        # requires real model/generator/filter/etc for plan-building, but
        # our manager tests don't invoke those code paths directly.
        ctx = StrategyContext.__new__(StrategyContext)
        ctx.config = c
        ctx.model = type("M", (), {"version_": "v-test"})()
        ctx.signal_generator = None
        ctx.signal_filter = None
        ctx.portfolio_constructor = None
        ctx.feature_pipeline = None
        ctx.current_plan = None
        contexts[c.id] = ctx
    manager = StrategyManager(
        contexts=contexts,
        allocator=alloc,
        broker=_FakeBroker(),
        order_manager=_FakeOrderManager(),
        risk_engine=_FakeRiskEngine(),
        globals_=_globals(),
        environment="paper",
    )
    return manager


# ─── _upsert_position ─────────────────────────────────────────────────────────


def test_upsert_position_inserts_new_row(patched_session_factory):
    mgr = _make_manager(patched_session_factory, ["day"])
    with patched_session_factory() as s:
        mgr._upsert_position(
            s, symbol="AAPL", strategy_id="day",
            shares=100, price=150.0, stop=145.0, model_version="v1",
        )
        s.commit()
        rows = s.query(Position).all()
        assert len(rows) == 1
        r = rows[0]
        assert r.symbol == "AAPL"
        assert r.strategy_id == "day"
        assert r.qty == 100
        assert r.strategy_shares == 100
        assert r.avg_entry_price == pytest.approx(150.0)
        assert r.hard_stop == pytest.approx(145.0)
        assert r.entry_ts is not None
        assert r.strategy_model_version == "v1"


def test_upsert_position_averages_on_repeat_entry(patched_session_factory):
    mgr = _make_manager(patched_session_factory, ["day"])
    with patched_session_factory() as s:
        mgr._upsert_position(
            s, symbol="AAPL", strategy_id="day",
            shares=100, price=150.0, stop=145.0, model_version="v1",
        )
        s.commit()
        mgr._upsert_position(
            s, symbol="AAPL", strategy_id="day",
            shares=100, price=160.0, stop=155.0, model_version="v1",
        )
        s.commit()
        row = s.query(Position).filter_by(symbol="AAPL", strategy_id="day").one()
        assert row.strategy_shares == 200
        assert row.qty == 200
        # Share-weighted average of two 100-share lots at 150 and 160
        assert row.avg_entry_price == pytest.approx(155.0)


def test_upsert_position_separate_rows_per_strategy(patched_session_factory):
    mgr = _make_manager(patched_session_factory, ["day", "swing"])
    with patched_session_factory() as s:
        mgr._upsert_position(s, symbol="AAPL", strategy_id="day",
                             shares=50, price=150.0, stop=145.0, model_version="v1")
        mgr._upsert_position(s, symbol="AAPL", strategy_id="swing",
                             shares=30, price=152.0, stop=147.0, model_version="v2")
        s.commit()
        rows = s.query(Position).filter_by(symbol="AAPL").all()
        assert {r.strategy_id for r in rows} == {"day", "swing"}
        day = next(r for r in rows if r.strategy_id == "day")
        swing = next(r for r in rows if r.strategy_id == "swing")
        assert day.strategy_shares == 50
        assert swing.strategy_shares == 30


# ─── execute_time_exits_all ───────────────────────────────────────────────────


def _seed_position(session, *, symbol: str, strategy_id: str, shares: int, age_days: float, environment: str = "paper"):
    now = datetime.now(timezone.utc)
    pos = Position(
        symbol=symbol,
        strategy_id=strategy_id,
        qty=shares,
        strategy_shares=shares,
        avg_entry_price=100.0,
        current_price=100.0,
        market_value=shares * 100.0,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
        entry_ts=now - timedelta(days=age_days),
        environment=environment,
        updated_at=now,
    )
    session.add(pos)
    session.commit()


def test_execute_time_exits_only_exits_old_positions(patched_session_factory, monkeypatch):
    mgr = _make_manager(
        patched_session_factory,
        ["day", "swing"],
        max_hold_per={"day": 5, "swing": 10},
    )
    with patched_session_factory() as s:
        # Day strategy: one 2-day position (should hold) + one 7-day position (should exit)
        _seed_position(s, symbol="AAPL", strategy_id="day", shares=100, age_days=2)
        _seed_position(s, symbol="MSFT", strategy_id="day", shares=50, age_days=7)
        # Swing strategy: one 7-day (should hold, max_hold=10), one 15-day (should exit)
        _seed_position(s, symbol="GOOG", strategy_id="swing", shares=20, age_days=7)
        _seed_position(s, symbol="AMZN", strategy_id="swing", shares=10, age_days=15)

    exited_calls: list = []

    def _fake_execute_plan(om, session, symbol, shares, side, plan, strategy_id=None):
        exited_calls.append((strategy_id, symbol, shares, side))

    def _fake_plan_exit_stop():
        return object()

    monkeypatch.setattr("zeus.execution.execution_algo.execute_plan", _fake_execute_plan)
    monkeypatch.setattr("zeus.execution.execution_algo.plan_exit_stop", _fake_plan_exit_stop)

    result = mgr.execute_time_exits_all()
    assert result == {"day": 1, "swing": 1}
    assert sorted(exited_calls) == sorted([
        ("day", "MSFT", 50, "sell"),
        ("swing", "AMZN", 10, "sell"),
    ])


def test_execute_time_exits_skips_positions_missing_entry_ts(patched_session_factory, monkeypatch):
    mgr = _make_manager(patched_session_factory, ["day"], max_hold_per={"day": 5})
    with patched_session_factory() as s:
        pos = Position(
            symbol="AAPL", strategy_id="day", qty=100, strategy_shares=100,
            avg_entry_price=100.0, current_price=100.0, market_value=10_000.0,
            unrealized_pnl=0.0, unrealized_pnl_pct=0.0,
            entry_ts=None, environment="paper",
        )
        s.add(pos)
        s.commit()

    calls: list = []
    monkeypatch.setattr(
        "zeus.execution.execution_algo.execute_plan",
        lambda *a, **k: calls.append(("exec", a, k)),
    )
    monkeypatch.setattr("zeus.execution.execution_algo.plan_exit_stop", lambda: object())

    result = mgr.execute_time_exits_all()
    assert result["day"] == 0
    assert calls == []


def test_execute_time_exits_respects_kill_switch(patched_session_factory, monkeypatch):
    mgr = _make_manager(patched_session_factory, ["day"], max_hold_per={"day": 1})
    mgr._risk = _FakeRiskEngine(kill=True)
    with patched_session_factory() as s:
        _seed_position(s, symbol="AAPL", strategy_id="day", shares=100, age_days=30)

    calls: list = []
    monkeypatch.setattr(
        "zeus.execution.execution_algo.execute_plan",
        lambda *a, **k: calls.append(("exec", a, k)),
    )
    monkeypatch.setattr("zeus.execution.execution_algo.plan_exit_stop", lambda: object())

    result = mgr.execute_time_exits_all()
    assert result == {"day": 0}
    assert calls == []


def test_summaries_counts_per_strategy_positions(patched_session_factory):
    mgr = _make_manager(patched_session_factory, ["day", "swing"])
    with patched_session_factory() as s:
        _seed_position(s, symbol="AAPL", strategy_id="day", shares=100, age_days=1)
        _seed_position(s, symbol="MSFT", strategy_id="day", shares=50, age_days=1)
        _seed_position(s, symbol="GOOG", strategy_id="swing", shares=20, age_days=1)

    summaries = mgr.summaries()
    by_id = {s.strategy_id: s for s in summaries}
    assert by_id["day"].n_open_positions == 2
    assert by_id["swing"].n_open_positions == 1
    # AAPL (100*100) + MSFT (50*100) = 15_000
    assert by_id["day"].deployed_notional == pytest.approx(15_000.0)
    assert by_id["swing"].deployed_notional == pytest.approx(2_000.0)


# ─── Manager construction ────────────────────────────────────────────────────


def test_manager_requires_at_least_one_context(patched_session_factory):
    alloc = StrategyAllocator([_cfg("day", 1.0)], _globals())
    with pytest.raises(ValueError, match="at least one"):
        StrategyManager(
            contexts={},
            allocator=alloc,
            broker=_FakeBroker(),
            order_manager=_FakeOrderManager(),
            risk_engine=_FakeRiskEngine(),
            globals_=_globals(),
        )


def test_manager_strategy_ids_matches_contexts(patched_session_factory):
    mgr = _make_manager(patched_session_factory, ["day", "swing", "long_term"])
    assert set(mgr.strategy_ids()) == {"day", "swing", "long_term"}
