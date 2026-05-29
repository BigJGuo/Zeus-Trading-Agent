"""Tests for `zeus.execution.trade_emitter` and the reconciler's
idempotency-via-refinement path.

The contract being tested:
  - trader-initiated close paths (stop, time exit, signal exit) write
    a Trade row with the *correct* exit_reason at submit time, with
    placeholder exit_price = current mark
  - reconciler's catch-net path runs later, sees the existing row, and
    *refines* exit_price + alpaca_order_id from the actual fill —
    never inserts a duplicate
  - reconciler still emits Trade rows for genuinely orphaned closes
    (broker-side liquidations with no prior trader-initiated emit)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from zeus.data.storage.database import Position, Trade
from zeus.execution.reconciler import PositionReconciler
from zeus.execution.trade_emitter import (
    emit_close_trade_at_submit,
    recent_close_trade_exists,
)


def _seed_position(
    session,
    *,
    symbol: str = "AAPL",
    strategy_id: str = "day",
    shares: int = 100,
    avg_entry_price: float = 150.0,
    current_price: float = 155.0,
    entry_ts: Optional[datetime] = None,
    peak_price: float = 158.0,
    hard_stop: float = 145.0,
) -> Position:
    if entry_ts is None:
        entry_ts = datetime.now(timezone.utc) - timedelta(days=3)
    pos = Position(
        symbol=symbol,
        strategy_id=strategy_id,
        qty=shares,
        strategy_shares=shares,
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        market_value=shares * current_price,
        unrealized_pnl=(current_price - avg_entry_price) * shares,
        unrealized_pnl_pct=0.0,
        hard_stop=hard_stop,
        peak_price=peak_price,
        entry_ts=entry_ts,
        environment="paper",
    )
    session.add(pos)
    session.commit()
    session.refresh(pos)
    return pos


# ─── emit_close_trade_at_submit ─────────────────────────────────────────────


def test_emit_writes_trade_with_correct_exit_reason(session):
    pos = _seed_position(session)
    trade = emit_close_trade_at_submit(
        session, pos,
        shares=100, exit_price=153.0,
        exit_reason="stop_loss", environment="paper",
    )
    session.commit()
    assert trade is not None
    persisted = session.query(Trade).one()
    assert persisted.exit_reason == "stop_loss"
    assert persisted.shares == 100
    assert persisted.entry_price == pytest.approx(150.0)
    assert persisted.exit_price == pytest.approx(153.0)
    assert persisted.net_pnl == pytest.approx(300.0)  # (153-150)*100
    assert persisted.strategy_id == "day"
    assert persisted.alpaca_order_id is None  # filled later by reconciler


def test_emit_distinct_reasons_round_trip(session):
    """Every exit_reason value the system uses must persist correctly —
    `stop_loss`, `time_exit`, `signal_exit`, `risk_force`, `kill_switch`."""
    reasons = ["stop_loss", "time_exit", "signal_exit", "risk_force", "kill_switch"]
    for i, reason in enumerate(reasons):
        pos = _seed_position(session, symbol=f"SYM{i}", strategy_id="day")
        emit_close_trade_at_submit(
            session, pos,
            shares=10, exit_price=200.0,
            exit_reason=reason, environment="paper",
        )
    session.commit()
    rows = session.query(Trade).order_by(Trade.symbol).all()
    assert sorted(t.exit_reason for t in rows) == sorted(reasons)


def test_emit_returns_none_on_zero_shares(session):
    pos = _seed_position(session)
    trade = emit_close_trade_at_submit(
        session, pos,
        shares=0, exit_price=153.0,
        exit_reason="stop_loss", environment="paper",
    )
    assert trade is None
    assert session.query(Trade).count() == 0


def test_emit_returns_none_when_entry_price_is_zero(session):
    """The Position schema's `avg_entry_price` is non-nullable, so the
    only invalid value we have to defend against is zero (a bookkeeping
    ghost). The emitter must refuse to write a Trade row in that case
    — emitting `(exit_price - 0) * shares` would inject a noise spike
    into realized-P&L stats."""
    pos = _seed_position(session, avg_entry_price=0.0)
    trade = emit_close_trade_at_submit(
        session, pos,
        shares=100, exit_price=153.0,
        exit_reason="stop_loss", environment="paper",
    )
    assert trade is None


def test_emit_handles_naive_entry_ts(session):
    """SQLite drops tz on `DateTime(timezone=True)` round-trips. The
    emitter must normalize to UTC so the entry_ts/exit_ts subtraction
    works in both SQLite (tests) and Postgres (prod)."""
    pos = _seed_position(session)
    pos.entry_ts = datetime.now() - timedelta(days=2)  # naive
    session.commit()
    trade = emit_close_trade_at_submit(
        session, pos,
        shares=100, exit_price=160.0,
        exit_reason="time_exit", environment="paper",
    )
    assert trade is not None
    assert trade.hold_days >= 0
    assert trade.hold_days < 3


# ─── recent_close_trade_exists (idempotency probe) ──────────────────────────


def test_recent_close_idempotency_probe(session):
    pos = _seed_position(session)
    # Before any trade exists.
    assert recent_close_trade_exists(
        session,
        symbol="AAPL", strategy_id="day",
        entry_ts=pos.entry_ts, environment="paper",
    ) is False

    emit_close_trade_at_submit(
        session, pos,
        shares=100, exit_price=160.0,
        exit_reason="time_exit", environment="paper",
    )
    session.commit()

    # After emit — same key returns True.
    assert recent_close_trade_exists(
        session,
        symbol="AAPL", strategy_id="day",
        entry_ts=pos.entry_ts, environment="paper",
    ) is True

    # Different symbol — False.
    assert recent_close_trade_exists(
        session,
        symbol="MSFT", strategy_id="day",
        entry_ts=pos.entry_ts, environment="paper",
    ) is False


# ─── Reconciler refinement path (skip-duplicate + price-update) ─────────────


@dataclass
class _BrokerPos:
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float = 100.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0


@dataclass
class _FakeFill:
    id: str
    symbol: str
    filled_qty: int
    filled_avg_price: float
    filled_at: datetime
    side: str = "sell"
    submitted_at: Optional[datetime] = None


class _FakeBroker:
    def __init__(self, positions, fills=None):
        self._positions = positions
        self._fills = fills or []

    def get_positions(self):
        return list(self._positions)

    def get_filled_orders_since(self, since, symbols=None, limit=500):
        symset = set(symbols) if symbols else None
        return [
            f for f in self._fills
            if f.filled_at >= since and (symset is None or f.symbol in symset)
        ]


def test_reconciler_refines_existing_trade_instead_of_duplicating(session):
    """The bug we're fixing: stop-loss path emits Trade with
    exit_reason='stop_loss' + placeholder exit_price; reconciler later
    sees broker closed the position and used to insert a SECOND Trade
    with exit_reason='broker_closed'. After the fix, the reconciler
    detects the existing row and *refines* it with the actual fill
    price, preserving the correct exit_reason."""
    pos = _seed_position(session, symbol="AAPL", strategy_id="day",
                         shares=100, avg_entry_price=150.0)
    emit_close_trade_at_submit(
        session, pos,
        shares=100,
        exit_price=152.50,  # placeholder mark at submit
        exit_reason="stop_loss",
        environment="paper",
    )
    session.commit()
    assert session.query(Trade).count() == 1

    # Now simulate: broker filled the sell at a slightly different price,
    # reconciler runs, broker no longer has the position.
    fill = _FakeFill(
        id="ord-stop-fill-123",
        symbol="AAPL", filled_qty=100, filled_avg_price=152.95,
        filled_at=datetime.now(timezone.utc),
    )
    broker = _FakeBroker([], fills=[fill])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    # Position deleted, but no duplicate Trade.
    assert "AAPL/day" in result["removed"]
    trades = session.query(Trade).filter_by(symbol="AAPL").all()
    assert len(trades) == 1, "reconciler must not duplicate the existing Trade row"

    refined = trades[0]
    # exit_reason preserved (the whole point of this fix).
    assert refined.exit_reason == "stop_loss"
    # exit_price refined to the actual fill.
    assert refined.exit_price == pytest.approx(152.95)
    # alpaca_order_id populated from the fill.
    assert refined.alpaca_order_id == "ord-stop-fill-123"
    # net_pnl recomputed.
    assert refined.net_pnl == pytest.approx((152.95 - 150.0) * 100)


def test_reconciler_still_emits_for_genuine_orphans(session):
    """If no trader-initiated Trade row exists for a closed position
    (true broker-initiated close — liquidation, manual broker UI),
    the reconciler must still emit a `broker_closed` row. The
    idempotency check should NOT silently drop these."""
    _seed_position(session, symbol="MSFT", strategy_id="swing",
                   shares=50, avg_entry_price=300.0)
    # No emit_close_trade_at_submit call — pure broker orphan.

    broker = _FakeBroker([])  # no positions, no fills
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert "MSFT/swing" in result["trades_emitted"]
    t = session.query(Trade).filter_by(symbol="MSFT").one()
    assert t.exit_reason == "broker_closed"


def test_reconciler_does_not_refine_old_trade_outside_window(session):
    """If a Trade exists but its exit_ts is OLDER than the idempotency
    lookback window, it represents a separate prior close, not the
    current one. The reconciler should emit a fresh row."""
    pos = _seed_position(session, symbol="GOOG", strategy_id="day",
                         shares=20, avg_entry_price=180.0)
    # Manually insert a stale Trade row from 2 days ago.
    stale_entry_ts = pos.entry_ts
    old_trade = Trade(
        symbol="GOOG", strategy_id="day", direction="LONG",
        entry_ts=stale_entry_ts,
        exit_ts=datetime.now(timezone.utc) - timedelta(days=2),
        entry_price=180.0, exit_price=185.0, shares=20,
        gross_pnl=100.0, net_pnl=100.0, hold_days=1.0,
        exit_reason="signal_exit", environment="paper",
    )
    session.add(old_trade)
    session.commit()

    broker = _FakeBroker([])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    reconciler.reconcile(session)

    # Two rows now: the old signal_exit + a fresh broker_closed.
    trades = session.query(Trade).filter_by(symbol="GOOG").all()
    assert len(trades) == 2
    reasons = sorted(t.exit_reason for t in trades)
    assert reasons == ["broker_closed", "signal_exit"]
