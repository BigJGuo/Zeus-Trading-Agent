"""Tests for PositionReconciler — drift rescale + unknown-symbol attribution
+ Trade-row emission on broker-initiated closes."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from zeus.data.storage.database import Position, Trade
from zeus.execution.reconciler import DEFAULT_STRATEGY, PositionReconciler


# ─── Test doubles ─────────────────────────────────────────────────────────────


@dataclass
class _BrokerPos:
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float = 100.0
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    unrealized_plpc: float = 0.0

    def __post_init__(self):
        if self.market_value == 0.0:
            self.market_value = self.qty * self.current_price


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
    def __init__(self, positions: List[_BrokerPos], fills: Optional[List[_FakeFill]] = None):
        self._positions = positions
        self._fills = fills or []

    def get_positions(self) -> List[_BrokerPos]:
        return list(self._positions)

    def get_filled_orders_since(
        self,
        since: datetime,
        symbols: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[_FakeFill]:
        symset = set(symbols) if symbols else None
        return [
            f for f in self._fills
            if f.filled_at >= since and (symset is None or f.symbol in symset)
        ]


def _seed(
    session,
    *,
    symbol: str,
    strategy_id: str,
    strategy_shares: int,
    qty: int,
    environment: str = "paper",
    avg_entry_price: float = 100.0,
    current_price: float = 100.0,
    entry_ts: Optional[datetime] = None,
):
    now = datetime.now(timezone.utc)
    pos = Position(
        symbol=symbol,
        strategy_id=strategy_id,
        qty=qty,
        strategy_shares=strategy_shares,
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        market_value=strategy_shares * current_price,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
        environment=environment,
        entry_ts=entry_ts,
        updated_at=now,
    )
    session.add(pos)
    session.commit()
    return pos


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_unknown_symbol_attributed_to_legacy(session):
    broker = _FakeBroker([_BrokerPos(symbol="NVDA", qty=25, avg_entry_price=500.0)])
    reconciler = PositionReconciler(broker=broker, environment="paper")

    result = reconciler.reconcile(session)

    assert f"NVDA/{DEFAULT_STRATEGY}" in result["added"]
    row = session.query(Position).filter_by(symbol="NVDA").one()
    assert row.strategy_id == DEFAULT_STRATEGY
    assert row.qty == 25
    assert row.strategy_shares == 25


def test_no_drift_updates_market_stats(session):
    _seed(session, symbol="AAPL", strategy_id="day", strategy_shares=60, qty=60)
    _seed(session, symbol="AAPL", strategy_id="swing", strategy_shares=40, qty=100)

    broker = _FakeBroker([
        _BrokerPos(symbol="AAPL", qty=100, avg_entry_price=150.0,
                   current_price=155.0, market_value=15_500.0,
                   unrealized_pl=500.0, unrealized_plpc=0.0333),
    ])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert result["drifted"] == []
    rows = session.query(Position).filter_by(symbol="AAPL").all()
    by_sid = {r.strategy_id: r for r in rows}
    # Shares preserved (60/40)
    assert by_sid["day"].strategy_shares == 60
    assert by_sid["swing"].strategy_shares == 40
    # Market value proportional to ownership fraction
    assert by_sid["day"].market_value == pytest.approx(15_500.0 * 0.6)
    assert by_sid["swing"].market_value == pytest.approx(15_500.0 * 0.4)
    assert by_sid["day"].unrealized_pnl == pytest.approx(500.0 * 0.6)
    assert by_sid["swing"].unrealized_pnl == pytest.approx(500.0 * 0.4)


def test_drift_rescales_proportionally(session):
    # We track 100 shares (60 day + 40 swing), broker says 80 (e.g. a manual sell of 20)
    _seed(session, symbol="AAPL", strategy_id="day", strategy_shares=60, qty=100)
    _seed(session, symbol="AAPL", strategy_id="swing", strategy_shares=40, qty=100)

    broker = _FakeBroker([
        _BrokerPos(symbol="AAPL", qty=80, avg_entry_price=150.0,
                   current_price=155.0, market_value=12_400.0),
    ])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert "AAPL" in result["drifted"]
    rows = session.query(Position).filter_by(symbol="AAPL").all()
    by_sid = {r.strategy_id: r for r in rows}
    # Proportional rescale: 60/100 → 48, 40/100 → 32
    assert by_sid["day"].strategy_shares == 48
    assert by_sid["swing"].strategy_shares == 32
    # And the sum matches broker qty exactly
    assert by_sid["day"].strategy_shares + by_sid["swing"].strategy_shares == 80


def test_drift_rescale_preserves_relative_ownership(session):
    # 70/30 split across two strategies
    _seed(session, symbol="TSLA", strategy_id="day", strategy_shares=70, qty=100)
    _seed(session, symbol="TSLA", strategy_id="long_term", strategy_shares=30, qty=100)

    broker = _FakeBroker([
        _BrokerPos(symbol="TSLA", qty=200, avg_entry_price=200.0,
                   current_price=210.0, market_value=42_000.0),
    ])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    reconciler.reconcile(session)

    rows = session.query(Position).filter_by(symbol="TSLA").all()
    by_sid = {r.strategy_id: r for r in rows}
    # 70/30 of 200 = 140 / 60
    assert by_sid["day"].strategy_shares == 140
    assert by_sid["long_term"].strategy_shares == 60


def test_symbol_removed_when_broker_no_longer_has_it(session):
    _seed(session, symbol="AAPL", strategy_id="day", strategy_shares=100, qty=100)
    _seed(session, symbol="MSFT", strategy_id="swing", strategy_shares=50, qty=50)

    # Broker only has MSFT now
    broker = _FakeBroker([
        _BrokerPos(symbol="MSFT", qty=50, avg_entry_price=300.0,
                   current_price=305.0, market_value=15_250.0),
    ])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert f"AAPL/day" in result["removed"]
    assert session.query(Position).filter_by(symbol="AAPL").count() == 0
    assert session.query(Position).filter_by(symbol="MSFT").count() == 1


def test_environment_scoping_does_not_touch_other_env(session):
    """Reconciling `paper` must not touch `live` rows.

    Positions are keyed by (symbol, strategy_id) — environment is *not*
    part of the PK, so the same (symbol, strategy_id) can't legitimately
    appear in both environments at once. To test the env filter we use
    different symbols per environment and verify cross-env isolation.
    """
    # paper-only symbol
    _seed(session, symbol="AAPL", strategy_id="day", strategy_shares=100,
          qty=100, environment="paper")
    # live-only symbol — reconciler scoped to "paper" should leave alone
    _seed(session, symbol="TSLA", strategy_id="day", strategy_shares=500,
          qty=500, environment="live")

    # Reconcile paper only, broker returns nothing → paper row should be removed
    broker = _FakeBroker([])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    reconciler.reconcile(session)

    remaining = session.query(Position).all()
    # paper row removed, live row preserved
    assert len(remaining) == 1
    assert remaining[0].environment == "live"
    assert remaining[0].symbol == "TSLA"
    assert remaining[0].strategy_shares == 500


def test_trade_emitted_when_broker_closes_position_no_fill_data(session):
    # Broker no longer has AAPL — no closing-fill history available, so the
    # reconciler should still emit a Trade using cached current_price + now.
    entry_ts = datetime.now(timezone.utc) - timedelta(days=3)
    _seed(
        session,
        symbol="AAPL",
        strategy_id="day",
        strategy_shares=100,
        qty=100,
        avg_entry_price=150.0,
        current_price=160.0,
        entry_ts=entry_ts,
    )

    broker = _FakeBroker([])  # no positions, no fills
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert "AAPL/day" in result["removed"]
    assert "AAPL/day" in result["trades_emitted"]

    trades = session.query(Trade).filter_by(symbol="AAPL").all()
    assert len(trades) == 1
    t = trades[0]
    assert t.strategy_id == "day"
    assert t.direction == "LONG"
    assert t.exit_reason == "broker_closed"
    assert t.shares == 100
    assert t.entry_price == pytest.approx(150.0)
    assert t.exit_price == pytest.approx(160.0)  # fallback: cached current_price
    assert t.net_pnl == pytest.approx(1000.0)    # (160 - 150) * 100
    assert t.commission == pytest.approx(0.0)
    assert t.environment == "paper"
    assert t.alpaca_order_id is None  # no fill matched


def test_trade_emitted_uses_alpaca_fill_when_available(session):
    entry_ts = datetime.now(timezone.utc) - timedelta(days=5)
    fill_ts = datetime.now(timezone.utc) - timedelta(hours=2)
    _seed(
        session,
        symbol="MSFT",
        strategy_id="swing",
        strategy_shares=50,
        qty=50,
        avg_entry_price=300.0,
        current_price=310.0,
        entry_ts=entry_ts,
    )

    # Broker dropped the symbol and reports a matching closing fill.
    fill = _FakeFill(
        id="ord-abc-123",
        symbol="MSFT",
        filled_qty=50,
        filled_avg_price=315.50,
        filled_at=fill_ts,
        side="sell",
    )
    broker = _FakeBroker([], fills=[fill])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert "MSFT/swing" in result["trades_emitted"]
    t = session.query(Trade).filter_by(symbol="MSFT").one()
    # Exit price + alpaca_order_id come from the fill, not the cached snapshot.
    assert t.exit_price == pytest.approx(315.50)
    assert t.alpaca_order_id == "ord-abc-123"
    assert t.net_pnl == pytest.approx((315.50 - 300.0) * 50)
    assert t.hold_days == pytest.approx((fill_ts - entry_ts).total_seconds() / 86400.0, rel=1e-3)


def test_trade_not_emitted_for_position_with_no_entry_data(session):
    # Zero-share row from a prior drift rescale — emitting a Trade for this
    # would inject a noise zero into realized-P&L stats.
    _seed(
        session,
        symbol="GHOST",
        strategy_id="day",
        strategy_shares=0,
        qty=0,
        avg_entry_price=100.0,
        current_price=100.0,
    )
    broker = _FakeBroker([])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert "GHOST/day" in result["removed"]
    assert "GHOST/day" not in result["trades_emitted"]
    assert session.query(Trade).count() == 0


def test_zero_tracked_shares_assigns_all_to_first_row(session):
    # Pathological but possible — both strategy rows at 0 shares, broker says 10.
    _seed(session, symbol="AAPL", strategy_id="day", strategy_shares=0, qty=10)
    _seed(session, symbol="AAPL", strategy_id="swing", strategy_shares=0, qty=10)

    broker = _FakeBroker([
        _BrokerPos(symbol="AAPL", qty=10, avg_entry_price=100.0,
                   current_price=100.0, market_value=1_000.0),
    ])
    reconciler = PositionReconciler(broker=broker, environment="paper")
    result = reconciler.reconcile(session)

    assert "AAPL" in result["drifted"]
    rows = session.query(Position).filter_by(symbol="AAPL").order_by(Position.strategy_id).all()
    # One of them gets all 10 shares, the other stays at 0. Total = 10.
    total = sum(r.strategy_shares for r in rows)
    assert total == 10
