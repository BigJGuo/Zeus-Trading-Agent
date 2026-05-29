"""Order lifecycle management: submit, poll, persist."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional, cast

import structlog
from sqlalchemy.orm import Session

from zeus.data.storage.database import Order as DBOrder
from zeus.monitoring.trading_log import append_fill

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker

log = structlog.get_logger(__name__)


class OrderManager:
    """
    Thin wrapper around AlpacaBroker that persists every order event to the DB.
    """

    def __init__(self, broker: "AlpacaBroker", environment: str = "paper"):
        self._broker = broker
        self._env = environment

    # ─── Submit ───────────────────────────────────────────────────────────────
    def submit_market(
        self, session: Session, symbol: str, qty: int, side: str,
        *, strategy_id: str = "legacy",
    ) -> DBOrder:
        alpaca_order = self._broker.submit_market_order(symbol=symbol, qty=qty, side=side)
        return self._persist(session, alpaca_order, "market", strategy_id=strategy_id)

    def submit_limit(
        self, session: Session, symbol: str, qty: int, side: str, limit_price: float,
        *, strategy_id: str = "legacy",
    ) -> DBOrder:
        alpaca_order = self._broker.submit_limit_order(
            symbol=symbol, qty=qty, side=side, limit_price=limit_price
        )
        return self._persist(
            session, alpaca_order, "limit", limit_price=limit_price, strategy_id=strategy_id,
        )

    # ─── Cancel ───────────────────────────────────────────────────────────────
    def cancel(self, session: Session, order_id: str) -> None:
        self._broker.cancel_order(order_id)
        row = session.get(DBOrder, order_id)
        if row is not None:
            r = cast(Any, row)
            r.status = "canceled"
            r.updated_at = datetime.now(timezone.utc)
            session.commit()

    # ─── Poll / sync ──────────────────────────────────────────────────────────
    def refresh_order(self, session: Session, order_id: str) -> Optional[DBOrder]:
        order_id = str(order_id)
        try:
            alpaca_order: Any = self._broker.get_order(order_id)
        except Exception as e:
            log.warning("order_refresh_failed", order_id=order_id, error=str(e))
            return None

        row = session.get(DBOrder, order_id)
        if row is None:
            return self._persist(session, alpaca_order, "unknown")

        r = cast(Any, row)
        prior_filled = int(r.filled_qty or 0)
        r.status = str(alpaca_order.status).lower().split(".")[-1]
        new_filled = int(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0
        r.filled_qty = new_filled
        r.filled_avg_price = (
            float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None
        )
        if alpaca_order.filled_at:
            r.filled_at = alpaca_order.filled_at
        r.updated_at = datetime.now(timezone.utc)
        session.commit()

        delta = new_filled - prior_filled
        if delta > 0:
            append_fill(
                symbol=r.symbol,
                side=r.side,
                shares=delta,
                price=r.filled_avg_price,
                order_id=r.id,
                order_type=r.order_type or "unknown",
                environment=self._env,
            )
        return row

    def refresh_all_open(self, session: Session) -> List[DBOrder]:
        from zeus.data.storage.database import Order as DBOrderModel
        # Refresh every DB-tracked order that's not in a terminal state.
        # Using only broker.get_open_orders() misses orders that already
        # filled/canceled at Alpaca but still show a non-terminal status in our DB.
        terminal = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}
        open_rows = (
            session.query(DBOrderModel)
            .filter(~DBOrderModel.status.in_(terminal))
            .all()
        )
        refreshed = []
        for row in open_rows:
            res = self.refresh_order(session, cast(str, row.id))
            if res is not None:
                refreshed.append(res)
        return refreshed

    # ─── Internal ─────────────────────────────────────────────────────────────
    def _persist(
        self,
        session: Session,
        alpaca_order: Any,
        order_type: str,
        limit_price: Optional[float] = None,
        *,
        strategy_id: str = "legacy",
    ) -> DBOrder:
        now = datetime.now(timezone.utc)
        row = DBOrder(
            id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            strategy_id=strategy_id,
            side=str(alpaca_order.side).lower().split(".")[-1],
            order_type=order_type,
            qty=int(alpaca_order.qty),
            limit_price=limit_price,
            submitted_at=now,
            filled_qty=int(getattr(alpaca_order, "filled_qty", 0) or 0),
            status=str(alpaca_order.status).lower().split(".")[-1],
            environment=self._env,
            created_at=now,
            updated_at=now,
        )
        session.merge(row)
        session.commit()
        r = cast(Any, row)
        log.info(
            "order_persisted",
            order_id=r.id, symbol=r.symbol, side=r.side,
            qty=r.qty, type=order_type, status=r.status,
        )
        if r.filled_qty and r.filled_qty > 0:
            fill_price = (
                float(getattr(alpaca_order, "filled_avg_price", 0) or 0)
                or limit_price
            )
            append_fill(
                symbol=r.symbol,
                side=r.side,
                shares=int(r.filled_qty),
                price=fill_price,
                order_id=r.id,
                order_type=order_type,
                environment=self._env,
            )
        return row
