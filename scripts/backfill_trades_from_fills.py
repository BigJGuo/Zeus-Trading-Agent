"""Backfill the `trades` table from Alpaca's filled-order history.

Closes the feedback loop retroactively — before the reconciler started
emitting Trade rows on broker-initiated closes, exits that bypassed the
trading-loop's close path (liquidations, manual broker UI actions, the
Position-deletion path in the reconciler itself) left realized P&L
nowhere in the local DB. This script reconstructs those Trade rows from
Alpaca's authoritative fill history.

How pairing works
─────────────────
For each symbol, pull every filled buy/sell since `--since` (default 90d
ago), FIFO-match sells against buys, and emit one Trade row per closed
lot. The strategy_id is recovered from the local Order table when the
order_id is present there; otherwise defaults to 'legacy' so the
overseer can investigate the orphaned exposure.

Idempotency
───────────
Existing Trade rows are skipped by alpaca_order_id (where present) and
by (symbol, entry_ts, exit_ts, shares) fingerprint otherwise. Safe to
re-run.

Run:
    python -m scripts.backfill_trades_from_fills --since 90
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from zeus.config.settings import get_settings
from zeus.data.storage.database import Order, Trade, get_session_factory
from zeus.execution.alpaca_broker import AlpacaBroker

log = structlog.get_logger(__name__)


# Most recent first when paging Alpaca; we re-sort within each symbol below.
_PAGE_SIZE = 500


@dataclass
class _OpenLot:
    """Unconsumed buy fill awaiting a matching sell."""
    qty_remaining: int
    entry_price: float
    entry_ts: datetime
    strategy_id: str
    strategy_model_version: Optional[str]
    alpaca_order_id: Optional[str]


def _coerce_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _fetch_all_filled(
    broker: AlpacaBroker, since: datetime, until: datetime
) -> List[object]:
    """Page through Alpaca closed orders since `since`, oldest first.

    Alpaca returns at most 500 per call and only supports `after` (lower
    bound) as a filter. We page by walking `after` forward whenever a full
    page comes back.
    """
    cursor = since
    seen_ids: set = set()
    out: List[object] = []
    while True:
        batch = broker.get_filled_orders_since(since=cursor, limit=_PAGE_SIZE)
        if not batch:
            break
        new_in_batch = 0
        for o in batch:
            oid = str(getattr(o, "id", "") or "")
            if oid and oid in seen_ids:
                continue
            filled_at = _coerce_dt(getattr(o, "filled_at", None))
            if filled_at is None or filled_at > until:
                continue
            seen_ids.add(oid)
            out.append(o)
            new_in_batch += 1
        # Move the cursor forward to the most-recent fill_at in the batch
        # so the next page advances; break when no new fills land.
        if new_in_batch == 0:
            break
        latest_in_batch = max(
            (_coerce_dt(getattr(o, "filled_at", None)) for o in batch),
            default=None,
        )
        if latest_in_batch is None or latest_in_batch <= cursor:
            break
        cursor = latest_in_batch
        if new_in_batch < _PAGE_SIZE:
            break
    out.sort(key=lambda o: _coerce_dt(getattr(o, "filled_at", datetime.min.replace(tzinfo=timezone.utc))) or datetime.min.replace(tzinfo=timezone.utc))
    return out


def _strategy_for_order(session: Session, order_id: str) -> Tuple[str, Optional[str]]:
    """Recover (strategy_id, strategy_model_version) for an Alpaca order_id
    by looking it up in the local Order table. Falls back to 'legacy' when
    the order isn't tracked locally (e.g. manual broker UI fills)."""
    if not order_id:
        return ("legacy", None)
    row = session.execute(
        select(Order).where(Order.id == order_id)
    ).scalar_one_or_none()
    if row is None:
        return ("legacy", None)
    # Order doesn't carry model_version directly; the trade does. Just
    # return strategy_id, leave model_version unrecoverable from order
    # history alone.
    return (row.strategy_id or "legacy", None)


def _trade_already_exists(
    session: Session,
    symbol: str,
    entry_ts: datetime,
    exit_ts: datetime,
    shares: int,
    alpaca_order_id: Optional[str],
) -> bool:
    """Idempotency check: skip if this lot is already in trades.

    Prefer matching by alpaca_order_id (the sell side) when available;
    otherwise fingerprint on (symbol, entry_ts, exit_ts, shares).
    """
    if alpaca_order_id:
        existing = session.execute(
            select(Trade.id).where(Trade.alpaca_order_id == alpaca_order_id)
        ).first()
        if existing is not None:
            return True
    fingerprint = session.execute(
        select(Trade.id).where(
            Trade.symbol == symbol,
            Trade.entry_ts == entry_ts,
            Trade.exit_ts == exit_ts,
            Trade.shares == shares,
        )
    ).first()
    return fingerprint is not None


def backfill(
    session: Session,
    broker: AlpacaBroker,
    since: datetime,
    until: datetime,
    environment: str,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Reconstruct Trade rows from Alpaca fills in [since, until]."""
    log.info("backfill_start", since=since.isoformat(), until=until.isoformat(), environment=environment)

    fills = _fetch_all_filled(broker, since, until)
    log.info("backfill_fills_fetched", count=len(fills))

    # Group by symbol, then FIFO-match.
    by_symbol: Dict[str, List[object]] = defaultdict(list)
    for o in fills:
        sym = getattr(o, "symbol", None)
        if sym is None:
            continue
        by_symbol[sym].append(o)

    inserted = 0
    skipped_existing = 0
    skipped_unmatched_sell = 0

    for symbol, sym_fills in by_symbol.items():
        open_lots: Deque[_OpenLot] = deque()
        for o in sym_fills:
            side = str(getattr(o, "side", "")).lower()
            try:
                qty = int(float(getattr(o, "filled_qty", 0) or 0))
            except (TypeError, ValueError):
                qty = 0
            if qty <= 0:
                continue
            price = getattr(o, "filled_avg_price", None)
            if price is None:
                continue
            price = float(price)
            ts = _coerce_dt(getattr(o, "filled_at", None))
            if ts is None:
                continue
            oid = str(getattr(o, "id", "") or "") or None

            if "buy" in side:
                strategy_id, model_version = _strategy_for_order(session, oid or "")
                open_lots.append(
                    _OpenLot(
                        qty_remaining=qty,
                        entry_price=price,
                        entry_ts=ts,
                        strategy_id=strategy_id,
                        strategy_model_version=model_version,
                        alpaca_order_id=oid,
                    )
                )
                continue

            # Sell side: FIFO-match against open lots.
            remaining = qty
            while remaining > 0 and open_lots:
                lot = open_lots[0]
                take = min(lot.qty_remaining, remaining)
                if _trade_already_exists(
                    session, symbol, lot.entry_ts, ts, take, oid
                ):
                    skipped_existing += 1
                else:
                    gross = (price - lot.entry_price) * take
                    hold_days = max(0.0, (ts - lot.entry_ts).total_seconds() / 86400.0)
                    trade = Trade(
                        symbol=symbol,
                        strategy_id=lot.strategy_id,
                        strategy_model_version=lot.strategy_model_version,
                        direction="LONG",
                        entry_ts=lot.entry_ts,
                        exit_ts=ts,
                        entry_price=lot.entry_price,
                        exit_price=price,
                        shares=take,
                        gross_pnl=gross,
                        commission=0.0,
                        net_pnl=gross,
                        hold_days=hold_days,
                        exit_reason="backfilled_from_fills",
                        actual_return=(price / lot.entry_price) - 1.0 if lot.entry_price else None,
                        environment=environment,
                        alpaca_order_id=oid,
                    )
                    if not dry_run:
                        session.add(trade)
                    inserted += 1
                lot.qty_remaining -= take
                remaining -= take
                if lot.qty_remaining <= 0:
                    open_lots.popleft()
            if remaining > 0:
                # Sell with no matching buy in the window — either the buy
                # predates `since` or it's a short open (not currently
                # supported). Log and move on.
                skipped_unmatched_sell += 1
                log.warning(
                    "backfill_unmatched_sell",
                    symbol=symbol,
                    sell_qty=qty,
                    unmatched_qty=remaining,
                    sell_ts=ts.isoformat(),
                )

    if not dry_run:
        session.commit()

    result = {
        "inserted": inserted,
        "skipped_existing": skipped_existing,
        "skipped_unmatched_sell": skipped_unmatched_sell,
        "symbols_processed": len(by_symbol),
    }
    log.info("backfill_complete", **result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        type=int,
        default=90,
        help="Days back from now to start the backfill window (default: 90).",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Override target environment ('paper'|'live'). Default: settings.environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be inserted without writing.",
    )
    args = parser.parse_args()

    settings = get_settings()
    environment = args.environment or settings.environment

    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=settings.is_paper(),
    )

    until = datetime.now(timezone.utc)
    since = until - timedelta(days=args.since)

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        result = backfill(
            session=session,
            broker=broker,
            since=since,
            until=until,
            environment=environment,
            dry_run=args.dry_run,
        )
    finally:
        session.close()

    print(
        f"inserted={result['inserted']} "
        f"skipped_existing={result['skipped_existing']} "
        f"skipped_unmatched_sell={result['skipped_unmatched_sell']} "
        f"symbols={result['symbols_processed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
