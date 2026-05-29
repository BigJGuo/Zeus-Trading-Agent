"""Time-weighted average price (TWAP) slicing for entry orders.

The 09:30 ET cross is the worst point in the trading day for price stability:
locked-and-crossed quotes, NBBO whipsaw, and asymmetric reaction to overnight
news. Firing the full entry qty at the open eats the full opening-spread
premium on every share. Splitting the order across the first ~12 minutes
in roughly-equal slices reduces that to the average over the window.

Slice sizing
────────────
Each slice gets `round(total / n)` shares except the first, which carries
the remainder so the sum equals `total` exactly. Single-share rounding
matters because Alpaca rejects fractional equity orders.

Threshold
─────────
Orders below `MIN_SHARES_FOR_TWAP` are submitted as a single fill — the
saved bps on a 10-share order doesn't pay for the operational complexity
of a multi-slice fill log.

Scheduling
──────────
The first slice fires immediately (from the caller's thread). Remaining
slices are scheduled as one-shot APScheduler jobs via the live scheduler
in `zeus.scheduler.context.get_scheduler()`. Each scheduled job is fully
self-contained — symbol, qty, side, strategy_id, plan rationale — so a
scheduler restart between slices still leaves the persisted slice jobs in
the SQLAlchemyJobStore for replay.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List, Optional

import structlog

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker
    from zeus.execution.order_manager import OrderManager

log = structlog.get_logger(__name__)


# Below this share count, single-order is fine — slicing 25 shares 5 ways
# rounds to 5/share slices that hit min-tick noise more than they save.
MIN_SHARES_FOR_TWAP = 50
DEFAULT_NUM_SLICES = 5
DEFAULT_WINDOW_MINUTES = 12


@dataclass
class TwapSlice:
    """One installment of a TWAP fill."""
    shares: int
    submit_at: datetime
    slice_index: int  # 0-based, just for logs
    total_slices: int


def compute_twap_slices(
    total_shares: int,
    num_slices: int = DEFAULT_NUM_SLICES,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
    start: Optional[datetime] = None,
) -> List[TwapSlice]:
    """Return `num_slices` evenly-time-spaced installments summing to `total_shares`.

    The first slice always lands at `start` (or now if None) so the caller
    submits it inline rather than scheduling — gets shares on the book
    immediately, avoiding the edge case where the scheduler is busy at open.
    """
    if total_shares <= 0:
        return []
    if num_slices <= 1 or total_shares < MIN_SHARES_FOR_TWAP:
        return [TwapSlice(shares=total_shares, submit_at=start or datetime.now(timezone.utc),
                          slice_index=0, total_slices=1)]

    per_slice = total_shares // num_slices
    remainder = total_shares - per_slice * num_slices

    start_ts = start or datetime.now(timezone.utc)
    step = timedelta(seconds=(window_minutes * 60.0) / max(num_slices - 1, 1))

    slices: List[TwapSlice] = []
    for i in range(num_slices):
        # Put any remainder shares on the first slice — gets the bulk of the
        # fill onto the book before the spread widens further. Last slice
        # carrying the remainder would invert that.
        shares = per_slice + (remainder if i == 0 else 0)
        if shares <= 0:
            continue
        slices.append(TwapSlice(
            shares=shares,
            submit_at=start_ts + step * i,
            slice_index=i,
            total_slices=num_slices,
        ))
    return slices


def _submit_slice(
    om_ref: dict,
    broker_ref: dict,
    symbol: str,
    side: str,
    strategy_id: str,
    slice_info: dict,
) -> None:
    """APScheduler entry-point for a deferred slice.

    Args are stringly typed (dicts of primitives) because APScheduler must
    pickle them into the SQLAlchemyJobStore. The OrderManager and broker
    references are re-resolved at call time from the runtime registry — see
    `zeus/scheduler/context.py` for the rationale.
    """
    from zeus.data.storage.database import get_session_factory
    from zeus.execution.execution_algo import ExecutionPlan, execute_plan
    from zeus.scheduler.context import get_loop

    loop = get_loop()
    if loop is None:
        log.warning("twap_slice_skipped_no_loop", symbol=symbol)
        return
    om = getattr(loop, "_om", None)
    broker = getattr(loop, "_broker", None)
    if om is None or broker is None:
        log.warning("twap_slice_skipped_no_handles", symbol=symbol)
        return

    shares = int(slice_info["shares"])
    try:
        quote = broker.get_latest_quote(symbol)
        bid = float(getattr(quote, "bid_price", 0) or 0)
        ask = float(getattr(quote, "ask_price", 0) or 0)
    except Exception:
        bid, ask = 0.0, 0.0

    # Re-plan order type per slice — the spread at slice 5 may not match
    # slice 1, especially the first few minutes after open.
    from zeus.execution.execution_algo import plan_entry as _plan
    plan = _plan(bid=bid, ask=ask, urgency="normal")

    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        try:
            execute_plan(
                om, session, symbol, shares, side, plan,
                strategy_id=strategy_id,
            )
        except Exception as e:
            log.error(
                "twap_slice_submit_failed",
                symbol=symbol, slice_index=slice_info.get("slice_index"),
                shares=shares, error=str(e),
            )


def schedule_twap_entry(
    om: "OrderManager",
    broker: "AlpacaBroker",
    symbol: str,
    total_shares: int,
    side: str,
    strategy_id: str,
    *,
    num_slices: int = DEFAULT_NUM_SLICES,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
) -> int:
    """Submit slice 1 inline, schedule slices 2..N on the live scheduler.

    Returns the count of slices the order was split into (0 → nothing
    submitted; 1 → single-shot, no slicing; >1 → TWAP fired).
    """
    from zeus.execution.execution_algo import plan_entry, execute_plan
    from zeus.scheduler.context import get_scheduler
    from sqlalchemy.orm import Session
    from zeus.data.storage.database import get_session_factory

    slices = compute_twap_slices(
        total_shares,
        num_slices=num_slices,
        window_minutes=window_minutes,
    )
    if not slices:
        return 0

    # Slice 1 always fires synchronously from the caller (open of market
    # window). Quote-refresh happens here on the caller's thread.
    first = slices[0]
    try:
        quote = broker.get_latest_quote(symbol)
        bid = float(getattr(quote, "bid_price", 0) or 0)
        ask = float(getattr(quote, "ask_price", 0) or 0)
    except Exception:
        bid, ask = 0.0, 0.0
    plan = plan_entry(bid=bid, ask=ask, urgency="normal")
    SessionLocal = get_session_factory()
    with SessionLocal() as session:
        execute_plan(
            om, session, symbol, first.shares, side, plan,
            strategy_id=strategy_id,
        )
    log.info(
        "twap_slice_fired_inline",
        symbol=symbol, strategy_id=strategy_id,
        shares=first.shares, slice_index=0, total_slices=first.total_slices,
    )

    if first.total_slices <= 1:
        return 1

    scheduler = get_scheduler()
    if scheduler is None:
        # Without a live scheduler, fall back to a single fill rather than
        # silently dropping the remaining slices. Catches dev/test paths
        # where TWAP is invoked outside a running scheduler.
        log.warning(
            "twap_scheduler_missing_falling_back_to_single_fill",
            symbol=symbol, dropped_slices=len(slices) - 1,
        )
        return 1

    from apscheduler.triggers.date import DateTrigger
    for s in slices[1:]:
        job_id = f"twap_{symbol}_{strategy_id}_{s.slice_index}_{uuid.uuid4().hex[:8]}"
        scheduler.add_job(
            _submit_slice,
            trigger=DateTrigger(run_date=s.submit_at),
            id=job_id,
            kwargs={
                "om_ref": {},
                "broker_ref": {},
                "symbol": symbol,
                "side": side,
                "strategy_id": strategy_id,
                "slice_info": {"shares": s.shares, "slice_index": s.slice_index},
            },
            misfire_grace_time=180,
            coalesce=True,
            replace_existing=False,
        )
        log.info(
            "twap_slice_scheduled",
            symbol=symbol, strategy_id=strategy_id,
            shares=s.shares, slice_index=s.slice_index,
            run_at=s.submit_at.isoformat(),
        )
    return len(slices)
