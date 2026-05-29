"""Centralized Trade-row emission for every position-close path.

Why this exists
───────────────
Before this module, the reconciler ([zeus/execution/reconciler.py]) was
the only code path emitting Trade rows — and only when it detected a
broker-side position drop. That captured realized P&L but always
stamped `exit_reason='broker_closed'`, even when the close was
actually triggered by a stop-loss, a time-exit, or a signal exit
initiated by the trader. For attribution analytics (segmenting trade
outcomes by *why* we exited), the wrong-exit-reason bug erased the
signal we cared most about.

The fix: every code path that submits a closing sell calls
`emit_close_trade_at_submit()` *before* the sell goes to the broker.
The Trade row is stamped with the correct `exit_reason` and the
current quote as a placeholder `exit_price`. When the actual fill
lands a few seconds later, the reconciler's existing path either
updates the placeholder via `alpaca_order_id` linkage or skips
re-emission (idempotency via `_trade_already_open_for_close`).

This keeps a single source of truth (`emit_close_trade_at_submit`) for
the close → Trade-row mapping, regardless of whether the trigger was
stop_loss, time_exit, signal_exit, risk_force, or kill_switch.

Idempotency
───────────
The reconciler still runs and still emits Trade rows for genuine
broker-initiated closes (liquidations, manual broker UI actions,
corporate actions). Before doing so it checks via
`recent_close_trade_exists()` whether a row already exists for this
(symbol, strategy_id, entry_ts) — if so, the reconciler updates the
existing row with the actual fill price rather than inserting a
duplicate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

import structlog
from sqlalchemy import select

from zeus.data.storage.database import Position, Trade

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)


# Window for the idempotency check. A Trade row written by the trader-
# close path will have exit_ts within seconds of the reconciler picking
# up the actual fill. 1 hour is generous — much shorter than the
# minimum hold period of any strategy.
_RECENT_CLOSE_LOOKBACK = timedelta(hours=1)


def _as_utc(dt: datetime) -> datetime:
    """Normalize tz so postgres ↔ SQLite round-trips don't trip."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def emit_close_trade_at_submit(
    session: "Session",
    position: Position,
    *,
    shares: int,
    exit_price: float,
    exit_reason: str,
    environment: str,
    now: Optional[datetime] = None,
) -> Optional[Trade]:
    """Write a Trade row at the moment a closing sell is submitted.

    Returns the new Trade or None if the input data is incomplete
    (zero shares, missing entry data). Caller is responsible for the
    `session.commit()`.

    `exit_price` is the current market mark — a placeholder that the
    reconciler can refine to the actual fill price once available.
    `exit_reason` is the true trigger — `stop_loss`, `time_exit`,
    `signal_exit`, `risk_force`, or `kill_switch`.
    """
    if shares <= 0:
        return None
    entry_price = float(position.avg_entry_price) if position.avg_entry_price else None
    if entry_price is None or entry_price <= 0:
        return None

    now = _as_utc(now or datetime.now(timezone.utc))
    entry_ts = _as_utc(position.entry_ts or position.updated_at or now)
    exit_ts = now if now >= entry_ts else entry_ts

    hold_days = (exit_ts - entry_ts).total_seconds() / 86400.0
    gross_pnl = (exit_price - entry_price) * shares
    actual_return = (exit_price / entry_price) - 1.0 if entry_price else None

    trade = Trade(
        symbol=position.symbol,
        strategy_id=position.strategy_id,
        strategy_model_version=position.strategy_model_version,
        direction="LONG",
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_price=entry_price,
        exit_price=exit_price,
        shares=shares,
        gross_pnl=gross_pnl,
        commission=0.0,
        net_pnl=gross_pnl,
        hold_days=hold_days,
        exit_reason=exit_reason,
        peak_price=position.peak_price,
        entry_stop=position.hard_stop,
        actual_return=actual_return,
        environment=environment,
        alpaca_order_id=None,  # filled in later when reconciler matches the fill
    )
    session.add(trade)
    log.info(
        "trade_emitted_at_close_submit",
        symbol=position.symbol,
        strategy_id=position.strategy_id,
        shares=shares,
        exit_reason=exit_reason,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl=gross_pnl,
    )
    return trade


def recent_close_trade_exists(
    session: "Session",
    *,
    symbol: str,
    strategy_id: str,
    entry_ts: Optional[datetime],
    environment: str,
    within: timedelta = _RECENT_CLOSE_LOOKBACK,
) -> bool:
    """Idempotency check for the reconciler's catch-net path.

    True when there's already a Trade row for this (symbol, strategy_id)
    whose entry_ts matches the position-row's entry_ts and whose
    exit_ts is within `within` of now. Used by the reconciler so that
    a fill caught by *both* the trader-submit path AND the
    broker-close path doesn't produce two Trade rows for the same lot.
    """
    if entry_ts is None:
        return False
    entry_ts = _as_utc(entry_ts)
    cutoff = datetime.now(timezone.utc) - within
    q = select(Trade.id).where(
        Trade.symbol == symbol,
        Trade.strategy_id == strategy_id,
        Trade.environment == environment,
        Trade.entry_ts == entry_ts,
        Trade.exit_ts >= cutoff,
    )
    return session.execute(q).first() is not None
