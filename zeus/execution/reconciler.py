"""Position reconciliation between Alpaca (source of truth) and local DB.

Multi-strategy: the broker sees aggregate shares per symbol. Our DB keys
positions on (symbol, strategy_id) with `strategy_shares` partitioning the
aggregate between strategies. Drift — sum(strategy_shares) != broker_qty —
is expected after stops or manual broker interventions and gets rescaled
proportionally, with a `strategy_share_drift` log event so the overseer
can investigate.

Trade emission: when the broker no longer reports a symbol that we still
track, that's a close — the trading loop normally writes a Trade row when
it submits the close itself, but a broker-initiated close (liquidation,
manual broker UI action, corporate action) bypasses that path. In that
case the reconciler emits a Trade row tagged `exit_reason='broker_closed'`
with the best exit_price/exit_ts we can recover from the recent fill
history, falling back to the last cached `current_price` and `now`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from zeus.data.storage.database import Position, Trade

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker

log = structlog.get_logger(__name__)

DEFAULT_STRATEGY = "legacy"

# How far back to query Alpaca for closing fills when attributing exits.
_FILL_LOOKBACK_HOURS = 48


def _as_utc(dt: datetime) -> datetime:
    """Attach UTC if `dt` is tz-naive, otherwise return unchanged."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class PositionReconciler:
    """Sync local (symbol, strategy_id) position rows with broker aggregate qty.

    Alpaca is authoritative on aggregate shares. The splitting of those shares
    between strategies is our own bookkeeping — on drift, we rescale the
    `strategy_shares` proportionally to preserve relative ownership.
    """

    def __init__(self, broker: "AlpacaBroker", environment: str = "paper"):
        self._broker = broker
        self._env = environment

    def reconcile(self, session: Session) -> Dict[str, List[str]]:
        """Pulls positions from Alpaca, upserts by (symbol, strategy_id), rescales
        strategy_shares on drift, removes rows where broker no longer has the
        symbol at all. Emits a Trade row for each removed Position so realized
        P&L is captured even for broker-initiated closes.
        Returns {'added': [...], 'updated': [...], 'removed': [...], 'drifted': [...], 'trades_emitted': [...]}
        """
        try:
            broker_positions = self._broker.get_positions()
        except Exception as e:
            log.error("reconcile_failed_fetch", error=str(e))
            raise

        broker_symbols = {p.symbol for p in broker_positions}

        db_rows = session.execute(
            select(Position).where(Position.environment == self._env)
        ).scalars().all()

        db_by_key: Dict[tuple, Position] = {(r.symbol, r.strategy_id): r for r in db_rows}
        db_by_symbol: Dict[str, List[Position]] = {}
        for r in db_rows:
            db_by_symbol.setdefault(r.symbol, []).append(r)

        added, updated, removed, drifted, trades_emitted = [], [], [], [], []
        now = datetime.now(timezone.utc)

        for pos in broker_positions:
            qty = int(pos.qty)
            avg_entry = float(pos.avg_entry_price)
            cp = getattr(pos, "current_price", None)
            mv = getattr(pos, "market_value", None)
            current_price = float(cp) if cp else None
            market_value = float(mv) if mv else qty * avg_entry
            unrealized_pnl = float(getattr(pos, "unrealized_pl", 0.0))
            unrealized_pct = float(getattr(pos, "unrealized_plpc", 0.0))

            existing = db_by_symbol.get(pos.symbol, [])

            if not existing:
                # Symbol the broker has but we don't know about — attribute to
                # the default 'legacy' bucket so the overseer can investigate.
                row = Position(
                    symbol=pos.symbol,
                    strategy_id=DEFAULT_STRATEGY,
                    qty=qty,
                    strategy_shares=qty,
                    avg_entry_price=avg_entry,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pct,
                    environment=self._env,
                    updated_at=now,
                )
                session.add(row)
                added.append(f"{pos.symbol}/{DEFAULT_STRATEGY}")
                continue

            total_tracked = sum(int(r.strategy_shares or r.qty or 0) for r in existing)

            # Drift: aggregate qty we track disagrees with broker qty
            if total_tracked != qty:
                drifted.append(pos.symbol)
                log.warning(
                    "strategy_share_drift",
                    symbol=pos.symbol,
                    broker_qty=qty,
                    tracked_qty=total_tracked,
                    n_strategies=len(existing),
                )
                # Proportional rescale — preserve relative ownership.
                if total_tracked > 0:
                    for r in existing:
                        share = (r.strategy_shares or r.qty or 0) / total_tracked
                        r.strategy_shares = max(0, int(round(share * qty)))
                else:
                    # No prior allocation — put everything in the first (typically 'legacy').
                    existing[0].strategy_shares = qty

            # Update per-strategy rows with aggregate market stats.
            for r in existing:
                r.qty = qty
                r.avg_entry_price = avg_entry
                r.current_price = current_price
                # market_value is per-strategy: scale by this strategy's ownership fraction
                own_frac = (
                    (r.strategy_shares or 0) / qty if qty > 0 else 0.0
                )
                r.market_value = market_value * own_frac
                r.unrealized_pnl = unrealized_pnl * own_frac
                r.unrealized_pnl_pct = unrealized_pct
                r.updated_at = now
                updated.append(f"{r.symbol}/{r.strategy_id}")

        # Build a per-symbol map of recent closing fills so we can attribute
        # exit price/timestamp when emitting Trade rows for removed positions.
        closed_symbols = [s for s in db_by_symbol.keys() if s not in broker_symbols]
        fills_by_symbol = self._fetch_recent_sell_fills(closed_symbols, now)

        # Remove rows the broker no longer has, emitting a Trade row first.
        for (symbol, strategy_id), row in db_by_key.items():
            if symbol not in broker_symbols:
                trade_id = self._emit_close_trade(
                    session, row, fills_by_symbol.get(symbol, []), now
                )
                if trade_id is not None:
                    trades_emitted.append(f"{symbol}/{strategy_id}")
                session.delete(row)
                removed.append(f"{symbol}/{strategy_id}")

        session.commit()

        result = {
            "added": added,
            "updated": updated,
            "removed": removed,
            "drifted": drifted,
            "trades_emitted": trades_emitted,
        }
        log.info("reconcile_complete", **{k: len(v) for k, v in result.items()})
        return result

    # ─── Trade emission ───────────────────────────────────────────────────────

    def _fetch_recent_sell_fills(
        self, symbols: List[str], now: datetime
    ) -> Dict[str, List[object]]:
        """Best-effort lookup of recent filled sell orders, grouped by symbol.

        Tolerates broker failure — if the call fails, we fall back to using
        cached `current_price`/`now` when constructing the Trade row.
        """
        if not symbols:
            return {}
        since = now - timedelta(hours=_FILL_LOOKBACK_HOURS)
        getter = getattr(self._broker, "get_filled_orders_since", None)
        if getter is None:
            return {}
        try:
            orders = getter(since=since, symbols=symbols)
        except Exception as e:
            log.warning("reconcile_fill_lookup_failed", error=str(e))
            return {}
        grouped: Dict[str, List[object]] = {}
        for o in orders:
            side = str(getattr(o, "side", "")).lower()
            # Long-only book — closes are sells. (Schema supports SHORT, but
            # CrossHorizonGatedPredictor zeros shorts; defensive future-proof.)
            if "sell" not in side:
                continue
            sym = getattr(o, "symbol", None)
            if sym is None:
                continue
            grouped.setdefault(sym, []).append(o)
        # Most-recent-first so the first match is the closing fill.
        for sym, lst in grouped.items():
            lst.sort(
                key=lambda o: getattr(o, "filled_at", None) or getattr(o, "submitted_at", now),
                reverse=True,
            )
        return grouped

    def _emit_close_trade(
        self,
        session: Session,
        row: Position,
        candidate_fills: List[object],
        now: datetime,
    ) -> Optional[object]:
        """Construct and add a Trade row for a Position the broker has closed.

        Skips emission (returns None) when there is no usable entry data —
        a Position with a null entry price or zero shares represents a
        bookkeeping ghost rather than a real fill, and emitting a Trade for
        it would just inject zeros into the realized-P&L stats.

        Also skips emission when a Trade row already exists for this
        (symbol, strategy_id, entry_ts) within the recent-close window —
        that means the trader-initiated close path
        (`emit_close_trade_at_submit`) already wrote the Trade with the
        correct exit_reason. In that case we *update* the existing row's
        exit_price + alpaca_order_id from the actual fill data instead of
        inserting a duplicate.
        """
        from zeus.execution.trade_emitter import (
            _RECENT_CLOSE_LOOKBACK,
            _as_utc as _emitter_as_utc,
        )

        shares = int(row.strategy_shares or row.qty or 0)
        if shares <= 0:
            return None
        entry_price = float(row.avg_entry_price) if row.avg_entry_price else None
        if entry_price is None or entry_price <= 0:
            return None

        fill = self._pick_matching_fill(candidate_fills, shares)
        if fill is not None:
            exit_price = float(fill.filled_avg_price)
            exit_ts = getattr(fill, "filled_at", None) or now
            alpaca_order_id = str(getattr(fill, "id", "") or "") or None
        else:
            # Fall back to last cached mark + now. This loses some precision
            # but is better than dropping the trade entirely.
            exit_price = float(row.current_price) if row.current_price else entry_price
            exit_ts = now
            alpaca_order_id = None

        entry_ts = row.entry_ts or row.updated_at or now
        # SQLite drops tz on DateTime(timezone=True) round-trips; postgres
        # preserves it. Normalize so arithmetic and comparisons are valid in
        # both environments.
        entry_ts = _as_utc(entry_ts)
        exit_ts = _as_utc(exit_ts)
        if exit_ts < entry_ts:
            # Clock skew or stale entry_ts — clamp.
            exit_ts = entry_ts

        # Idempotency: if the trader-initiated close path already emitted a
        # Trade row for this lot, refine it with the actual fill price
        # instead of inserting a duplicate. The existing row carries the
        # correct exit_reason (stop_loss / time_exit / signal_exit), which
        # is more informative than the broker_closed label we'd put on a
        # fresh row here.
        cutoff = _emitter_as_utc(now) - _RECENT_CLOSE_LOOKBACK
        existing = session.execute(
            select(Trade)
            .where(
                Trade.symbol == row.symbol,
                Trade.strategy_id == row.strategy_id,
                Trade.environment == self._env,
                Trade.entry_ts == entry_ts,
                Trade.exit_ts >= cutoff,
            )
            .limit(1)
        ).scalar_one_or_none()
        if existing is not None:
            existing.exit_price = exit_price
            existing.gross_pnl = (exit_price - entry_price) * shares
            existing.net_pnl = existing.gross_pnl
            existing.actual_return = (exit_price / entry_price) - 1.0 if entry_price else None
            if alpaca_order_id and not existing.alpaca_order_id:
                existing.alpaca_order_id = alpaca_order_id
            log.info(
                "trade_refined_from_reconciler",
                symbol=row.symbol, strategy_id=row.strategy_id,
                existing_exit_reason=existing.exit_reason,
                refined_exit_price=exit_price,
                attributed_fill=alpaca_order_id is not None,
            )
            return existing

        hold_days = (exit_ts - entry_ts).total_seconds() / 86400.0
        gross_pnl = (exit_price - entry_price) * shares
        actual_return = (exit_price / entry_price) - 1.0 if entry_price else None

        trade = Trade(
            symbol=row.symbol,
            strategy_id=row.strategy_id,
            strategy_model_version=row.strategy_model_version,
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
            exit_reason="broker_closed",
            peak_price=row.peak_price,
            entry_stop=row.hard_stop,
            actual_return=actual_return,
            environment=self._env,
            alpaca_order_id=alpaca_order_id,
        )
        session.add(trade)
        log.info(
            "trade_emitted_from_reconciler",
            symbol=row.symbol,
            strategy_id=row.strategy_id,
            shares=shares,
            entry_price=entry_price,
            exit_price=exit_price,
            net_pnl=gross_pnl,
            attributed_fill=alpaca_order_id is not None,
        )
        return trade

    @staticmethod
    def _pick_matching_fill(candidate_fills: List[Any], shares: int) -> Optional[Any]:
        """Pick the fill that best matches the strategy's share count.

        Multiple strategies on one symbol close into possibly-bundled fills.
        Prefer an exact share match; otherwise take the most recent fill.
        """
        if not candidate_fills:
            return None
        for o in candidate_fills:
            try:
                fq = int(float(getattr(o, "filled_qty", 0) or 0))
            except (TypeError, ValueError):
                fq = 0
            if fq == shares:
                return o
        return candidate_fills[0]
