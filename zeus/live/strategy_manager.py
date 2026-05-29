"""Multi-strategy orchestrator.

Owns 3 `StrategyContext` objects and delegates the session lifecycle to each:
  - plan_all()           — build SessionPlan per strategy (next-session planning)
  - execute_entries_all()— submit buys for every strategy, honoring global caps
  - execute_time_exits_all() — force-exit positions past their per-strategy
                               max_hold_days
  - summaries()          — per-strategy metric snapshot for Telegram/overseer

Shares a single broker + order_manager + risk_engine across strategies — the
broker sees the aggregate exposure. Per-strategy budgets come from the
StrategyAllocator.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import structlog

from zeus.config.strategies import GlobalLimits
from zeus.data.storage.database import Position, get_session_factory
from zeus.live.strategy import StrategyAllocator, StrategyBudget, StrategyContext
from zeus.live.trading_loop import SessionPlan

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker
    from zeus.execution.order_manager import OrderManager
    from zeus.risk.engine import RiskEngine

log = structlog.get_logger(__name__)


@dataclass
class StrategySummary:
    strategy_id: str
    model_version: str
    n_open_positions: int
    deployed_notional: float
    planned_entries: int
    planned_exits: int


class StrategyManager:
    """Orchestrates N concurrent trader agents sharing one broker account."""

    def __init__(
        self,
        contexts: Dict[str, StrategyContext],
        allocator: StrategyAllocator,
        broker: "AlpacaBroker",
        order_manager: "OrderManager",
        risk_engine: "RiskEngine",
        globals_: GlobalLimits,
        environment: str = "paper",
    ):
        if not contexts:
            raise ValueError("StrategyManager requires at least one StrategyContext")
        self._ctxs = contexts
        self._alloc = allocator
        self._broker = broker
        self._om = order_manager
        self._risk = risk_engine
        self._globals = globals_
        self._env = environment
        self._session_factory = get_session_factory()

    # ─── Introspection ────────────────────────────────────────────────────────
    def strategy_ids(self) -> List[str]:
        return list(self._ctxs.keys())

    def context(self, strategy_id: str) -> StrategyContext:
        return self._ctxs[strategy_id]

    # ─── Session planning ─────────────────────────────────────────────────────
    def plan_all(self, regime: str, as_of: Optional[date] = None) -> Dict[str, SessionPlan]:
        """Build a SessionPlan per enabled strategy. Delegates the heavy lifting
        to `run_next_session_planning_for_strategy` in after_hours.py."""
        from zeus.research.after_hours import run_next_session_planning_for_strategy

        as_of = as_of or date.today()
        account = self._broker.get_account()
        pv = float(account.portfolio_value or 0)
        bp = float(account.buying_power or 0)
        budgets = self._alloc.budgets(pv, bp)

        plans: Dict[str, SessionPlan] = {}
        for sid, ctx in self._ctxs.items():
            try:
                plan = run_next_session_planning_for_strategy(
                    ctx=ctx,
                    budget=budgets[sid],
                    broker=self._broker,
                    regime=regime,
                    as_of=as_of,
                )
                ctx.current_plan = plan
                plans[sid] = plan
                log.info(
                    "strategy_plan_built",
                    strategy_id=sid,
                    entries=len(plan.entries),
                    exits=len(plan.exits),
                )
            except Exception as e:
                log.error("strategy_plan_failed", strategy_id=sid, error=str(e))
        return plans

    # ─── Entry execution ──────────────────────────────────────────────────────
    def execute_entries_all(self) -> Dict[str, int]:
        """Submit planned entries for each strategy. Enforces global per-symbol
        overlap cap — if two strategies both want AAPL, the second gets rejected
        if the combined notional would exceed globals.global_max_position_pct.
        """
        if self._risk.kill_switch_active:
            log.warning("entries_blocked_kill_switch")
            return {sid: 0 for sid in self._ctxs}

        account = self._broker.get_account()
        pv = float(account.portfolio_value or 0)
        dd_state = self._risk.update_portfolio_value(pv)
        if dd_state.stop_new_entries:
            log.warning("entries_blocked_drawdown", level=dd_state.level.value)
            return {sid: 0 for sid in self._ctxs}

        # Build a picture of (symbol -> {strategy_id: notional}) already deployed
        # so the allocator can enforce global_max_position_pct when a second
        # strategy tries to enter the same name.
        existing: Dict[str, Dict[str, float]] = {}
        with self._session_factory() as session:
            for row in session.query(Position).filter(Position.environment == self._env).all():
                row_symbol = cast(str, row.symbol)
                row_strategy_id = cast(str, row.strategy_id)
                existing.setdefault(row_symbol, {})[row_strategy_id] = float(cast(Any, row.market_value) or 0.0)
            total_deployed = sum(v for d in existing.values() for v in d.values())

            from zeus.execution.execution_algo import execute_plan, plan_entry
            from zeus.risk.engine import ProposedTrade

            results: Dict[str, int] = {}
            for sid, ctx in self._ctxs.items():
                if getattr(self._risk, "is_agent_halted", lambda _id: False)(sid):
                    log.warning("strategy_halted_by_overseer", strategy_id=sid,
                                reason=self._risk.halted_agents().get(sid, "unknown"))
                    results[sid] = 0
                    continue

                plan = ctx.current_plan
                if plan is None or plan.plan_date != date.today():
                    log.info("strategy_no_plan", strategy_id=sid)
                    results[sid] = 0
                    continue

                portfolio = self._strategy_portfolio_snapshot(session, sid, account)
                submitted = 0
                rejected = 0

                for entry in plan.entries:
                    symbol = entry["symbol"]
                    shares = int(entry["shares"])
                    price = float(entry.get("target_price", 0.0))
                    if shares <= 0 or price <= 0:
                        rejected += 1
                        continue

                    notional = shares * price
                    # Global overlap gate
                    if not self._alloc.admits_global_overlap(
                        symbol, notional, existing.get(symbol, {}), pv
                    ):
                        log.info(
                            "overlap_rejected",
                            strategy_id=sid,
                            symbol=symbol,
                            reason="global_max_position_pct",
                        )
                        rejected += 1
                        continue
                    # Total exposure gate
                    if not self._alloc.admits_total_exposure(total_deployed + notional, pv):
                        log.info(
                            "total_exposure_rejected",
                            strategy_id=sid,
                            symbol=symbol,
                        )
                        rejected += 1
                        continue

                    trade = ProposedTrade(
                        symbol=symbol, shares=shares, price=price,
                        side="buy", sector=entry.get("sector"),
                        avg_daily_dollar_volume=entry.get("adv_usd"),
                    )
                    check = self._risk.pre_trade_check(trade, portfolio, dd_state)
                    if not check.approved:
                        if check.adjusted_shares and check.adjusted_shares > 0:
                            shares = check.adjusted_shares
                        else:
                            rejected += 1
                            continue

                    try:
                        # TWAP-slice entries above the per-slice threshold so
                        # we don't eat the full opening spread in one go.
                        # `schedule_twap_entry` submits slice 1 inline and
                        # schedules slices 2..N as DateTrigger jobs.
                        from zeus.execution.twap import schedule_twap_entry
                        schedule_twap_entry(
                            om=self._om,
                            broker=self._broker,
                            symbol=symbol,
                            total_shares=shares,
                            side="buy",
                            strategy_id=sid,
                        )
                        submitted += 1
                        total_deployed += notional
                        existing.setdefault(symbol, {})[sid] = (
                            existing.get(symbol, {}).get(sid, 0.0) + notional
                        )
                        # Upsert the Position row for this (symbol, strategy)
                        self._upsert_position(
                            session,
                            symbol=symbol,
                            strategy_id=sid,
                            shares=shares,
                            price=price,
                            stop=entry.get("stop") or price,
                            model_version=plan.model_version,
                        )
                    except Exception as e:
                        log.error(
                            "entry_submit_failed",
                            strategy_id=sid, symbol=symbol, error=str(e),
                        )
                        rejected += 1

                log.info(
                    "strategy_entries_complete",
                    strategy_id=sid, submitted=submitted, rejected=rejected,
                )
                results[sid] = submitted
            session.commit()
        return results

    # ─── Time-based exits ─────────────────────────────────────────────────────
    def execute_time_exits_all(self, now: Optional[datetime] = None) -> Dict[str, int]:
        """For each strategy, exit positions held >= max_hold_days."""
        if self._risk.kill_switch_active:
            return {sid: 0 for sid in self._ctxs}
        now = now or datetime.now(timezone.utc)
        results: Dict[str, int] = {}

        from zeus.execution.execution_algo import execute_plan, plan_exit_stop

        with self._session_factory() as session:
            for sid, ctx in self._ctxs.items():
                max_hold = ctx.config.max_hold_days
                rows = (
                    session.query(Position)
                    .filter(Position.strategy_id == sid)
                    .filter(Position.environment == self._env)
                    .all()
                )
                exited = 0
                for row in rows:
                    if row.entry_ts is None or row.strategy_shares is None:
                        continue
                    # SQLite drops tz on DateTime(timezone=True) round-trips;
                    # postgres preserves it. Normalize to UTC-aware so the
                    # subtraction below works in both environments.
                    entry_ts = row.entry_ts
                    if entry_ts.tzinfo is None:
                        entry_ts = entry_ts.replace(tzinfo=timezone.utc)
                    age_days = (now - entry_ts).total_seconds() / 86400.0
                    if age_days < max_hold:
                        continue
                    shares = int(cast(int, row.strategy_shares))
                    if shares <= 0:
                        continue
                    try:
                        # Pull a current mark for the placeholder exit_price.
                        # The reconciler will refine to the actual fill price
                        # once the sell lands. If the quote fetch fails we
                        # fall back to current_price on the Position row,
                        # which is the last-cached mark from reconciliation.
                        current_price = float(cast(Any, row.current_price) or cast(Any, row.avg_entry_price))
                        try:
                            quote = self._broker.get_latest_quote(cast(str, row.symbol))
                            bid = float(getattr(quote, "bid_price", None) or 0)
                            ask = float(getattr(quote, "ask_price", None) or 0)
                            if bid > 0 and ask > 0:
                                current_price = (bid + ask) / 2
                        except Exception:
                            pass

                        # Emit Trade row at submit time with the correct
                        # exit_reason='time_exit'. The reconciler's catch-net
                        # path skips re-emit via the idempotency check.
                        from zeus.execution.trade_emitter import emit_close_trade_at_submit
                        emit_close_trade_at_submit(
                            session, row,
                            shares=shares,
                            exit_price=current_price,
                            exit_reason="time_exit",
                            environment=self._env,
                        )
                        session.commit()

                        plan = plan_exit_stop()
                        execute_plan(
                            self._om, session, cast(str, row.symbol), shares, "sell", plan,
                            strategy_id=sid,
                        )
                        exited += 1
                        log.info(
                            "time_exit",
                            strategy_id=sid, symbol=row.symbol,
                            age_days=round(age_days, 2),
                            max_hold_days=max_hold,
                        )
                    except Exception as e:
                        log.error(
                            "time_exit_failed",
                            strategy_id=sid, symbol=row.symbol, error=str(e),
                        )
                results[sid] = exited
            session.commit()
        return results

    # ─── Summaries ────────────────────────────────────────────────────────────
    def summaries(self) -> List[StrategySummary]:
        out: List[StrategySummary] = []
        with self._session_factory() as session:
            for sid, ctx in self._ctxs.items():
                rows = (
                    session.query(Position)
                    .filter(Position.strategy_id == sid)
                    .filter(Position.environment == self._env)
                    .all()
                )
                deployed = sum(float(cast(Any, r.market_value) or 0.0) for r in rows)
                plan = ctx.current_plan
                out.append(StrategySummary(
                    strategy_id=sid,
                    model_version=getattr(ctx.model, "version_", "unknown"),
                    n_open_positions=len(rows),
                    deployed_notional=deployed,
                    planned_entries=len(plan.entries) if plan else 0,
                    planned_exits=len(plan.exits) if plan else 0,
                ))
        return out

    # ─── Helpers ──────────────────────────────────────────────────────────────
    def _strategy_portfolio_snapshot(self, session, strategy_id: str, account) -> Any:
        """Build a per-strategy PortfolioSnapshot so risk checks (sector cap,
        position cap) are scoped to this strategy's own slice."""
        from zeus.risk.engine import PortfolioSnapshot

        rows = (
            session.query(Position)
            .filter(Position.strategy_id == strategy_id)
            .filter(Position.environment == self._env)
            .all()
        )
        positions: Dict[str, dict] = {}
        for r in rows:
            positions[cast(str, r.symbol)] = {
                "shares": int(cast(Any, r.strategy_shares) or cast(Any, r.qty)),
                "market_value": float(cast(Any, r.market_value) or 0.0),
                "sector": None,
            }
        return PortfolioSnapshot(
            portfolio_value=float(account.portfolio_value or 0),
            cash_balance=float(account.cash or 0),
            buying_power=float(account.buying_power or 0),
            positions=positions,
        )

    def _upsert_position(
        self, session, *, symbol: str, strategy_id: str, shares: int,
        price: float, stop: float, model_version: str,
    ) -> None:
        """Insert or update the Position row for (symbol, strategy_id)."""
        from sqlalchemy import and_, select
        now = datetime.now(timezone.utc)
        row = session.execute(
            select(Position).where(
                and_(Position.symbol == symbol, Position.strategy_id == strategy_id)
            )
        ).scalar_one_or_none()
        if row is None:
            row = Position(
                symbol=symbol,
                strategy_id=strategy_id,
                qty=shares,
                strategy_shares=shares,
                avg_entry_price=price,
                current_price=price,
                market_value=shares * price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                hard_stop=stop,
                peak_price=price,
                entry_ts=now,
                strategy_model_version=model_version,
                environment=self._env,
                updated_at=now,
            )
            session.add(row)
        else:
            # Averaging in on a repeat entry
            total_shares = int(row.strategy_shares or 0) + shares
            if total_shares > 0:
                row.avg_entry_price = (
                    (row.avg_entry_price * (row.strategy_shares or 0)) + price * shares
                ) / total_shares
            row.strategy_shares = total_shares
            row.qty = total_shares
            row.updated_at = now
