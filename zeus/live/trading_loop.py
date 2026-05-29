"""Main trading loop — orchestrates all session jobs.

This module defines TradingLoop, which is driven by APScheduler (see
zeus.scheduler.runner). Each public method is invoked by a scheduled job.

Dependencies are injected via the constructor so the loop can be unit-tested
with mocks.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import structlog

from zeus.config.settings import get_settings
from zeus.data.storage.database import (
    Heartbeat,
    SystemMetrics,
    get_session_factory,
)
from zeus.live.kill_switch import KillSwitch
from zeus.live.session_manager import SessionState, classify_session
from zeus.risk.engine import PortfolioSnapshot, ProposedTrade, RiskEngine

if TYPE_CHECKING:
    from zeus.execution.alpaca_broker import AlpacaBroker
    from zeus.execution.order_manager import OrderManager
    from zeus.execution.reconciler import PositionReconciler
    from zeus.monitoring.telegram_bot import TelegramNotifier
    from zeus.models.base import BaseModel

log = structlog.get_logger(__name__)


def _extract_bid_ask(quote, fallback: float) -> tuple[float, float]:
    """Normalize a broker quote (Alpaca Quote object, dict, or None) to (bid, ask).

    Alpaca returns a Quote object with .bid_price / .ask_price; tests pass a
    plain dict. Fall back to the caller-supplied price if neither shape is
    present, so we never place an order against NaN.
    """
    if quote is None:
        return fallback, fallback
    if isinstance(quote, dict):
        return float(quote.get("bid", fallback)), float(quote.get("ask", fallback))
    bid = getattr(quote, "bid_price", None) or getattr(quote, "bid", None) or fallback
    ask = getattr(quote, "ask_price", None) or getattr(quote, "ask", None) or fallback
    return float(bid), float(ask)


@dataclass
class SessionPlan:
    """Pre-computed plan for tomorrow's session."""
    plan_date: date
    regime: str
    entries: list           # list of dicts: {symbol, shares, target_price, stop, signal_score}
    exits: list             # list of symbols to close
    holds: list             # list of symbols to hold
    model_version: str
    feature_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_date": self.plan_date.isoformat(),
            "regime": self.regime,
            "entries": self.entries,
            "exits": self.exits,
            "holds": self.holds,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionPlan":
        return cls(
            plan_date=date.fromisoformat(d["plan_date"]),
            regime=d["regime"],
            entries=d.get("entries", []),
            exits=d.get("exits", []),
            holds=d.get("holds", []),
            model_version=d.get("model_version", ""),
            feature_version=d.get("feature_version", ""),
        )


class TradingLoop:
    """
    Stateful orchestrator. One instance per running process.

    Responsibilities:
      - Load & persist the next-session plan
      - Execute entries/exits at market open
      - Monitor stops intraday, enforce risk
      - Run after-hours research pipeline (delegated to research.after_hours)
      - Emit heartbeats and Telegram summaries
    """

    STARTING_CAPITAL = 100_000.0

    def __init__(
        self,
        broker: "AlpacaBroker",
        order_manager: "OrderManager",
        reconciler: "PositionReconciler",
        telegram: "TelegramNotifier",
        risk_engine: RiskEngine,
        kill_switch: KillSwitch,
        environment: str = "paper",
    ):
        self._broker = broker
        self._om = order_manager
        self._reconciler = reconciler
        self._tg = telegram
        self._risk = risk_engine
        self._kill = kill_switch
        self._env = environment
        self._settings = get_settings()
        self._session_factory = get_session_factory()
        self._current_plan: Optional[SessionPlan] = None
        self._plans: Dict[str, SessionPlan] = {}     # per-strategy plans (multi-strategy mode)
        self._strategy_mgr: Any = None               # StrategyManager, set via attach_strategy_manager
        self._after_hours_ctx: Any = None            # set via attach_research_context
        self._agent_orchestrator: Any = None         # set by runner.py when agent system is enabled
        self._agent_allocator: Any = None
        self._paused = False

    # ─── Manual control (Telegram commands) ───────────────────────────────────
    def pause(self) -> None:
        self._paused = True
        log.warning("trading_paused")

    def resume(self) -> None:
        self._paused = False
        log.info("trading_resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ─── Lifecycle ────────────────────────────────────────────────────────────
    def startup(self) -> None:
        """Called once when the process starts."""
        log.info("trading_loop_startup", environment=self._env)

        if KillSwitch.is_tripped(self._settings.artifacts_path):
            log.critical("STARTUP_BLOCKED_KILL_SWITCH_TRIPPED")
            self._tg.send_sync(
                "🆘 STARTUP BLOCKED: kill switch lockfile present. "
                "Remove lockfile manually after review."
            )
            raise SystemExit(1)

        with self._session_factory() as session:
            self._reconciler.reconcile(session)

        account = self._broker.get_account()
        self._risk.update_portfolio_value(float(account.portfolio_value or 0))
        self._load_plan_for_today()
        self._tg.send_startup(
            environment=self._env, portfolio_value=float(account.portfolio_value or 0),
        )

    # ─── Plan management ──────────────────────────────────────────────────────
    def _plan_path(self, d: date) -> str:
        return os.path.join(
            self._settings.artifacts_path,
            "knowledge",
            "session_plans",
            f"{d.isoformat()}.json",
        )

    def _load_plan_for_today(self) -> None:
        path = self._plan_path(date.today())
        if not os.path.exists(path):
            log.info("no_plan_found", date=str(date.today()))
            self._current_plan = None
            return
        with open(path) as f:
            self._current_plan = SessionPlan.from_dict(json.load(f))
        log.info(
            "plan_loaded",
            date=str(self._current_plan.plan_date),
            entries=len(self._current_plan.entries),
        )

    def save_plan(self, plan: SessionPlan) -> str:
        path = self._plan_path(plan.plan_date)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        log.info("plan_saved", path=path, entries=len(plan.entries))
        return path

    # ─── Scheduled job methods ────────────────────────────────────────────────
    def run_premarket_summary(self) -> None:
        session_state = classify_session()
        if session_state == SessionState.WEEKEND:
            return
        account = self._broker.get_account()
        dd_state = self._risk.update_portfolio_value(float(account.portfolio_value or 0))

        # Multi-strategy: aggregate entries/exits across the attached plans.
        if self._strategy_mgr is not None:
            all_entries, all_exits, regime = [], [], "UNKNOWN"
            for sid in self._strategy_mgr.strategy_ids():
                ctx = self._strategy_mgr.context(sid)
                p = ctx.current_plan
                if p is None:
                    continue
                regime = p.regime
                all_entries.extend([{**e, "strategy_id": sid} for e in p.entries])
                all_exits.extend(p.exits)
            top = "\n".join(
                f"  • [{e.get('strategy_id')}] {e.get('symbol')} "
                f"(score {e.get('signal_score', 0):.2f})"
                for e in all_entries[:8]
            ) or "  (none)"
            self._tg.send_premarket_summary(
                planned_entries=len(all_entries),
                planned_exits=len(all_exits),
                regime=regime,
                portfolio_value=float(account.portfolio_value or 0),
                top_signals=top,
            )
            return

        self._load_plan_for_today()
        entries = self._current_plan.entries if self._current_plan else []
        top = "\n".join(
            f"  • {e.get('symbol')}  (score {e.get('signal_score', 0):.2f})"
            for e in entries[:5]
        ) or "  (none)"
        self._tg.send_premarket_summary(
            planned_entries=len(entries),
            planned_exits=len(self._current_plan.exits) if self._current_plan else 0,
            regime=self._current_plan.regime if self._current_plan else "UNKNOWN",
            portfolio_value=float(account.portfolio_value or 0),
            top_signals=top,
        )

    def run_manual_trim(self) -> None:
        """Execute operator-authored sell orders listed in
        `artifacts/manual_trims/<YYYY-MM-DD>.json` before the day's normal
        entries fire. Used to trim legacy / orphaned positions that aren't
        owned by any active strategy.

        File shape: `[{"symbol": "ABT", "shares": 164, "side": "sell"}, ...]`.
        After successful processing the file is renamed to `<...>.done` so a
        re-fire (misfire catch-up, manual retry) can't double-submit.
        Best-effort: any broker error on one symbol is logged but doesn't
        stop the other trims, and never blocks the subsequent execute_entries.
        """
        try:
            today = date.today().isoformat()
            trims_dir = os.path.join("artifacts", "manual_trims")
            path = os.path.join(trims_dir, f"{today}.json")
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                trims = json.load(f)
            if not isinstance(trims, list) or not trims:
                log.info("manual_trim_empty", path=path)
                return
            if self._risk.kill_switch_active:
                log.warning("manual_trim_skipped_kill_switch", path=path)
                return
            submitted, failed = 0, 0
            for t in trims:
                symbol = t.get("symbol")
                shares = int(t.get("shares") or 0)
                side = (t.get("side") or "sell").lower()
                if not symbol or shares <= 0:
                    log.warning("manual_trim_bad_entry", entry=t)
                    continue
                try:
                    order = self._broker.submit_market_order(symbol, shares, side)
                    submitted += 1
                    log.info(
                        "manual_trim_submitted",
                        symbol=symbol, shares=shares, side=side,
                        order_id=getattr(order, "id", None),
                    )
                except Exception as e:
                    failed += 1
                    log.error(
                        "manual_trim_failed",
                        symbol=symbol, shares=shares, side=side, error=str(e),
                    )
            try:
                os.rename(path, path + ".done")
            except Exception as e:
                log.warning("manual_trim_rename_failed", path=path, error=str(e))
            log.info("manual_trim_complete", submitted=submitted, failed=failed)
        except Exception as e:
            log.error("manual_trim_error", error=str(e))

    def execute_entries(self) -> None:
        if self._risk.kill_switch_active:
            return
        if self._paused:
            log.info("entries_skipped_paused")
            return

        # Multi-strategy path: delegate to StrategyManager so each strategy's
        # plan executes against its own budget with global overlap enforcement.
        if self._strategy_mgr is not None:
            results = self._strategy_mgr.execute_entries_all()
            log.info("multi_strategy_entries_complete", **results)
            return

        if self._current_plan is None or self._current_plan.plan_date != date.today():
            self._load_plan_for_today()
        if not self._current_plan:
            log.warning("no_plan_to_execute")
            return

        with self._session_factory() as session:
            self._reconciler.reconcile(session)
            account = self._broker.get_account()
            pv = float(account.portfolio_value or 0)
            dd_state = self._risk.update_portfolio_value(pv)

            if dd_state.stop_new_entries:
                log.warning("entries_blocked_drawdown", level=dd_state.level.value)
                return

            portfolio = self._build_portfolio_snapshot(account)

            from zeus.execution.execution_algo import execute_plan, plan_entry

            submitted = 0
            rejected = 0
            for entry in self._current_plan.entries:
                symbol = entry["symbol"]
                shares = int(entry["shares"])
                price = float(entry.get("target_price", 0.0))
                sector = entry.get("sector")

                trade = ProposedTrade(
                    symbol=symbol, shares=shares, price=price,
                    side="buy", sector=sector,
                    avg_daily_dollar_volume=entry.get("adv_usd"),
                )
                check = self._risk.pre_trade_check(trade, portfolio, dd_state)
                if not check.approved:
                    log.info("entry_rejected", symbol=symbol, reason=check.reason)
                    rejected += 1
                    if check.adjusted_shares and check.adjusted_shares > 0:
                        shares = check.adjusted_shares
                    else:
                        continue

                try:
                    # TWAP-slice entries through the first ~12 min of the
                    # open instead of eating the full 09:30 spread in a
                    # single market order.
                    from zeus.execution.twap import schedule_twap_entry
                    schedule_twap_entry(
                        om=self._om,
                        broker=self._broker,
                        symbol=symbol,
                        total_shares=shares,
                        side="buy",
                        strategy_id="legacy",
                    )
                    submitted += 1
                    self._tg.send_trade_open(
                        symbol=symbol, shares=shares, price=price,
                        signal_score=entry.get("signal_score", 0.0),
                        confidence=entry.get("confidence", 0.0),
                        stop=entry.get("stop") or price,
                        regime=self._current_plan.regime,
                    )
                except Exception as e:
                    log.error("entry_submit_failed", symbol=symbol, error=str(e))

            log.info("entries_complete", submitted=submitted, rejected=rejected)

    def check_stops_and_risk(self) -> None:
        if self._risk.kill_switch_active:
            return

        with self._session_factory() as session:
            try:
                refreshed = self._om.refresh_all_open(session)
                log.info("orders_refreshed", count=len(refreshed))
            except Exception as e:
                log.error("order_refresh_failed", error=str(e))

            positions = self._broker.get_positions()
            account = self._broker.get_account()
            pv = float(account.portfolio_value or 0)

            dd_state = self._risk.update_portfolio_value(pv)

            if dd_state.close_all:
                log.critical("drawdown_kill_trigger", level=dd_state.level.value)
                self._kill.activate(
                    f"Drawdown {dd_state.level.name}: -${dd_state.drawdown_usd:,.0f}",
                    session=session,
                )
                return

            self._check_position_stops(session, positions)
            self._emit_system_metrics(session, account, dd_state)

    def _check_position_stops(self, session, positions) -> None:
        """Check every open position against its stored stop.

        Positions are keyed on (symbol, strategy_id) — a single broker symbol
        may have rows in multiple strategies. Iterate per strategy row and
        close only that strategy's share count on a stop trigger.
        """
        from sqlalchemy import select
        from zeus.data.storage.database import Position
        from zeus.risk.stop_logic import should_exit_on_stop, update_trailing_stop
        from zeus.execution.execution_algo import plan_exit_stop, execute_plan

        for pos in positions:
            try:
                current_price = float(getattr(pos, "current_price", None) or 0)
                if current_price <= 0:
                    continue
                rows = session.execute(
                    select(Position).where(Position.symbol == pos.symbol)
                ).scalars().all()
                if not rows:
                    continue

                for db_row in rows:
                    if db_row.hard_stop is None:
                        continue
                    strategy_id = getattr(db_row, "strategy_id", "legacy")
                    strat_shares = int(
                        getattr(db_row, "strategy_shares", None)
                        or getattr(db_row, "shares", 0)
                        or 0
                    )
                    if strat_shares <= 0:
                        continue

                    new_stop, new_peak = update_trailing_stop(
                        current_price=current_price,
                        entry_price=db_row.avg_entry_price,
                        peak_price=db_row.peak_price or db_row.avg_entry_price,
                        hard_stop=db_row.hard_stop,
                        trailing_pct=db_row.trailing_pct or 0.5,
                        trailing_activation_price=db_row.avg_entry_price * 1.02,
                    )
                    if new_stop != db_row.hard_stop or new_peak != db_row.peak_price:
                        db_row.hard_stop = new_stop
                        db_row.peak_price = new_peak
                        session.commit()

                    if should_exit_on_stop(current_price, db_row.hard_stop):
                        log.warning(
                            "stop_triggered",
                            symbol=pos.symbol,
                            strategy_id=strategy_id,
                            shares=strat_shares,
                            price=current_price,
                            stop=db_row.hard_stop,
                        )
                        # Emit the Trade row BEFORE submitting the sell so the
                        # correct exit_reason='stop_loss' is captured. The
                        # reconciler's catch-net path will skip re-emitting
                        # this same close because of the idempotency check.
                        from zeus.execution.trade_emitter import emit_close_trade_at_submit
                        emit_close_trade_at_submit(
                            session, db_row,
                            shares=strat_shares,
                            exit_price=current_price,
                            exit_reason="stop_loss",
                            environment=self._env,
                        )
                        session.commit()
                        plan = plan_exit_stop()
                        execute_plan(
                            self._om, session, pos.symbol, strat_shares, "sell", plan,
                        )
                        self._tg.send_trade_close(
                            symbol=pos.symbol,
                            shares=strat_shares,
                            entry_price=float(cast(Any, db_row.avg_entry_price) or 0),
                            exit_price=current_price,
                            net_pnl=float(getattr(pos, "unrealized_pl", 0.0) or 0.0),
                            hold_days=0.0,
                            exit_reason="stop_loss",
                        )
            except Exception as e:
                log.error("stop_check_failed", symbol=pos.symbol, error=str(e))

    def run_closing(self) -> None:
        log.info("run_closing")
        # Multi-strategy: each agent enforces its own max_hold_days here.
        if self._strategy_mgr is not None:
            results = self._strategy_mgr.execute_time_exits_all()
            log.info("multi_strategy_time_exits_complete", **results)
            return
        # v1 single-agent: no forced time exits — positions carry overnight.

    def run_eod_report(self) -> None:
        account = self._broker.get_account()
        pv = float(account.portfolio_value or 0)
        dd_state = self._risk.update_portfolio_value(pv)

        with self._session_factory() as session:
            self._reconciler.reconcile(session)
            self._emit_system_metrics(session, account, dd_state)

        daily_pnl = float(getattr(account, "equity", pv) or pv) - float(
            getattr(account, "last_equity", pv) or pv
        )

        self._tg.send_daily_pnl(
            date_str=date.today().isoformat(),
            daily_pnl=daily_pnl,
            portfolio_value=pv,
            drawdown_usd=dd_state.drawdown_usd,
            open_positions=len(self._broker.get_positions()),
            win_count=0,   # v1: per-trade W/L attribution not yet wired
            loss_count=0,
            regime=self._current_plan.regime if self._current_plan else "UNKNOWN",
        )

    def _research_ctx(self):
        return getattr(self, "_after_hours_ctx", None)

    def attach_research_context(self, ctx) -> None:
        self._after_hours_ctx = ctx

    def attach_strategy_manager(self, manager) -> None:
        """Wire in the multi-strategy manager. When attached, execute_entries
        and run_closing delegate to per-strategy logic instead of the
        single-plan path."""
        self._strategy_mgr = manager
        log.info("strategy_manager_attached", strategies=manager.strategy_ids())

    def run_data_refresh(self) -> None:
        ctx = self._research_ctx()
        if ctx is None:
            log.warning("no_research_context")
            return
        from zeus.research.after_hours import run_data_refresh
        run_data_refresh(ctx)

    def run_intraday_backfill(self, lookback_minutes: int = 120) -> None:
        """Pull recent 5-min bars for the live watchlist so day_research has
        fresh intraday data. Safe to call mid-session; skips if market shut."""
        try:
            from zeus.data.ingestion.intraday_alpaca import (
                IntradayIngester, build_watchlist,
            )
            watchlist = build_watchlist(self._broker)
            if not watchlist:
                log.info("intraday_backfill_empty_watchlist")
                return
            ingester = IntradayIngester(self._broker)
            ingester.backfill_intraday(
                watchlist, lookback_minutes=lookback_minutes, timeframe="5m"
            )
        except Exception as e:
            log.warning("intraday_backfill_failed", error=str(e))

    def run_feature_engineering(self) -> None:
        ctx = self._research_ctx()
        if ctx is None:
            return
        from zeus.research.after_hours import run_feature_engineering
        run_feature_engineering(ctx)

    def run_next_session_planning(self) -> None:
        ctx = self._research_ctx()
        if ctx is None:
            return

        # Multi-strategy path: build a plan per strategy so each agent decides
        # against its own model + budget. Regime is detected once (shared input).
        if self._strategy_mgr is not None:
            from zeus.research.after_hours import _load_regime_series  # noqa: WPS437
            with self._session_factory() as session:
                spy, vix = _load_regime_series(session)
            regime = ctx.regime_detector.detect(spy, vix)
            log.info("regime_detected", regime=regime)
            plans = self._strategy_mgr.plan_all(regime=regime)
            self._plans = plans
            return

        from zeus.research.after_hours import run_next_session_planning
        plan = run_next_session_planning(ctx)
        self._current_plan = plan

    def run_company_research(self) -> None:
        ctx = self._research_ctx()
        if ctx is None:
            return
        from zeus.research.company_research import run_company_research
        run_company_research(ctx)

    def run_nightly_retrain(self) -> None:
        ctx = self._research_ctx()
        if ctx is None:
            return
        from zeus.research.after_hours import run_nightly_retrain
        result = run_nightly_retrain(ctx)
        self._tg.send_model_retrain(
            model_name="xgb_return",
            version=result.get("version_string", "unknown"),
            promoted=result.get("passed_gates", False),
            metrics=result.get("metrics", {}),
        )

    # ─── Heartbeat + metrics ──────────────────────────────────────────────────
    def emit_heartbeat(self) -> None:
        try:
            account = self._broker.get_account()
            pv = float(account.portfolio_value or 0)
            dd_state = self._risk.update_portfolio_value(pv)
            with self._session_factory() as session:
                session.add(Heartbeat(
                    ts=datetime.now(timezone.utc),
                    component="trading_loop",
                    status="ok",
                    details={"portfolio_value": pv, "drawdown_usd": dd_state.drawdown_usd},
                ))
                session.commit()
            position_count = len(self._broker.get_positions())
            regime = self._current_plan.regime if self._current_plan else "UNKNOWN"
            self._tg.send_heartbeat(
                portfolio_value=pv,
                drawdown_usd=dd_state.drawdown_usd,
                position_count=position_count,
                regime=regime,
            )
        except Exception as e:
            log.error("heartbeat_failed", error=str(e))

    def _emit_system_metrics(self, session, account, dd_state) -> None:
        try:
            session.add(SystemMetrics(
                ts=datetime.now(timezone.utc),
                portfolio_value=float(account.portfolio_value or 0),
                cash_balance=float(account.cash or 0),
                unrealized_pnl=float(getattr(account, "unrealized_pl", 0.0) or 0.0),
                realized_pnl_daily=float(getattr(account, "realized_pl_daily", 0.0) or 0.0),
                drawdown_from_peak_usd=dd_state.drawdown_usd,
                drawdown_from_peak_pct=dd_state.drawdown_pct,
                open_position_count=len(self._broker.get_positions()),
                regime=self._current_plan.regime if self._current_plan else None,
                environment=self._env,
            ))
            session.commit()
        except Exception as e:
            log.error("metrics_emit_failed", error=str(e))

    def _build_portfolio_snapshot(self, account) -> PortfolioSnapshot:
        positions = {}
        for p in self._broker.get_positions():
            positions[p.symbol] = {
                "shares": int(p.qty),
                "market_value": float(getattr(p, "market_value", None) or 0),
                "sector": None,  # filled by portfolio constructor when building entries
            }
        return PortfolioSnapshot(
            portfolio_value=float(account.portfolio_value or 0),
            cash_balance=float(account.cash or 0),
            buying_power=float(account.buying_power or 0),
            positions=positions,
        )
