"""APScheduler job functions for the 7-agent trading pipeline.

Timeline (America/New_York):

   Evening previous day
   ─────────────────────
   19:00   swing_research_eod_job           → swing_research briefs for tomorrow
   19:15   long_term_deep_dive_or_review    → optional memo / thesis refresh
   19:45   trader_overnight_plan_job        → swing + long_term trader decisions,
                                              plan stored on their strategy contexts
   20:00   overseer_daily_aggregate_job     → per-agent metrics + halts
   Sunday 10:00  swing_weekly_watchlist_job
   Sunday 11:00  overseer_weekly_review_job

   Morning
   ──────────
   04:30   premkt_candidate_prep_job (cheap — just refresh the candidate list)
   05:00, 07:00, 09:00  day_research_premarket_job
   09:00   day_trader_morning_plan_job       → writes today's day-trader plan
   09:28   trader_stage_orders_job           → trading_loop.stage_orders (hook)
   09:30   market_open_job (existing)        → executes entries from all 3 plans

   Intraday
   ────────
   */5 min (09:35-15:55) day_research_intraday_refresh_job
   */5 min (09:35-15:55) overseer_realtime_monitor_job

   Close
   ────────
   16:15   day_research_postmarket_wrap_job

Every job accepts an `Orchestrator` that bundles the AgentBundle + the
TradingLoop so jobs stay one-liners and failures get isolated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import structlog

from zeus.scheduler.context import get_orch

if TYPE_CHECKING:
    from zeus.agents.runtime import AgentBundle
    from zeus.live.strategy_manager import StrategyManager
    from zeus.live.trading_loop import TradingLoop

log = structlog.get_logger(__name__)


def _require_orch(name: str) -> Optional["AgentOrchestrator"]:
    """Fetch the live AgentOrchestrator. Returns None when the 7-agent
    system is disabled (no Anthropic key) or before runner setup runs —
    jobs short-circuit on None rather than crashing the worker."""
    orch = get_orch()
    if orch is None:
        log.warning("agent_job_skipped_no_orch", name=name)
    return orch


# ─── Orchestrator wrapper ─────────────────────────────────────────────────────


@dataclass
class AgentOrchestrator:
    """Glue between the TradingLoop and the AgentBundle. Owned by the runner.

    All the scheduler job functions below take one of these and delegate —
    keeping the job functions simple enough that failures are trivially
    contained per-job.
    """
    bundle: "AgentBundle"
    trading_loop: "TradingLoop"
    strategy_manager: "StrategyManager"
    regime_fn: Optional[Callable[[], str]] = None

    def current_regime(self) -> str:
        try:
            return (self.regime_fn() if self.regime_fn else "normal") or "normal"
        except Exception:
            return "normal"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _positions_for(
    orch: AgentOrchestrator, strategy_id: str,
) -> List[Dict[str, Any]]:
    """Current open positions scoped to one strategy, in the shape the
    trader LLM expects: {symbol, strategy_shares, entry_ts, market_value}."""
    from sqlalchemy import select
    from zeus.data.storage.database import Position, get_session_factory

    out: List[Dict[str, Any]] = []
    try:
        with get_session_factory()() as s:
            rows = s.execute(
                select(Position)
                .where(Position.strategy_id == strategy_id)
                .where(Position.strategy_shares > 0)
            ).scalars().all()
            for r in rows:
                out.append({
                    "symbol": r.symbol,
                    "shares": int(r.strategy_shares or 0),
                    "entry_ts": r.entry_ts.isoformat() if r.entry_ts is not None else None,
                    "market_value": float(r.market_value or 0.0),
                })
    except Exception as e:
        log.warning("positions_query_failed", strategy_id=strategy_id, error=str(e))
    return out


def _safe(name: str, fn: Callable[[], Any]) -> None:
    """Wrap a job body so a single agent blowing up never kills the scheduler."""
    log.info("agent_job_start", name=name)
    try:
        fn()
        log.info("agent_job_done", name=name)
    except Exception as e:
        log.error("agent_job_failed", name=name, error=str(e), exc_info=True)


# ─── Evening (previous session) ──────────────────────────────────────────────


def swing_research_eod_job() -> None:
    orch = _require_orch("swing_research_eod")
    if orch is None:
        return
    from zeus.agents.swing_research import SwingResearchContext
    from zeus.research.candidate_builder import build_swing_eod_candidates

    def _go():
        candidates = build_swing_eod_candidates()
        orch.bundle.swing_research.run_eod(SwingResearchContext(
            candidates=candidates,
            regime=orch.current_regime(),
        ))
    _safe("swing_research_eod", _go)


def long_term_deep_dive_or_review_job() -> None:
    """Alternate deep-dive / review. If any open LT position is missing a
    memo in the last 10 days, deep-dive that one; else pick a top-ADV name
    that hasn't been covered in the last 30 days."""
    orch = _require_orch("long_term_deep_dive_or_review")
    if orch is None:
        return
    from zeus.agents.long_term_research import DeepDiveContext
    from zeus.research.candidate_builder import pick_long_term_deep_dive_target

    def _go():
        symbol = pick_long_term_deep_dive_target()
        if symbol is None:
            log.info("long_term_deep_dive_skipped_no_target")
            return
        orch.bundle.long_term_research.run_deep_dive(DeepDiveContext(
            symbol=symbol, regime=orch.current_regime(),
        ))
    _safe("long_term_deep_dive_or_review", _go)


def long_term_monthly_review_job() -> None:
    """Monthly: re-examine every long-term holding against its original memo."""
    orch = _require_orch("long_term_monthly_review")
    if orch is None:
        return
    from zeus.agents.long_term_research import ReviewContext

    def _go():
        positions = _positions_for(orch, "long_term")
        if not positions:
            log.info("long_term_monthly_review_skipped_no_positions")
            return
        orch.bundle.long_term_research.run_monthly_review(ReviewContext(
            open_positions=positions, regime=orch.current_regime(),
        ))
    _safe("long_term_monthly_review", _go)


def swing_weekly_watchlist_job() -> None:
    orch = _require_orch("swing_weekly_watchlist")
    if orch is None:
        return
    from zeus.research.candidate_builder import build_swing_weekly_universe

    def _go():
        universe = build_swing_weekly_universe()
        if not universe:
            log.info("swing_weekly_watchlist_skipped_empty")
            return
        orch.bundle.swing_research.run_weekly_watchlist(
            universe=universe, regime=orch.current_regime(),
        )
    _safe("swing_weekly_watchlist", _go)


def _decide_and_attach_plan(
    orch: AgentOrchestrator,
    *, trader, strategy_id: str, plan_date: date,
) -> None:
    """Run a trader agent and attach its SessionPlan to the StrategyContext.

    After this, `StrategyManager.execute_entries_all()` at market open will
    pick up the plan and route it through the normal risk/execution path.
    """
    from zeus.agents.trader_base import TraderContext
    from zeus.research.candidate_builder import build_trader_predictions

    if orch.strategy_manager is None:
        log.warning("no_strategy_manager", strategy_id=strategy_id)
        return
    try:
        ctx = orch.strategy_manager.context(strategy_id)
    except KeyError:
        log.warning("strategy_context_missing", strategy_id=strategy_id)
        return

    predictions = build_trader_predictions(ctx)
    positions = _positions_for(orch, strategy_id)

    trader_ctx = TraderContext(
        strategy_id=strategy_id,
        model_version=getattr(ctx.model, "version_", "unknown"),
        feature_version=getattr(ctx.feature_pipeline, "feature_version", "v1"),
        regime=orch.current_regime(),
        predictions=predictions,
        current_positions=positions,
        plan_date=plan_date,
    )
    decision = trader.decide(trader_ctx)
    ctx.current_plan = decision.plan
    log.info(
        "trader_plan_attached",
        strategy_id=strategy_id,
        entries=len(decision.plan.entries),
        exits=len(decision.plan.exits),
    )


def trader_overnight_plan_job() -> None:
    """Swing + long-term traders decide the night before so their plans are
    ready to execute at market open.

    plan_date picks the very next market open after *now*: if called pre-open
    on a trading day, that's today; otherwise the next trading day. This keeps
    the scheduled 21:30 ET run (Tue→Wed) and pre-market catchup boot
    (Wed 01:00→Wed) both pointing at the correct session.
    """
    orch = _require_orch("trader_overnight_plan")
    if orch is None:
        return
    from datetime import date as _date, datetime as _dt
    from zoneinfo import ZoneInfo as _ZoneInfo
    from zeus.scheduler.market_schedule import (
        is_trading_day, market_open_time, next_trading_day,
    )

    def _next_session_date() -> _date:
        now_et = _dt.now(_ZoneInfo("America/New_York"))
        today = now_et.date()
        if is_trading_day(today):
            open_t = market_open_time(today)
            if open_t is not None and now_et < open_t:
                return today
        return next_trading_day(today)

    def _go():
        plan_date = _next_session_date()
        _decide_and_attach_plan(
            orch, trader=orch.bundle.swing_trader,
            strategy_id="swing", plan_date=plan_date,
        )
        _decide_and_attach_plan(
            orch, trader=orch.bundle.long_term_trader,
            strategy_id="long_term", plan_date=plan_date,
        )
    _safe("trader_overnight_plan", _go)


def overseer_daily_aggregate_job() -> None:
    orch = _require_orch("overseer_daily_aggregate")
    if orch is None:
        return
    def _go():
        orch.bundle.overseer.run_daily_aggregate()
    _safe("overseer_daily_aggregate", _go)


def overseer_weekly_review_job() -> None:
    orch = _require_orch("overseer_weekly_review")
    if orch is None:
        return
    def _go():
        orch.bundle.overseer.run_weekly_review()
    _safe("overseer_weekly_review", _go)


# ─── Morning (trading day) ───────────────────────────────────────────────────


def day_research_premarket_job() -> None:
    orch = _require_orch("day_research_premarket")
    if orch is None:
        return
    from zeus.agents.day_research import DayResearchContext
    from zeus.research.candidate_builder import build_day_premarket_candidates

    def _go():
        candidates = build_day_premarket_candidates()
        orch.bundle.day_research.run_premarket_scan(DayResearchContext(
            candidates=candidates,
            regime=orch.current_regime(),
        ))
    _safe("day_research_premarket", _go)


def day_trader_morning_plan_job() -> None:
    orch = _require_orch("day_trader_morning_plan")
    if orch is None:
        return
    from datetime import date as _date

    def _go():
        _decide_and_attach_plan(
            orch, trader=orch.bundle.day_trader,
            strategy_id="day", plan_date=_date.today(),
        )
    _safe("day_trader_morning_plan", _go)


# ─── Intraday ────────────────────────────────────────────────────────────────


def day_research_intraday_refresh_job() -> None:
    orch = _require_orch("day_research_intraday_refresh")
    if orch is None:
        return
    from zeus.agents.day_research import DayResearchContext
    from zeus.research.candidate_builder import build_day_premarket_candidates

    def _go():
        # Re-use the same candidate builder — what's moved in the last session
        # is what the intraday refresh should re-score.
        candidates = build_day_premarket_candidates(max_candidates=20)
        orch.bundle.day_research.run_intraday_refresh(DayResearchContext(
            candidates=candidates, regime=orch.current_regime(),
        ))
    _safe("day_research_intraday_refresh", _go)


def overseer_realtime_monitor_job() -> None:
    orch = _require_orch("overseer_realtime_monitor")
    if orch is None:
        return
    def _go():
        orch.bundle.overseer.run_realtime_monitor()
    _safe("overseer_realtime_monitor", _go)


# ─── Close ───────────────────────────────────────────────────────────────────


def day_research_postmarket_wrap_job() -> None:
    orch = _require_orch("day_research_postmarket_wrap")
    if orch is None:
        return
    from sqlalchemy import select
    from zeus.data.storage.database import Trade, get_session_factory
    from datetime import datetime, timedelta, timezone

    def _go():
        since = datetime.now(timezone.utc) - timedelta(days=1)
        realized: List[Dict[str, Any]] = []
        with get_session_factory()() as s:
            rows = s.execute(
                select(Trade)
                .where(Trade.strategy_id == "day")
                .where(Trade.exit_ts.is_not(None))
                .where(Trade.exit_ts >= since)
            ).scalars().all()
            for t in rows:
                realized.append({
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry": float(t.entry_price or 0),
                    "exit": float(t.exit_price or 0),
                    "shares": int(t.shares or 0),
                    "net_pnl": float(t.net_pnl or 0),
                    "exit_reason": t.exit_reason,
                    "hold_days": float(t.hold_days or 0),
                })
        orch.bundle.day_research.run_postmarket_wrap(
            realized_trades=realized, regime=orch.current_regime(),
        )
    _safe("day_research_postmarket_wrap", _go)


# ─── Job manifest ────────────────────────────────────────────────────────────


AGENT_JOB_MANIFEST: List[Dict[str, Any]] = [
    # Evening previous session (writes briefs/memos + overnight decisions).
    # Order matters: research agents publish briefs first, overseer audits
    # yesterday's trades, then the existing 21:00 model-plan job runs (from
    # JOB_MANIFEST), and finally the LLM traders refine those plans at 21:30
    # using the fresh research + model predictions.
    {"func": swing_research_eod_job,            "trigger": "cron", "hour": 19, "minute": 0,  "day_of_week": "mon-fri"},
    {"func": long_term_deep_dive_or_review_job, "trigger": "cron", "hour": 19, "minute": 15, "day_of_week": "mon-fri"},
    {"func": overseer_daily_aggregate_job,      "trigger": "cron", "hour": 20, "minute": 0,  "day_of_week": "mon-fri"},
    {"func": trader_overnight_plan_job,         "trigger": "cron", "hour": 21, "minute": 30, "day_of_week": "mon-fri"},

    # Weekend
    {"func": long_term_monthly_review_job,      "trigger": "cron", "hour": 10, "minute": 0,  "day_of_week": "sat", "day": "1-7"},
    {"func": swing_weekly_watchlist_job,        "trigger": "cron", "hour": 10, "minute": 0,  "day_of_week": "sun"},
    {"func": overseer_weekly_review_job,        "trigger": "cron", "hour": 11, "minute": 0,  "day_of_week": "sun"},

    # Morning (pre-open) — day-research multi-shot, then day-trader plans once
    {"func": day_research_premarket_job,        "trigger": "cron", "hour": 5,  "minute": 0,  "day_of_week": "mon-fri"},
    {"func": day_research_premarket_job,        "trigger": "cron", "hour": 7,  "minute": 0,  "day_of_week": "mon-fri"},
    {"func": day_research_premarket_job,        "trigger": "cron", "hour": 9,  "minute": 0,  "day_of_week": "mon-fri"},
    {"func": day_trader_morning_plan_job,       "trigger": "cron", "hour": 9,  "minute": 10, "day_of_week": "mon-fri"},

    # Intraday (market open).
    # `overseer_realtime_monitor_job` is intentionally NOT registered here —
    # its deterministic risk scan is folded into `intraday_monitor_job`
    # (JOB_MANIFEST, 15-min cadence) so we don't double-poll fills + journal
    # on overlapping crons. The function is still importable for direct
    # invocation from tests / one-shot ops scripts.
    {"func": day_research_intraday_refresh_job, "trigger": "cron", "hour": "9-15", "minute": "35,45,55,5,15,25", "day_of_week": "mon-fri"},

    # Close
    {"func": day_research_postmarket_wrap_job,  "trigger": "cron", "hour": 16, "minute": 15, "day_of_week": "mon-fri"},
]
