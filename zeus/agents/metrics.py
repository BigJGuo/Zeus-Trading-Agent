"""Per-agent rolling metrics: Sharpe, hit rate, expectancy, R-multiple.

Computed straight from `trades` (for filled outcomes) and `agent_journal`
(for activity counts — briefs written, decisions issued, alerts raised).

These feed the overseer's weekly reallocation loop and the paper-burn-in
"go/no-go" dashboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast

import structlog
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zeus.agents.journal import (
    KNOWN_AGENTS,
    PAIRED_RESEARCH,
    TRADER_AGENTS,
)
from zeus.data.storage.database import AgentJournal as JournalRow
from zeus.data.storage.database import Trade, get_session_factory

log = structlog.get_logger(__name__)


@dataclass
class AgentMetrics:
    """Trader metrics. 0 for `research_*` agents (they have no trades)."""
    agent_id: str
    window_days: int
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    hit_rate: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy_per_trade: float = 0.0
    avg_r_multiple: float = 0.0
    sharpe_annualized: float = 0.0
    max_drawdown_pct: float = 0.0
    mandate_violations: int = 0


@dataclass
class ResearchMetrics:
    """Research-agent output volume + coupling audit."""
    agent_id: str
    window_days: int
    n_briefs: int = 0
    n_memos: int = 0
    n_alerts: int = 0
    n_reviews: int = 0
    avg_confidence: Optional[float] = None
    trader_decoupled_trades: int = 0   # trader fills with no paired brief in window


# ─── Trader metrics ───────────────────────────────────────────────────────────


def compute_trader_metrics(
    agent_id: str,
    window_days: int = 28,
    environment: str = "paper",
    *,
    session: Optional[Session] = None,
) -> AgentMetrics:
    """Compute rolling-window metrics for a trader agent from `trades`."""
    if agent_id not in TRADER_AGENTS:
        raise ValueError(f"{agent_id!r} is not a trader agent")

    since = datetime.now(timezone.utc) - timedelta(days=window_days)

    def _run(s: Session) -> AgentMetrics:
        stmt = (
            select(Trade)
            .where(Trade.strategy_id == agent_id)
            .where(Trade.environment == environment)
            .where(Trade.exit_ts.is_not(None))
            .where(Trade.exit_ts >= since)
            .order_by(Trade.exit_ts.asc())
        )
        # SQLAlchemy legacy `Column(...)` typing surfaces `Column[X]` /
        # `ColumnElement` on attribute access; at runtime each `t.field` is
        # the underlying value. Rebind as `list[Any]` so arithmetic and
        # boolean ops below type-check cleanly.
        trades: list[Any] = cast("list[Any]", list(s.execute(stmt).scalars().all()))
        if not trades:
            return AgentMetrics(agent_id=agent_id, window_days=window_days)

        wins = [t for t in trades if (t.net_pnl or 0.0) > 0]
        losses = [t for t in trades if (t.net_pnl or 0.0) < 0]
        net = sum((t.net_pnl or 0.0) for t in trades)
        gross = sum((t.gross_pnl or 0.0) for t in trades)
        avg_win = (sum(t.net_pnl for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(t.net_pnl for t in losses) / len(losses)) if losses else 0.0
        expectancy = net / len(trades)

        # R-multiple: (exit - entry) / (entry - entry_stop) for longs.
        r_multiples: List[float] = []
        for t in trades:
            if (
                t.direction == "LONG"
                and t.entry_stop
                and t.entry_price
                and t.exit_price
                and t.entry_price > t.entry_stop
            ):
                risk = t.entry_price - t.entry_stop
                r_multiples.append((t.exit_price - t.entry_price) / risk)
        avg_r = (sum(r_multiples) / len(r_multiples)) if r_multiples else 0.0

        # Annualized Sharpe from daily PnL grouping by exit date.
        by_day: Dict[datetime, float] = {}
        for t in trades:
            d = t.exit_ts.date()
            by_day[d] = by_day.get(d, 0.0) + (t.net_pnl or 0.0)
        daily = list(by_day.values())
        sharpe = 0.0
        if len(daily) >= 2:
            mu = sum(daily) / len(daily)
            var = sum((x - mu) ** 2 for x in daily) / (len(daily) - 1)
            sigma = math.sqrt(var)
            if sigma > 0:
                sharpe = (mu / sigma) * math.sqrt(252)

        # Equity-curve drawdown from daily-cum PnL
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for x in daily:
            equity += x
            peak = max(peak, equity)
            if peak > 0:
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

        return AgentMetrics(
            agent_id=agent_id,
            window_days=window_days,
            n_trades=len(trades),
            n_wins=len(wins),
            n_losses=len(losses),
            hit_rate=(len(wins) / len(trades)) if trades else 0.0,
            gross_pnl=gross,
            net_pnl=net,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy_per_trade=expectancy,
            avg_r_multiple=avg_r,
            sharpe_annualized=sharpe,
            max_drawdown_pct=max_dd,
            mandate_violations=_count_mandate_violations(s, agent_id, since),
        )

    if session is not None:
        return _run(session)
    with get_session_factory()() as s:
        return _run(s)


def _count_mandate_violations(session: Session, agent_id: str, since: datetime) -> int:
    """Violations this window:
      - day trader holding > 5 days
      - swing trader holding > 15 days
      - long_term trader holding > 180 days (6mo spec ceiling)
    """
    cutoffs = {"day": 5.0, "swing": 15.0, "long_term": 180.0}
    cap = cutoffs.get(agent_id)
    if cap is None:
        return 0
    stmt = (
        select(func.count())
        .select_from(Trade)
        .where(Trade.strategy_id == agent_id)
        .where(Trade.exit_ts.is_not(None))
        .where(Trade.exit_ts >= since)
        .where(Trade.hold_days > cap)
    )
    return int(session.execute(stmt).scalar() or 0)


# ─── Research metrics ─────────────────────────────────────────────────────────


def compute_research_metrics(
    agent_id: str,
    window_days: int = 28,
    *,
    session: Optional[Session] = None,
) -> ResearchMetrics:
    """Output volume for a research agent, plus a count of paired trader
    fills with no matching brief (decoupling audit)."""
    if agent_id not in {"day_research", "swing_research", "long_term_research"}:
        raise ValueError(f"{agent_id!r} is not a research agent")

    since = datetime.now(timezone.utc) - timedelta(days=window_days)

    def _run(s: Session) -> ResearchMetrics:
        kind_counts_stmt = (
            select(JournalRow.kind, func.count(), func.avg(JournalRow.confidence))
            .where(JournalRow.agent_id == agent_id)
            .where(JournalRow.ts >= since)
            .group_by(JournalRow.kind)
        )
        by_kind = {row[0]: (int(row[1]), row[2]) for row in s.execute(kind_counts_stmt).all()}

        # Paired trader
        paired_trader = next(
            (t for t, r in PAIRED_RESEARCH.items() if r == agent_id), None,
        )
        decoupled = 0
        if paired_trader is not None:
            decoupled = _count_decoupled_fills(s, paired_trader, agent_id, since)

        # avg confidence across all entry kinds (weighted ignored — simple mean)
        all_conf_stmt = (
            select(func.avg(JournalRow.confidence))
            .where(JournalRow.agent_id == agent_id)
            .where(JournalRow.ts >= since)
            .where(JournalRow.confidence.is_not(None))
        )
        avg_conf = s.execute(all_conf_stmt).scalar()

        return ResearchMetrics(
            agent_id=agent_id,
            window_days=window_days,
            n_briefs=by_kind.get("brief", (0, None))[0],
            n_memos=by_kind.get("memo", (0, None))[0],
            n_alerts=by_kind.get("alert", (0, None))[0],
            n_reviews=by_kind.get("review", (0, None))[0],
            avg_confidence=float(avg_conf) if avg_conf is not None else None,
            trader_decoupled_trades=decoupled,
        )

    if session is not None:
        return _run(session)
    with get_session_factory()() as s:
        return _run(s)


def _count_decoupled_fills(
    session: Session, trader_id: str, research_id: str, since: datetime,
) -> int:
    """Count trader fills since `since` that have NO brief/memo/review from
    the paired research agent for the same symbol within the preceding 24h."""
    fills = list(
        session.execute(
            select(Trade.symbol, Trade.entry_ts)
            .where(Trade.strategy_id == trader_id)
            .where(Trade.entry_ts >= since)
        ).all()
    )
    decoupled = 0
    for symbol, entry_ts in fills:
        window_start = entry_ts - timedelta(hours=24)
        match = session.execute(
            select(func.count())
            .select_from(JournalRow)
            .where(JournalRow.agent_id == research_id)
            .where(JournalRow.symbol == symbol)
            .where(JournalRow.kind.in_(("brief", "memo", "review", "alert")))
            .where(JournalRow.ts >= window_start)
            .where(JournalRow.ts <= entry_ts)
        ).scalar()
        if int(match or 0) == 0:
            decoupled += 1
    return decoupled


# ─── Summary fan-out ──────────────────────────────────────────────────────────


def all_metrics(
    window_days: int = 28,
    environment: str = "paper",
    *,
    session: Optional[Session] = None,
) -> Dict[str, AgentMetrics | ResearchMetrics]:
    """Per-agent metrics for every known agent_id. Used by the overseer's
    daily aggregation and the `/api/agents/{id}/metrics` endpoint."""
    out: Dict[str, AgentMetrics | ResearchMetrics] = {}
    for aid in KNOWN_AGENTS:
        if aid in TRADER_AGENTS:
            out[aid] = compute_trader_metrics(aid, window_days, environment, session=session)
        elif aid in {"day_research", "swing_research", "long_term_research"}:
            out[aid] = compute_research_metrics(aid, window_days, session=session)
    return out
