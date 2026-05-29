"""Overseer Agent (A7): portfolio manager + risk overseer + reallocator.

Cadences (scheduled by APScheduler):
  - Daily 20:00: aggregate per-agent metrics, write `kind='decision'` journal
    entry, run circuit-breaker checks (drawdown, mandate drift).
  - Weekly Sunday 11:00: LLM-driven post-mortem + allocation review. May
    propose a reallocation via `OverseerStrategyAllocator.reallocate(...)`.
  - Real-time 5-min monitor during market hours: lightweight drift +
    research-trader coupling check. Halts agents on critical breaches.

The weekly review uses an LLM (Opus tier) so the narrative post-mortem can
reason about why an agent is up/down and whether the reallocation shift is
warranted. The daily aggregate + real-time monitor are deterministic — no
LLM — so they remain cheap to run and easy to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import structlog

from zeus.agents.base import AgentLoop, AgentRunResult
from zeus.agents.journal import (
    AgentJournal,
    KNOWN_AGENTS,
    PAIRED_RESEARCH,
    TRADER_AGENTS,
    audit_research_coupling,
    query_journal,
)
from zeus.agents.metrics import (
    AgentMetrics,
    ResearchMetrics,
    compute_research_metrics,
    compute_trader_metrics,
)
from zeus.data.storage.database import Trade, get_session_factory
from zeus.llm.client import AnthropicClient
from zeus.risk.engine import RiskEngine
from zeus.risk.strategy_allocator import AllocationUpdate, OverseerStrategyAllocator

log = structlog.get_logger(__name__)


# Circuit-breaker thresholds.
# Phase 5 defaults; tunable via config later.
MAX_DD_HALT_PCT = 0.15              # 15% drawdown → halt
MIN_SHARPE_REVIEW = 0.0             # below = flag for reallocation
MAX_MANDATE_VIOLATIONS_HALT = 3     # >3 mandate violations in window → halt

# Research coupling: a trader fill with no paired brief/memo inside this
# window is considered decoupled.
RESEARCH_COUPLING_WINDOW_HOURS = 24


@dataclass
class OverseerReport:
    ts: datetime
    metrics: Dict[str, AgentMetrics | ResearchMetrics]
    halts_issued: List[tuple[str, str]] = field(default_factory=list)  # (agent, reason)
    decoupled_fills: List[Dict[str, Any]] = field(default_factory=list)
    journal_id: Optional[int] = None


@dataclass
class WeeklyReview:
    ts: datetime
    allocation_update: Optional[AllocationUpdate]
    agent_run: AgentRunResult


class OverseerAgent:
    """Orchestrates daily aggregation, weekly reallocation, real-time monitor."""

    def __init__(
        self,
        *,
        llm: AnthropicClient,
        risk_engine: RiskEngine,
        allocator: OverseerStrategyAllocator,
        regime_fn: Optional[Callable[[], str]] = None,
    ):
        self._llm = llm
        self._risk_engine = risk_engine
        self._allocator = allocator
        self._journal = AgentJournal("overseer")
        self._loop = AgentLoop(
            "overseer",
            llm=llm,
            regime_fn=regime_fn,
            max_tokens=8192,
            temperature=0.2,
            memory_window_hours=7 * 24,
            memory_limit=50,
        )

    @property
    def journal(self) -> AgentJournal:
        return self._journal

    # ─── Daily aggregation + circuit breakers ────────────────────────────────
    def run_daily_aggregate(
        self,
        *, window_days: int = 28,
        environment: str = "paper",
    ) -> OverseerReport:
        """20:00 nightly run — deterministic (no LLM)."""
        metrics = self._gather_metrics(window_days, environment)
        halts = self._apply_circuit_breakers(metrics)
        decoupled = self._scan_decoupled_fills(environment=environment)

        body = self._fmt_daily_body(metrics, halts, decoupled)
        structured = {
            "metrics": {
                aid: _metrics_to_dict(m) for aid, m in metrics.items()
            },
            "halts_issued": halts,
            "decoupled_fills": decoupled,
            "window_days": window_days,
        }
        jid = self._journal.record_decision(
            title=f"daily aggregate {datetime.now(timezone.utc).date().isoformat()}",
            body=body,
            structured=structured,
            tags=["overseer", "daily"],
        )
        return OverseerReport(
            ts=datetime.now(timezone.utc),
            metrics=metrics,
            halts_issued=halts,
            decoupled_fills=decoupled,
            journal_id=jid,
        )

    # ─── Weekly LLM-driven review + reallocation ─────────────────────────────
    def run_weekly_review(
        self,
        *, window_days: int = 28,
        environment: str = "paper",
    ) -> WeeklyReview:
        """Sunday 11:00 — LLM writes the post-mortem memo and may propose a
        reweighting. Proposed reweighting is validated against
        `OverseerStrategyAllocator.reallocate` bounds."""
        metrics = self._gather_metrics(window_days, environment)
        halted = self._risk_engine.halted_agents()

        payload = {
            "window_days": window_days,
            "current_weights": self._allocator.effective_weights(),
            "base_weights": self._allocator.base_weights(),
            "halted_agents": halted,
            "metrics": {aid: _metrics_to_dict(m) for aid, m in metrics.items()},
        }

        user_msg = (
            "Weekly portfolio review. Based on the metrics below, write a "
            "short post-mortem (2-3 paragraphs) covering what worked, what "
            "broke, and any research-trader coupling issues. Then propose "
            "next week's allocation weights — they must sum to 1.0 and no "
            "single weight can shift more than 10pp from its base. Return "
            "the proposal as JSON in a fenced code block like:\n"
            "```json\n{\"day\": 0.3, \"swing\": 0.4, \"long_term\": 0.3, "
            "\"rationale\": \"...\"}\n```\n"
            "If you would not change weights, return the current weights "
            "unchanged and say so in the rationale."
        )

        run = self._loop.run(
            user_message=user_msg,
            extra_context=payload,
            persist_output_as="memo",
            persist_title=f"weekly review {datetime.now(timezone.utc).date().isoformat()}",
        )

        proposal = _extract_weight_proposal(run.text)
        update: Optional[AllocationUpdate] = None
        if proposal is not None:
            rationale = proposal.pop("rationale", "LLM weekly review")
            update = self._allocator.reallocate(proposal, reason=rationale)
            # Persist the allocator decision so the monitoring API can surface it.
            self._journal.record_decision(
                title="weekly reallocation",
                body=(
                    f"prior={update.prior_weights} → new={update.new_weights} "
                    f"rationale={rationale} rejected={update.rejected_reason}"
                ),
                structured={
                    "prior": update.prior_weights,
                    "new": update.new_weights,
                    "reason": update.reason,
                    "rejected_reason": update.rejected_reason,
                },
                tags=["overseer", "reallocation"],
            )
        return WeeklyReview(
            ts=datetime.now(timezone.utc),
            allocation_update=update,
            agent_run=run,
        )

    # ─── Real-time monitor (5-min cadence during market hours) ───────────────
    def run_realtime_monitor(self, *, environment: str = "paper") -> List[Dict[str, Any]]:
        """Lightweight deterministic check — halt agents with new decouplings
        or fresh mandate breaches since the last check. Returns list of events
        for the monitoring API."""
        events: List[Dict[str, Any]] = []

        # 1. Coupling drift — fills in last 15 minutes without paired research.
        horizon = datetime.now(timezone.utc) - timedelta(minutes=15)
        decoupled = self._scan_decoupled_fills(environment=environment, since=horizon)
        for row in decoupled:
            self._risk_engine.halt_agent(
                row["agent_id"],
                reason=f"research_trader_disconnect:{row['symbol']}",
            )
            self._journal.record_alert(
                title="research_trader_disconnect",
                body=(
                    f"{row['agent_id']} filled {row['symbol']} at {row['entry_ts']} "
                    f"without paired research brief. Agent halted."
                ),
                symbol=row["symbol"],
                structured=row,
            )
            events.append({"type": "halt", "agent_id": row["agent_id"], "reason": "decoupled"})

        # 2. Kill-switch cascade: if the global kill switch is active, halt all
        # agents for visibility (they're already blocked at pre-trade).
        if self._risk_engine.kill_switch_active:
            for aid in TRADER_AGENTS:
                if not self._risk_engine.is_agent_halted(aid):
                    self._risk_engine.halt_agent(aid, reason="global_kill_switch")
                    events.append({"type": "halt", "agent_id": aid, "reason": "global_kill_switch"})
        return events

    # ─── Internals ───────────────────────────────────────────────────────────
    def _gather_metrics(
        self, window_days: int, environment: str,
    ) -> Dict[str, AgentMetrics | ResearchMetrics]:
        out: Dict[str, AgentMetrics | ResearchMetrics] = {}
        with get_session_factory()() as s:
            for aid in KNOWN_AGENTS:
                if aid in TRADER_AGENTS:
                    out[aid] = compute_trader_metrics(
                        aid, window_days, environment, session=s,
                    )
                elif aid.endswith("_research"):
                    out[aid] = compute_research_metrics(
                        aid, window_days, session=s,
                    )
        return out

    def _apply_circuit_breakers(
        self, metrics: Dict[str, AgentMetrics | ResearchMetrics],
    ) -> List[tuple[str, str]]:
        halts: List[tuple[str, str]] = []
        for aid in TRADER_AGENTS:
            m = metrics.get(aid)
            if not isinstance(m, AgentMetrics):
                continue
            if m.max_drawdown_pct >= MAX_DD_HALT_PCT:
                reason = f"max_dd_{m.max_drawdown_pct:.2%}"
                self._risk_engine.halt_agent(aid, reason=reason)
                halts.append((aid, reason))
            elif m.mandate_violations >= MAX_MANDATE_VIOLATIONS_HALT:
                reason = f"mandate_violations_{m.mandate_violations}"
                self._risk_engine.halt_agent(aid, reason=reason)
                halts.append((aid, reason))
        return halts

    def _scan_decoupled_fills(
        self, *, environment: str, since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """List trader fills with no paired research brief in the preceding
        `RESEARCH_COUPLING_WINDOW_HOURS`. Used by daily aggregate + real-time."""
        since = since or (datetime.now(timezone.utc) - timedelta(days=1))
        out: List[Dict[str, Any]] = []
        with get_session_factory()() as s:
            from sqlalchemy import select
            fills = list(
                s.execute(
                    select(Trade.symbol, Trade.entry_ts, Trade.strategy_id)
                    .where(Trade.environment == environment)
                    .where(Trade.entry_ts >= since)
                    .where(Trade.strategy_id.in_(TRADER_AGENTS))
                ).all()
            )
            for symbol, entry_ts, strategy_id in fills:
                if strategy_id not in PAIRED_RESEARCH:
                    continue
                match = audit_research_coupling(
                    strategy_id, symbol, entry_ts,
                    window_hours=RESEARCH_COUPLING_WINDOW_HOURS,
                    session=s,
                )
                if match is None:
                    out.append({
                        "agent_id": strategy_id,
                        "symbol": symbol,
                        "entry_ts": entry_ts.isoformat(),
                    })
        return out

    @staticmethod
    def _fmt_daily_body(
        metrics: Dict[str, AgentMetrics | ResearchMetrics],
        halts: List[tuple[str, str]],
        decoupled: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = ["### Per-agent metrics"]
        for aid in TRADER_AGENTS:
            m = metrics.get(aid)
            if isinstance(m, AgentMetrics):
                lines.append(
                    f"- {aid}: n={m.n_trades} hit={m.hit_rate:.2%} "
                    f"expectancy=${m.expectancy_per_trade:.2f} "
                    f"sharpe={m.sharpe_annualized:.2f} "
                    f"max_dd={m.max_drawdown_pct:.2%} "
                    f"mandate_violations={m.mandate_violations}"
                )
        for aid in ("day_research", "swing_research", "long_term_research"):
            r = metrics.get(aid)
            if isinstance(r, ResearchMetrics):
                lines.append(
                    f"- {aid}: briefs={r.n_briefs} memos={r.n_memos} "
                    f"alerts={r.n_alerts} decoupled_fills={r.trader_decoupled_trades}"
                )
        if halts:
            lines.append("### Halts issued")
            for a, reason in halts:
                lines.append(f"- {a}: {reason}")
        if decoupled:
            lines.append("### Decoupled fills (last day)")
            for row in decoupled:
                lines.append(
                    f"- {row['agent_id']} {row['symbol']} @ {row['entry_ts']}"
                )
        return "\n".join(lines)


def _metrics_to_dict(m: AgentMetrics | ResearchMetrics) -> Dict[str, Any]:
    if isinstance(m, AgentMetrics):
        return {
            "kind": "trader",
            "n_trades": m.n_trades,
            "hit_rate": m.hit_rate,
            "sharpe": m.sharpe_annualized,
            "max_dd": m.max_drawdown_pct,
            "expectancy": m.expectancy_per_trade,
            "net_pnl": m.net_pnl,
            "mandate_violations": m.mandate_violations,
        }
    return {
        "kind": "research",
        "n_briefs": m.n_briefs,
        "n_memos": m.n_memos,
        "n_alerts": m.n_alerts,
        "n_reviews": m.n_reviews,
        "avg_confidence": m.avg_confidence,
        "decoupled_fills": m.trader_decoupled_trades,
    }


def _extract_weight_proposal(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first ```json {...} ``` block containing weights from LLM text."""
    import json
    import re

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    # Keep only known strategies + rationale.
    allowed = set(TRADER_AGENTS) | {"rationale"}
    filtered = {k: v for k, v in parsed.items() if k in allowed}
    if not any(k in TRADER_AGENTS for k in filtered):
        return None
    return filtered
