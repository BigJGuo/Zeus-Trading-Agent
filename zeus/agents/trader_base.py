"""Shared base for the three trader agents.

Each trader reads model predictions + the paired research journal, runs an
AgentLoop with its role-specific prompt, and returns a structured
`TraderDecision` that the scheduler can hand to StrategyManager.

Proposal -> execution path:
  1. AgentLoop collects `propose_trade(...)` calls from the LLM.
  2. Each proposal already wrote a `trade_rationale` journal row (via
     AgentLoop._persist_output + its proposals loop).
  3. This base maps proposals to SessionPlan-style dicts that
     StrategyManager.execute_entries_all can consume once attached to the
     strategy's current_plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import structlog

from zeus.agents.base import AgentLoop, AgentRunResult
from zeus.live.trading_loop import SessionPlan
from zeus.llm.client import AnthropicClient

log = structlog.get_logger(__name__)


@dataclass
class TraderContext:
    """Input payload fed to a trader's .decide() call."""
    strategy_id: str
    model_version: str
    feature_version: str
    regime: str
    # Per-symbol predictions from the horizon-specific gated predictor. Each
    # item is something like {symbol, expected_return, up_probability, score}.
    predictions: List[Dict[str, Any]]
    current_positions: List[Dict[str, Any]] = field(default_factory=list)
    plan_date: Optional[date] = None


@dataclass
class TraderDecision:
    strategy_id: str
    plan: SessionPlan                       # drop-in for StrategyManager
    proposals: List[Dict[str, Any]]         # raw LLM proposals (for audit)
    summary_text: str
    agent_run: AgentRunResult


class TraderAgentBase:
    """Concrete traders subclass with agent_id + default max_tokens set."""

    agent_id: str = "day"                  # override per trader
    max_tokens: int = 4096
    temperature: float = 0.1

    def __init__(
        self,
        *,
        llm: AnthropicClient,
        broker: Optional[Any] = None,
        regime_fn: Optional[Callable[[], str]] = None,
    ):
        self._loop = AgentLoop(
            self.agent_id,
            llm=llm,
            prompt_role=f"{self.agent_id}_trader",
            broker=broker,
            regime_fn=regime_fn,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            memory_window_hours=self._memory_window_hours(),
            memory_limit=30,
        )

    def _memory_window_hours(self) -> int:
        return {"day": 12, "swing": 72, "long_term": 24 * 14}.get(self.agent_id, 24)

    # ─── Main entry ───────────────────────────────────────────────────────────
    def decide(self, ctx: TraderContext) -> TraderDecision:
        """Run the LLM loop, convert proposals into a SessionPlan."""
        result = self._loop.run(
            user_message=self._user_message(ctx),
            extra_context={
                "strategy_id": ctx.strategy_id,
                "regime": ctx.regime,
                "model_version": ctx.model_version,
                "predictions": ctx.predictions,
                "current_positions": ctx.current_positions,
            },
            persist_output_as="decision",
            persist_title=f"{self.agent_id} decision session",
        )

        entries, exits = self._split_proposals(result.proposals)
        plan = SessionPlan(
            plan_date=ctx.plan_date or date.today(),
            regime=ctx.regime,
            entries=entries,
            exits=exits,
            holds=[],
            model_version=ctx.model_version,
            feature_version=ctx.feature_version,
        )
        log.info(
            "trader_decision",
            agent_id=self.agent_id, n_entries=len(entries), n_exits=len(exits),
            cost_usd=round(result.cost_usd, 4),
        )

        # Surface zero-entries sessions to the dashboard's Risk panel so
        # future trade-droughts are visible on day 1 instead of day N.
        # Excerpts the agent's own summary text so we know WHY nothing
        # was proposed (gate failure, no data, low conviction, etc.).
        if len(entries) == 0:
            self._record_zero_entries_event(
                strategy_id=ctx.strategy_id,
                summary_text=result.text or "",
                n_proposals_total=len(result.proposals),
                n_exits=len(exits),
            )

        return TraderDecision(
            strategy_id=ctx.strategy_id,
            plan=plan,
            proposals=list(result.proposals),
            summary_text=result.text,
            agent_run=result,
        )

    def _record_zero_entries_event(
        self,
        *,
        strategy_id: str,
        summary_text: str,
        n_proposals_total: int,
        n_exits: int,
    ) -> None:
        """Write a `risk_events` row when this trader returned 0 entries.

        Severity = INFO. Dedupes by (strategy_id, event_type) within a
        12-hour window so a strategy that runs both an overnight_plan and
        a morning_plan on the same day only logs once."""
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import func, select
        try:
            from zeus.data.storage.database import RiskEvent, get_session_factory
        except Exception:
            return  # never let telemetry block decision flow

        # Trim the summary to a useful excerpt — last paragraph if available.
        text = (summary_text or "").strip()
        if not text:
            excerpt = "(agent produced no summary text)"
        else:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            excerpt = paragraphs[-1] if paragraphs else text
            if len(excerpt) > 500:
                excerpt = excerpt[:497] + "..."

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=12)
        try:
            SF = get_session_factory()
            with SF() as s:
                recent = s.execute(
                    select(func.count(RiskEvent.id))
                    .where(RiskEvent.event_type == "zero_entries")
                    .where(RiskEvent.description.like(f"[{strategy_id}]%"))
                    .where(RiskEvent.ts >= cutoff)
                ).scalar()
                if recent and recent > 0:
                    return
                row = RiskEvent(
                    ts=now,
                    event_type="zero_entries",
                    severity="INFO",
                    description=(
                        f"[{strategy_id}] proposed 0 entries "
                        f"(n_proposals={n_proposals_total}, n_exits={n_exits}). "
                        f"Reason: {excerpt}"
                    ),
                    action_taken="surface_to_dashboard",
                    environment="paper",
                )
                s.add(row)
                s.commit()
        except Exception as e:
            log.warning(
                "zero_entries_event_write_failed",
                strategy_id=strategy_id, error=str(e),
            )

    # ─── Hooks subclasses may override ────────────────────────────────────────
    def _user_message(self, ctx: TraderContext) -> str:
        return (
            "Make today's trading decisions. Read your paired research "
            "briefs, cross-check the model predictions below, and call "
            "`propose_trade` once per final decision. For each entry, "
            "include a `brief_id` (or `memo_id` for long-term) referencing "
            "the supporting research. After all proposals, give a short "
            "summary."
        )

    # Max allowed drift between proposed target_price and latest close.
    # Above this, we treat the LLM's price as hallucinated (e.g. wrong
    # ticker, stale split-unadjusted value) and drop the proposal.
    TARGET_PRICE_MAX_DEVIATION = 0.30

    def _latest_close(self, symbol: str) -> Optional[float]:
        """Best-effort lookup of a symbol's most recent trade price.

        Returns None if the broker isn't attached or the lookup fails —
        in that case the caller skips the ±30% deviation guard rather
        than dropping otherwise-valid proposals on a transient data gap.
        """
        broker = getattr(self._loop, "_broker", None)
        if broker is None:
            return None
        try:
            price = broker.get_latest_price(symbol)
            return float(price) if price else None
        except Exception as e:
            log.warning(
                "trader_latest_close_failed",
                agent_id=self.agent_id, symbol=symbol, error=str(e),
            )
            return None

    def _split_proposals(
        self, proposals: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        entries: List[Dict[str, Any]] = []
        exits: List[str] = []
        for p in proposals:
            action = p.get("action")
            symbol = p.get("symbol")
            if not symbol:
                continue
            if action == "enter":
                target_price = float(p.get("target_price") or 0.0)
                notional_raw = p.get("notional_usd")
                if notional_raw is None:
                    notional_raw = p.get("notional")
                notional = float(notional_raw or 0.0)
                shares = int(p.get("shares") or 0)

                # Reject missing notional explicitly — without it the
                # schema can't derive a size, and the resulting proposal
                # silently drops. Surface it with its own reason.
                if shares <= 0 and notional <= 0:
                    log.warning(
                        "trader_proposal_dropped",
                        agent_id=self.agent_id, symbol=symbol,
                        reason="missing_notional_usd",
                        target_price=target_price,
                    )
                    continue

                # Price anchoring guard: reject target_price that drifts
                # more than ±30% from the latest close. Catches common
                # hallucinations (wrong ticker, split-unadjusted prices,
                # multi-bagger thesis numbers).
                last_close = self._latest_close(symbol)
                if last_close and last_close > 0 and target_price > 0:
                    deviation = abs(target_price - last_close) / last_close
                    if deviation > self.TARGET_PRICE_MAX_DEVIATION:
                        log.warning(
                            "trader_proposal_dropped",
                            agent_id=self.agent_id, symbol=symbol,
                            reason="target_price_deviation_exceeds_30pct",
                            target_price=target_price,
                            last_close=last_close,
                            deviation=round(deviation, 3),
                        )
                        continue

                if shares <= 0 and notional > 0 and target_price > 0:
                    shares = int(notional // target_price)
                if shares <= 0 or target_price <= 0:
                    log.warning(
                        "trader_proposal_dropped",
                        agent_id=self.agent_id, symbol=symbol,
                        reason="shares_or_price_nonpositive",
                        shares=shares, target_price=target_price,
                        notional_usd=p.get("notional_usd"),
                    )
                    continue
                entries.append({
                    "symbol": symbol,
                    "shares": shares,
                    "target_price": target_price,
                    "sector": None,
                    "adv_usd": None,
                    "stop": float(p.get("stop_price") or 0.0),
                    "signal_score": 0.0,
                    "strategy_id": self.agent_id,
                    "rationale": p.get("rationale", ""),
                    "brief_id": p.get("brief_id"),
                    "memo_id": p.get("memo_id"),
                })
            elif action == "exit":
                exits.append(symbol)
        return entries, exits
