"""Swing Research Agent (A5): EOD + weekly setup briefs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import structlog

from zeus.agents.base import AgentLoop, AgentRunResult
from zeus.llm.client import AnthropicClient

log = structlog.get_logger(__name__)

AGENT_ID = "swing_research"


@dataclass
class SwingResearchContext:
    candidates: List[Dict[str, Any]]   # {symbol, close, 20d_change, setup_hint, ...}
    regime: str
    earnings_this_week: Optional[List[Dict[str, Any]]] = None


class SwingResearchAgent:
    """Produces 1-page setup briefs the swing trader consumes nightly."""

    def __init__(
        self,
        *,
        llm: AnthropicClient,
        broker: Optional[Any] = None,
        regime_fn: Optional[Callable[[], str]] = None,
    ):
        self._loop = AgentLoop(
            AGENT_ID,
            llm=llm,
            broker=broker,
            regime_fn=regime_fn,
            max_tokens=6144,
            temperature=0.3,
            memory_window_hours=72,
            memory_limit=40,
        )

    def run_eod(self, ctx: SwingResearchContext) -> AgentRunResult:
        """19:00 EOD: re-score universe, publish tomorrow's setup briefs."""
        return self._loop.run(
            user_message=(
                "End-of-day swing setup scan. For each candidate below "
                "that qualifies as breakout / pullback / reversal / "
                "catalyst, produce a brief with levels (entry, stop, t1, "
                "t2), trigger, holding_window_days (2–10), thesis, "
                "confidence_1to5."
            ),
            extra_context={
                "candidates": ctx.candidates,
                "regime": ctx.regime,
                "earnings_this_week": ctx.earnings_this_week or [],
            },
            persist_output_as="brief",
            persist_title="EOD swing setups",
            persist_symbol="MARKET",
        )

    def run_weekly_watchlist(
        self, *, universe: List[str], regime: str,
    ) -> AgentRunResult:
        """Sunday 10:00: refresh the rolling 2–3 week watchlist."""
        return self._loop.run(
            user_message=(
                "Weekly watchlist refresh. Select the 15–25 most promising "
                "names for the next 2–3 weeks. Justify each selection with "
                "a 1-sentence thesis."
            ),
            extra_context={"universe": universe, "regime": regime},
            persist_output_as="memo",
            persist_title="weekly swing watchlist",
            persist_symbol="MARKET",
        )

    def run_event_driven(
        self, *, symbol: str, event_summary: str, regime: str,
    ) -> AgentRunResult:
        """Event-driven (earnings / upgrade / news): targeted brief ~30m."""
        return self._loop.run(
            user_message=(
                f"Event-driven update on {symbol}. Event: {event_summary}. "
                "If it changes the setup, emit a fresh brief; else emit an "
                "alert so the trader knows to skip."
            ),
            extra_context={"symbol": symbol, "regime": regime},
            persist_output_as="brief",
            persist_title=f"event-driven {symbol}",
            persist_symbol=symbol,
        )
