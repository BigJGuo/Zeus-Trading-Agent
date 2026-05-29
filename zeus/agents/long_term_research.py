"""Long-Term Research Agent (A6): deep-dive memos + monthly reviews."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import structlog

from zeus.agents.base import AgentLoop, AgentRunResult
from zeus.llm.client import AnthropicClient

log = structlog.get_logger(__name__)

AGENT_ID = "long_term_research"


@dataclass
class DeepDiveContext:
    symbol: str
    regime: str
    focus: Optional[str] = None            # e.g. "upcoming earnings", "thesis refresh"


@dataclass
class ReviewContext:
    open_positions: List[Dict[str, Any]]   # {symbol, entry_ts, memo_id, ...}
    regime: str


class LongTermResearchAgent:
    """Publishes investment memos + monthly reviews."""

    def __init__(
        self,
        *,
        llm: AnthropicClient,
        broker: Optional[Any] = None,
        regime_fn: Optional[Callable[[], str]] = None,
    ):
        # Opus-tier for deep-dives — cost envelope budgeted at $2/day.
        self._loop = AgentLoop(
            AGENT_ID,
            llm=llm,
            broker=broker,
            regime_fn=regime_fn,
            max_tokens=8192,
            temperature=0.3,
            memory_window_hours=30 * 24,   # 30 days for thesis continuity
            memory_limit=30,
        )

    def run_deep_dive(self, ctx: DeepDiveContext) -> AgentRunResult:
        """New memo on a single name. Cadence: one every 3–10 days."""
        return self._loop.run(
            user_message=(
                f"Deep-dive on {ctx.symbol}. Produce a memo with thesis "
                "bullets (3–5 load-bearing points), bull/base/bear targets, "
                "holding_window_days (20–120+), risks, invalidation triggers, "
                "sources, confidence_1to5. Pull fundamentals, the 1Y chart, "
                "and any news in the last week."
            ),
            extra_context={
                "symbol": ctx.symbol,
                "regime": ctx.regime,
                "focus": ctx.focus,
            },
            persist_output_as="memo",
            persist_title=f"deep dive {ctx.symbol}",
            persist_symbol=ctx.symbol,
        )

    def run_monthly_review(self, ctx: ReviewContext) -> List[AgentRunResult]:
        """One `review` entry per current long-term holding."""
        results: List[AgentRunResult] = []
        for pos in ctx.open_positions:
            symbol = pos.get("symbol")
            if not symbol:
                continue
            r = self._loop.run(
                user_message=(
                    f"Monthly review of long-term holding {symbol}. Re-read "
                    "your prior memo on this name (via query_journal). Is "
                    "the thesis intact? Have any invalidation triggers fired? "
                    "Output: thesis_intact (yes/no/partial), targets_refresh, "
                    "suggested_action (hold | trim | exit), rationale."
                ),
                extra_context={"position": pos, "regime": ctx.regime},
                persist_output_as="review",
                persist_title=f"monthly review {symbol}",
                persist_symbol=symbol,
            )
            results.append(r)
        return results
