"""Day Research Agent (A4): intraday watchlist + regime alerts.

Paired 1:1 with the day trader. Emits `brief` entries (Top-10 intraday
candidates) and `alert` entries (regime shifts) into the journal. Invoked
on schedule by APScheduler with the cadences spec'd in the prompt:
pre-market, intraday 5-min refresh, post-market wrap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import structlog

from zeus.agents.base import AgentLoop, AgentRunResult
from zeus.llm.client import AnthropicClient

log = structlog.get_logger(__name__)

AGENT_ID = "day_research"


@dataclass
class DayResearchContext:
    """Per-run inputs. `candidates` is the pre-filtered watchlist the LLM
    should produce briefs for — built upstream by a gap scanner."""
    candidates: List[Dict[str, Any]]   # {symbol, gap_pct, rvol, ...}
    regime: str
    news_highlights: Optional[List[Dict[str, Any]]] = None


class DayResearchAgent:
    """Orchestrates intraday research loop — pre-market / refresh / wrap."""

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
            max_tokens=4096,
            temperature=0.3,
            memory_window_hours=12,
            memory_limit=30,
        )

    # ─── Lifecycle entry points (scheduled) ───────────────────────────────────
    def run_premarket_scan(self, ctx: DayResearchContext) -> AgentRunResult:
        """05:00 / 07:00 / 09:00: pre-market gappers + overnight catalysts."""
        log.info("day_research_premarket", n_candidates=len(ctx.candidates),
                 regime=ctx.regime)
        return self._loop.run(
            user_message=(
                "Pre-market scan. For each candidate below, produce a "
                "compact brief (ticker, catalyst, VWAP/support/resistance, "
                "risk, confidence_1to5). Rank them; keep top 10 only."
            ),
            extra_context={
                "candidates": ctx.candidates,
                "regime": ctx.regime,
                "news_highlights": ctx.news_highlights or [],
            },
            persist_output_as="brief",
            persist_title=f"premkt scan {ctx.regime}",
            persist_symbol="MARKET",
        )

    def run_intraday_refresh(self, ctx: DayResearchContext) -> AgentRunResult:
        """Every 5 min during 09:30–16:00: update briefs on active names."""
        return self._loop.run(
            user_message=(
                "Intraday refresh. For active watchlist names below, update "
                "price-action notes + emit an ALERT if the regime is shifting "
                "or a brief's setup has invalidated."
            ),
            extra_context={
                "candidates": ctx.candidates,
                "regime": ctx.regime,
            },
            persist_output_as="brief",
            persist_title="intraday refresh",
            persist_symbol="MARKET",
        )

    def run_postmarket_wrap(
        self, *, realized_trades: List[Dict[str, Any]], regime: str,
    ) -> AgentRunResult:
        """16:15: wrap-up — what worked, lessons for tomorrow."""
        return self._loop.run(
            user_message=(
                "Post-market wrap. Review today's realized trades below, "
                "compare to the briefs you emitted this morning, and write "
                "one lesson (what to change tomorrow)."
            ),
            extra_context={
                "realized_trades": realized_trades,
                "regime": regime,
            },
            persist_output_as="lesson",
            persist_title="day research post-mkt wrap",
        )
