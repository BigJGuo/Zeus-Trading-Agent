"""AgentRuntime — one place to construct all 7 agents + deps.

Separating this from the scheduler makes it testable: a test can pass a
stub AnthropicClient and still get a fully-wired bundle of agents to drive
through the cadence methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import structlog

from zeus.agents.day_research import DayResearchAgent
from zeus.agents.day_trader import DayTraderAgent
from zeus.agents.long_term_research import LongTermResearchAgent
from zeus.agents.long_term_trader import LongTermTraderAgent
from zeus.agents.overseer import OverseerAgent
from zeus.agents.swing_research import SwingResearchAgent
from zeus.agents.swing_trader import SwingTraderAgent
from zeus.llm.client import AnthropicClient
from zeus.risk.engine import RiskEngine
from zeus.risk.strategy_allocator import OverseerStrategyAllocator

log = structlog.get_logger(__name__)


@dataclass
class AgentBundle:
    """All 7 agents wired together. Passed to the scheduler jobs."""
    # Research
    day_research: DayResearchAgent
    swing_research: SwingResearchAgent
    long_term_research: LongTermResearchAgent
    # Traders
    day_trader: DayTraderAgent
    swing_trader: SwingTraderAgent
    long_term_trader: LongTermTraderAgent
    # Overseer
    overseer: OverseerAgent

    def all_traders(self):
        return (self.day_trader, self.swing_trader, self.long_term_trader)

    def all_research(self):
        return (self.day_research, self.swing_research, self.long_term_research)


def build_agent_bundle(
    *,
    broker: Optional[Any],
    risk_engine: RiskEngine,
    allocator: OverseerStrategyAllocator,
    opus_llm: AnthropicClient,
    sonnet_llm: AnthropicClient,
    haiku_llm: AnthropicClient,
    regime_fn: Optional[Callable[[], str]] = None,
) -> AgentBundle:
    """Wire every agent with its correct LLM tier + shared broker/regime.

    Tiering (per the plan's cost envelope):
      - day_research          → haiku  (high volume pre-market)
      - swing_research        → sonnet
      - long_term_research    → opus   (deep-dives)
      - day/swing trader      → sonnet
      - long_term_trader      → opus
      - overseer              → opus
    """
    return AgentBundle(
        day_research=DayResearchAgent(
            llm=haiku_llm, broker=broker, regime_fn=regime_fn,
        ),
        swing_research=SwingResearchAgent(
            llm=sonnet_llm, broker=broker, regime_fn=regime_fn,
        ),
        long_term_research=LongTermResearchAgent(
            llm=opus_llm, broker=broker, regime_fn=regime_fn,
        ),
        day_trader=DayTraderAgent(
            llm=sonnet_llm, broker=broker, regime_fn=regime_fn,
        ),
        swing_trader=SwingTraderAgent(
            llm=sonnet_llm, broker=broker, regime_fn=regime_fn,
        ),
        long_term_trader=LongTermTraderAgent(
            llm=opus_llm, broker=broker, regime_fn=regime_fn,
        ),
        overseer=OverseerAgent(
            llm=opus_llm,
            risk_engine=risk_engine,
            allocator=allocator,
            regime_fn=regime_fn,
        ),
    )
