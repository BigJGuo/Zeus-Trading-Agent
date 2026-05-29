"""Long-Term Trader Agent (A3): 20d-6mo holds on h=20d horizon."""
from __future__ import annotations

from zeus.agents.trader_base import TraderAgentBase, TraderContext


class LongTermTraderAgent(TraderAgentBase):
    """Consumes long_term_research memos + cross_horizon_long_term_h20 predictions."""

    agent_id = "long_term"
    max_tokens = 4096
    temperature = 0.1

    def _user_message(self, ctx: TraderContext) -> str:
        return (
            "Decide long-term positioning. Horizon is 20-120+ days, max "
            "hold 180 days. Pull the relevant investment memos + monthly "
            "reviews via query_journal, re-weigh the thesis against the "
            "model predictions below, and call `propose_trade` once per "
            "final decision. Every entry MUST include a `memo_id` "
            "referencing the supporting memo. For exits, the rationale "
            "must quote either an invalidation trigger from the memo or a "
            "review that flagged the thesis as broken. After all "
            "proposals, give a 2-3 sentence summary of portfolio stance."
        )
