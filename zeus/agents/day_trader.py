"""Day Trader Agent (A1): intraday entries/exits on h=3d horizon."""
from __future__ import annotations

from zeus.agents.trader_base import TraderAgentBase, TraderContext


class DayTraderAgent(TraderAgentBase):
    """Consumes day_research briefs + cross_horizon_day_h3 predictions."""

    agent_id = "day"
    max_tokens = 4096
    temperature = 0.1

    def _user_message(self, ctx: TraderContext) -> str:
        return (
            "Make today's day-trading decisions. Horizon is 1-3 days, max "
            "hold 5 days. Read the most recent day_research briefs via "
            "query_journal, cross-check against the model predictions "
            "below, and call `propose_trade` once per final decision. "
            "Every entry MUST include a `brief_id` referencing the "
            "supporting day_research brief. Size risk to 1R per name with "
            "a defined stop. After all proposals, give a 2-sentence "
            "summary of the session's thesis."
        )
