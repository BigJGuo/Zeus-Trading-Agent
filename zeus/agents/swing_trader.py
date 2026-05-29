"""Swing Trader Agent (A2): 2-10 day holds on h=5d horizon."""
from __future__ import annotations

from zeus.agents.trader_base import TraderAgentBase, TraderContext


class SwingTraderAgent(TraderAgentBase):
    """Consumes swing_research briefs + cross_horizon_swing_h5 predictions."""

    agent_id = "swing"
    max_tokens = 4096
    temperature = 0.1

    def _user_message(self, ctx: TraderContext) -> str:
        return (
            "Make tonight's swing decisions for the next session. Horizon "
            "is 2-10 days, max hold 15 days. Pull the latest EOD / weekly "
            "swing_research briefs via query_journal and cross-check them "
            "against the model predictions below. Call `propose_trade` "
            "once per final decision — each entry MUST include a "
            "`brief_id` referencing the paired research. Exits should "
            "cite either a broken level from the brief or a fired "
            "invalidation trigger. After all proposals, give a 2-sentence "
            "summary of portfolio posture for the week."
        )
