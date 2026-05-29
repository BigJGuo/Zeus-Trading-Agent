"""Hard risk limits — configurable pre-trade checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20
    max_sector_pct: float = 0.35
    max_single_order_notional: float = 25_000.0
    min_liquidity_adv_multiple: float = 0.05
    max_portfolio_gross_exposure: float = 1.0
    broker_buying_power_buffer: float = 0.05
    min_position_notional: float = 2_000.0

    # Percentage thresholds (fraction of peak NAV). DrawdownGuard classifies
    # on these — they scale naturally with the account so a doubled NAV
    # doesn't get permanently locked out by a normal 10% pullback.
    drawdown_l1_pct: float = 0.10
    drawdown_l2_pct: float = 0.20
    drawdown_l3_pct: float = 0.35
    drawdown_l4_pct: float = 0.50

    # Legacy absolute-dollar thresholds. Kept for any callers that still
    # read them; DrawdownGuard no longer classifies on these.
    drawdown_l1_usd: float = 10_000.0
    drawdown_l2_usd: float = 20_000.0
    drawdown_l3_usd: float = 35_000.0
    drawdown_l4_usd: float = 50_000.0

    daily_loss_warn_usd: float = 3_000.0
    daily_loss_stop_new_usd: float = 5_000.0
    daily_loss_kill_usd: float = 7_500.0

    @classmethod
    def from_config(cls, cfg: Dict) -> "RiskLimits":
        trading = cfg.get("trading", {})
        risk = cfg.get("risk", {})
        return cls(
            max_position_pct=trading.get("max_position_pct", 0.20),
            max_sector_pct=trading.get("max_sector_pct", 0.35),
            min_position_notional=trading.get("min_position_notional", 2_000.0),
            drawdown_l1_pct=risk.get("drawdown_l1_pct", 0.10),
            drawdown_l2_pct=risk.get("drawdown_l2_pct", 0.20),
            drawdown_l3_pct=risk.get("drawdown_l3_pct", 0.35),
            drawdown_l4_pct=risk.get("drawdown_l4_pct", 0.50),
            drawdown_l1_usd=risk.get("drawdown_l1_usd", 10_000.0),
            drawdown_l2_usd=risk.get("drawdown_l2_usd", 20_000.0),
            drawdown_l3_usd=risk.get("drawdown_l3_usd", 35_000.0),
            drawdown_l4_usd=risk.get("drawdown_l4_usd", 50_000.0),
            daily_loss_warn_usd=risk.get("daily_loss_limit_usd", 3_000.0),
            daily_loss_stop_new_usd=risk.get("daily_loss_stop_new_entries", 5_000.0),
            daily_loss_kill_usd=risk.get("daily_loss_kill_usd", 7_500.0),
        )
