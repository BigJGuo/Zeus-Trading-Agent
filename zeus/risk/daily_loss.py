"""Intraday daily-loss circuit breaker (separate from drawdown-from-peak guard)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from zeus.risk.limits import RiskLimits


class DailyLossLevel(str, Enum):
    OK = "ok"
    WARN = "warn"
    STOP_NEW = "stop_new"
    KILL = "kill"


@dataclass
class DailyLossState:
    level: DailyLossLevel
    daily_pnl_usd: float
    stop_new_entries: bool
    close_all: bool
    message: str


class DailyLossMonitor:
    def __init__(self, limits: RiskLimits):
        self._limits = limits

    def check(self, daily_pnl_usd: float) -> DailyLossState:
        pnl = daily_pnl_usd  # already signed; losses are negative

        if pnl <= -self._limits.daily_loss_kill_usd:
            return DailyLossState(
                DailyLossLevel.KILL, pnl, True, True,
                f"DAILY KILL: intraday loss ${pnl:,.0f}. Closing all positions.",
            )
        if pnl <= -self._limits.daily_loss_stop_new_usd:
            return DailyLossState(
                DailyLossLevel.STOP_NEW, pnl, True, False,
                f"Daily loss limit L2: ${pnl:,.0f}. No new entries.",
            )
        if pnl <= -self._limits.daily_loss_warn_usd:
            return DailyLossState(
                DailyLossLevel.WARN, pnl, False, False,
                f"Daily loss warning: ${pnl:,.0f}.",
            )
        return DailyLossState(DailyLossLevel.OK, pnl, False, False, "")
