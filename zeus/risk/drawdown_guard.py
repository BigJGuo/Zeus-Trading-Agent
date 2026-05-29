"""Drawdown tracking and escalation protocol — 4 levels → kill switch."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog

from zeus.risk.limits import RiskLimits

log = structlog.get_logger(__name__)


class DrawdownLevel(int, Enum):
    NORMAL = 0
    WARNING = 1      # -$10k — reduce exposure
    DEGRADED = 2     # -$20k — fewer positions, smaller size
    PRESERVATION = 3  # -$35k — no new entries
    KILL = 4         # -$50k — close everything


@dataclass
class DrawdownState:
    level: DrawdownLevel
    drawdown_usd: float
    drawdown_pct: float
    peak_value: float
    current_value: float
    max_new_positions: int
    max_new_position_pct: float
    stop_new_entries: bool
    close_all: bool
    telegram_message: str


class DrawdownGuard:
    """
    Tracks peak portfolio value since inception and computes drawdown escalation.
    Peak can be persisted externally (pass starting_peak in constructor).
    """

    def __init__(
        self,
        starting_capital: float,
        limits: RiskLimits,
        starting_peak: Optional[float] = None,
    ):
        self._starting_capital = starting_capital
        self._limits = limits
        self._peak = max(starting_capital, starting_peak or 0.0)

    @property
    def peak_value(self) -> float:
        return self._peak

    def update(self, current_value: float) -> DrawdownState:
        self._peak = max(self._peak, current_value)
        dd_usd = self._peak - current_value
        dd_pct = dd_usd / self._peak if self._peak > 0 else 0.0
        level = self._classify(dd_pct)
        return self._state_for(level, dd_usd, dd_pct, current_value)

    def _classify(self, dd_pct: float) -> DrawdownLevel:
        # Classify on percentage drawdown so thresholds scale with NAV.
        # The absolute-dollar fields on RiskLimits are retained for
        # legacy callers but are no longer the trip condition here.
        if dd_pct >= self._limits.drawdown_l4_pct:
            return DrawdownLevel.KILL
        if dd_pct >= self._limits.drawdown_l3_pct:
            return DrawdownLevel.PRESERVATION
        if dd_pct >= self._limits.drawdown_l2_pct:
            return DrawdownLevel.DEGRADED
        if dd_pct >= self._limits.drawdown_l1_pct:
            return DrawdownLevel.WARNING
        return DrawdownLevel.NORMAL

    def _state_for(
        self, level: DrawdownLevel, dd_usd: float, dd_pct: float, current: float
    ) -> DrawdownState:
        max_new_positions: int
        max_new_position_pct: float
        stop_new_entries: bool
        close_all: bool
        msg: str

        if level is DrawdownLevel.NORMAL:
            max_new_positions = 25
            max_new_position_pct = self._limits.max_position_pct
            stop_new_entries = False
            close_all = False
            msg = ""
        elif level is DrawdownLevel.WARNING:
            max_new_positions = 12
            max_new_position_pct = 0.12
            stop_new_entries = False
            close_all = False
            msg = f"WARNING L1: -${dd_usd:,.0f} ({dd_pct:.1%}) drawdown. Exposure reduced."
        elif level is DrawdownLevel.DEGRADED:
            max_new_positions = 6
            max_new_position_pct = 0.08
            stop_new_entries = False
            close_all = False
            msg = f"ALERT L2: -${dd_usd:,.0f} ({dd_pct:.1%}) drawdown. Degraded mode."
        elif level is DrawdownLevel.PRESERVATION:
            max_new_positions = 2
            max_new_position_pct = 0.05
            stop_new_entries = True
            close_all = False
            msg = f"CRITICAL L3: -${dd_usd:,.0f} ({dd_pct:.1%}) drawdown. Cash preservation."
        else:  # DrawdownLevel.KILL
            max_new_positions = 0
            max_new_position_pct = 0.0
            stop_new_entries = True
            close_all = True
            msg = f"EMERGENCY L4: -${dd_usd:,.0f} ({dd_pct:.1%}) drawdown. KILL SWITCH."

        return DrawdownState(
            level=level,
            drawdown_usd=dd_usd,
            drawdown_pct=dd_pct,
            peak_value=self._peak,
            current_value=current,
            max_new_positions=max_new_positions,
            max_new_position_pct=max_new_position_pct,
            stop_new_entries=stop_new_entries,
            close_all=close_all,
            telegram_message=msg,
        )
