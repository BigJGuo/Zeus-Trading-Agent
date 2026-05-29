"""Session state machine for the live trading loop."""
from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from zeus.scheduler.market_schedule import (
    is_market_open,
    is_premarket,
    is_after_hours,
    is_trading_day,
)

ET = ZoneInfo("America/New_York")


class SessionState(str, Enum):
    PREMARKET = "premarket"
    MARKET_OPEN = "market_open"
    INTRADAY = "intraday"
    CLOSING = "closing"        # last 30 min
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"
    EMERGENCY = "emergency"


def classify_session(now: Optional[datetime] = None) -> SessionState:
    """Classify the current moment into a session state."""
    if now is None:
        now = datetime.now(tz=ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)

    today = now.date()
    if not is_trading_day(today):
        return SessionState.WEEKEND

    if is_premarket(now):
        return SessionState.PREMARKET

    if is_market_open(now):
        t = now.time()
        # Market open (9:30) → first 30 seconds is "just opened"
        if t < time(9, 31):
            return SessionState.MARKET_OPEN
        if t >= time(15, 30):
            return SessionState.CLOSING
        return SessionState.INTRADAY

    if is_after_hours(now):
        return SessionState.AFTER_HOURS

    return SessionState.WEEKEND
