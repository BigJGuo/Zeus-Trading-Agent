"""
Market calendar and trading hours logic for ZEUS.
Uses pandas_market_calendars for NYSE schedule.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")
_nyse = mcal.get_calendar("NYSE")


def get_trading_days(start: date, end: date) -> List[date]:
    """Return list of NYSE trading days between start and end (inclusive)."""
    schedule = _nyse.schedule(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    return [d.date() for d in schedule.index]


def is_trading_day(d: Optional[date] = None) -> bool:
    """Return True if the given date (default: today) is a NYSE trading day."""
    d = d or date.today()
    schedule = _nyse.schedule(
        start_date=d.strftime("%Y-%m-%d"),
        end_date=d.strftime("%Y-%m-%d"),
    )
    return not schedule.empty


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return True if the NYSE is currently open for trading."""
    now = now or datetime.now(tz=ET)
    today = now.date()
    if not is_trading_day(today):
        return False
    schedule = _nyse.schedule(
        start_date=today.strftime("%Y-%m-%d"),
        end_date=today.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        return False
    market_open = schedule.iloc[0]["market_open"].to_pydatetime().replace(tzinfo=ET)
    market_close = schedule.iloc[0]["market_close"].to_pydatetime().replace(tzinfo=ET)
    return market_open <= now < market_close


def market_open_time(d: Optional[date] = None) -> Optional[datetime]:
    """Return the market open datetime for the given date."""
    d = d or date.today()
    schedule = _nyse.schedule(
        start_date=d.strftime("%Y-%m-%d"),
        end_date=d.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        return None
    return schedule.iloc[0]["market_open"].to_pydatetime().replace(tzinfo=ET)


def market_close_time(d: Optional[date] = None) -> Optional[datetime]:
    """Return the market close datetime for the given date."""
    d = d or date.today()
    schedule = _nyse.schedule(
        start_date=d.strftime("%Y-%m-%d"),
        end_date=d.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        return None
    return schedule.iloc[0]["market_close"].to_pydatetime().replace(tzinfo=ET)


def minutes_to_close(now: Optional[datetime] = None) -> int:
    """Return minutes until market close. Returns 0 if market is closed."""
    now = now or datetime.now(tz=ET)
    close = market_close_time(now.date())
    if close is None or now >= close:
        return 0
    return int((close - now).total_seconds() / 60)


def next_trading_day(d: Optional[date] = None) -> date:
    """Return the next NYSE trading day after d."""
    d = d or date.today()
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def prev_trading_day(d: Optional[date] = None) -> date:
    """Return the last NYSE trading day before d."""
    d = d or date.today()
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def is_premarket(now: Optional[datetime] = None) -> bool:
    """Return True if current time is in the pre-market window (7:00–9:30 AM ET)."""
    now = now or datetime.now(tz=ET)
    if not is_trading_day(now.date()):
        return False
    open_time = market_open_time(now.date())
    if open_time is None:
        return False
    premarket_start = open_time.replace(hour=7, minute=0, second=0)
    return premarket_start <= now < open_time


def is_after_hours(now: Optional[datetime] = None) -> bool:
    """Return True if market is closed but it's a weekday."""
    now = now or datetime.now(tz=ET)
    if not is_trading_day(now.date()):
        return False
    return not is_market_open(now) and not is_premarket(now)


def get_n_trading_days_ago(n: int, d: Optional[date] = None) -> date:
    """Return the trading day n days before d."""
    d = d or date.today()
    days = get_trading_days(d - timedelta(days=n * 2), d)
    if len(days) < n + 1:
        raise ValueError(f"Not enough trading days: requested {n} days before {d}")
    return days[-(n + 1)]
