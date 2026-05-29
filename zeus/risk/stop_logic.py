"""Stop-loss computation: ATR-based floor + trailing stop activation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StopLevels:
    hard_stop: float
    trailing_activation_price: float
    trailing_pct: float
    peak_price: float


def compute_initial_stops(
    entry_price: float,
    atr: float,
    vol_annual: float,
    expected_downside: Optional[float] = None,
    atr_multiple: float = 2.0,
    vol_daily_multiple: float = 1.5,
    trailing_activation_pct: float = 0.02,
    trailing_pct: float = 0.50,
) -> StopLevels:
    """
    Hard stop = highest (least aggressive) of:
      - entry - atr_multiple * ATR
      - entry * (1 - vol_daily_multiple * daily_vol)
      - entry * (1 + expected_downside * 1.5)  if provided and negative
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")

    daily_vol = vol_annual / np.sqrt(252) if vol_annual > 0 else 0.02

    atr_stop = entry_price - atr_multiple * atr if atr > 0 else 0.0
    vol_stop = entry_price * (1.0 - vol_daily_multiple * daily_vol)

    candidates = [atr_stop, vol_stop]
    if expected_downside is not None and expected_downside < 0:
        pred_stop = entry_price * (1.0 + expected_downside * 1.5)
        candidates.append(pred_stop)

    hard_stop = max(c for c in candidates if c > 0)
    hard_stop = min(hard_stop, entry_price * 0.97)  # never tighter than -3%

    return StopLevels(
        hard_stop=hard_stop,
        trailing_activation_price=entry_price * (1.0 + trailing_activation_pct),
        trailing_pct=trailing_pct,
        peak_price=entry_price,
    )


def update_trailing_stop(
    current_price: float,
    entry_price: float,
    peak_price: float,
    hard_stop: float,
    trailing_pct: float,
    trailing_activation_price: float,
) -> tuple[float, float]:
    """
    Returns (new_hard_stop, new_peak_price).
    Trailing only kicks in after price crosses activation.
    Trail at `trailing_pct` of gain from entry — higher = tighter.
    """
    new_peak = max(peak_price, current_price)

    if new_peak < trailing_activation_price:
        return hard_stop, new_peak

    gain = new_peak - entry_price
    trail_floor = entry_price + gain * trailing_pct
    new_hard_stop = max(hard_stop, trail_floor)
    return new_hard_stop, new_peak


def should_exit_on_stop(current_price: float, hard_stop: float) -> bool:
    return current_price <= hard_stop
