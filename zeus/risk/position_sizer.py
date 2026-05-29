"""Fractional-Kelly position sizing with volatility normalization and multipliers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


REGIME_EXPOSURE_MULT = {
    "BULL": 1.0,
    "RANGE": 0.75,
    "TRANSITION": 0.6,
    "BEAR": 0.5,
    "HIGH_VOL": 0.3,
}


@dataclass
class SizingInputs:
    expected_return_5d: float       # log return forecast over 5 days
    vol_estimate_annual: float      # annualized volatility
    confidence: float               # [0, 1]
    liquidity_adv_usd: float        # 20-day avg daily dollar volume
    regime: str
    recent_ic: float = 0.03         # rolling IC — reduces size if low
    correlation_to_book: float = 0.0  # 0 if independent, 1 if identical


@dataclass
class SizingResult:
    target_pct: float               # fraction of portfolio
    rationale: Dict[str, float]


def kelly_position_size(
    inputs: SizingInputs,
    max_position_pct: float = 0.20,
    kelly_fraction: float = 0.25,
) -> SizingResult:
    """
    Compute target position size as % of portfolio.
    Returns 0.0 if expected return is non-positive.
    """
    rationale: Dict[str, float] = {}

    if inputs.expected_return_5d <= 0 or inputs.vol_estimate_annual <= 0:
        return SizingResult(0.0, {"reason_zero": 1.0})

    annual_return = inputs.expected_return_5d * 252 / 5
    raw_kelly = annual_return / (inputs.vol_estimate_annual ** 2)
    base = max(0.0, raw_kelly * kelly_fraction)
    rationale["base_kelly"] = base

    conf_mult = 0.7 + 0.3 * max(0.0, min(1.0, inputs.confidence))
    rationale["confidence_mult"] = conf_mult

    regime_mult = REGIME_EXPOSURE_MULT.get(inputs.regime, 0.6)
    rationale["regime_mult"] = regime_mult

    liquidity_mult = 1.0
    if inputs.liquidity_adv_usd < 20_000_000:
        liquidity_mult = 0.7
    if inputs.liquidity_adv_usd < 5_000_000:
        liquidity_mult = 0.3
    rationale["liquidity_mult"] = liquidity_mult

    ic_mult = 1.0 if inputs.recent_ic >= 0.02 else 0.5
    rationale["ic_mult"] = ic_mult

    corr_mult = 1.0 - 0.3 * max(0.0, min(1.0, inputs.correlation_to_book - 0.6) / 0.4)
    rationale["corr_mult"] = corr_mult

    target = base * conf_mult * regime_mult * liquidity_mult * ic_mult * corr_mult
    target = min(target, max_position_pct)
    rationale["final"] = target

    return SizingResult(target, rationale)


def shares_from_target(target_pct: float, portfolio_value: float, price: float) -> int:
    """Convert target % to whole shares."""
    if price <= 0:
        return 0
    notional = portfolio_value * target_pct
    return int(notional // price)
