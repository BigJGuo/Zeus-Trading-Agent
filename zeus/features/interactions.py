"""Macro × per-symbol feature interaction terms.

Broadcasting VIX, the yield-curve spread, and 5-day SPY return as constants
across every symbol gives tree models nothing to learn cross-sectionally —
every row on a given date has the same macro value, so the model can only
mix macro non-linearly with other features, which gradient-boosted trees
do poorly relative to additive interaction features.

Lifting those constants to *interaction* columns (`feature × macro`) puts
the variation back: now a high-VIX day has different signal for a
high-momentum vs low-momentum name. The cross-section gains a real
dimension the model can split on. Cheap to compute, narrows the gap
between conventional features and "real" macro-regime conditioning
without the cost of training separate per-regime sub-models.

Design
──────
We pick a small set of base features that are themselves cross-sectional
(per-symbol momentum, vol, mean-reversion proxies) and cross them against
a small set of macro features whose direction we have priors on
(VIX up → momentum reverses; yield-curve flat → low-quality outperforms;
SPY drawdown → defensives bid). Total new columns: |bases| × |macros|.

Pinning the lists down rather than going exhaustive keeps the feature
dimensionality from blowing up (32 features × 3 macro = 96 interactions,
which we don't need); leave further expansion to deliberate iteration
once we can measure their marginal IC from realized trades.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

# Per-symbol features with the strongest cross-sectional signal — these
# are what the model needs to conditionally re-weight by macro regime.
DEFAULT_BASE_FEATURES: tuple[str, ...] = (
    "momentum_20d", "momentum_5d", "momentum_60d",
    "rsi_14", "rsi_5",
    "atr_14_pct", "hist_vol_20d",
    "roc_20", "bb_width_20", "vwap_deviation",
)

# Macro features that change *meaningfully* across the daily cadence. We
# intentionally exclude fed_funds_rate (changes at FOMC frequency, gives
# trees nothing to split on within a quarter).
DEFAULT_MACRO_FEATURES: tuple[str, ...] = (
    "vix_level", "vix_5d_change",
    "yield_curve_spread",
    "spy_return_5d", "spy_return_20d",
    "put_call_ratio", "put_call_ratio_5d_change",
)


def add_macro_interactions(
    panel: pd.DataFrame,
    *,
    base_features: Iterable[str] = DEFAULT_BASE_FEATURES,
    macro_features: Iterable[str] = DEFAULT_MACRO_FEATURES,
) -> pd.DataFrame:
    """Append `f_x_m` columns for every (base, macro) pair where both exist.

    Mutates the input frame in-place and returns it. Skips pairs where
    either column is missing — feature_pipeline already drops symbols with
    insufficient OHLCV history, so the base columns may be absent for new
    listings, and a missing macro key shouldn't crash the batch (the
    pipeline tolerates partial coverage upstream).
    """
    cols = set(panel.columns)
    added = 0
    for base in base_features:
        if base not in cols:
            continue
        for macro in macro_features:
            if macro not in cols:
                continue
            new_col = f"{base}_x_{macro}"
            # Multiplication propagates NaN cleanly — symbols with NaN
            # base feature OR NaN macro field get NaN interaction, which
            # is the right semantic.
            panel[new_col] = panel[base] * panel[macro]
            added += 1
    return panel


def list_interaction_columns(
    base_features: Iterable[str] = DEFAULT_BASE_FEATURES,
    macro_features: Iterable[str] = DEFAULT_MACRO_FEATURES,
) -> list[str]:
    """Convenience: the full cross-product of column names. Used by trainers
    that need an explicit feature list."""
    return [f"{b}_x_{m}" for b in base_features for m in macro_features]
