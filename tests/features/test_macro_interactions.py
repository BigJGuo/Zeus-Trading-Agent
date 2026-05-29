"""Tests for `add_macro_interactions`.

The interactions are intentionally additive — never delete or rewrite an
existing column. A regression that drops a base feature here would
silently zero out the model's primary signal, so the tests pin both the
new column shape and the preserved-column invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.features.interactions import (
    DEFAULT_BASE_FEATURES,
    DEFAULT_MACRO_FEATURES,
    add_macro_interactions,
    list_interaction_columns,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "NVDA"],
        "momentum_20d": [0.05, -0.02, 0.10],
        "momentum_5d": [0.01, 0.00, 0.03],
        "rsi_14": [55.0, 48.0, 70.0],
        "atr_14_pct": [1.2, 0.9, 2.4],
        "hist_vol_20d": [0.18, 0.14, 0.32],
        # Macro values broadcast — constant across rows on a real day.
        "vix_level": [20.0, 20.0, 20.0],
        "yield_curve_spread": [0.005, 0.005, 0.005],
        "spy_return_5d": [0.01, 0.01, 0.01],
        "put_call_ratio": [0.95, 0.95, 0.95],
    })


def test_interactions_added_for_every_pair():
    panel = _panel()
    n_before = len(panel.columns)
    add_macro_interactions(panel)
    expected = len(DEFAULT_BASE_FEATURES) * len(DEFAULT_MACRO_FEATURES)
    assert len(panel.columns) == n_before + expected
    for col in list_interaction_columns():
        assert col in panel.columns


def test_values_match_product():
    panel = _panel()
    add_macro_interactions(panel)
    expected = panel["momentum_20d"] * panel["vix_level"]
    pd.testing.assert_series_equal(
        panel["momentum_20d_x_vix_level"],
        expected,
        check_names=False,
    )


def test_existing_columns_not_overwritten():
    panel = _panel()
    snapshot = panel.copy(deep=True)
    add_macro_interactions(panel)
    for col in snapshot.columns:
        pd.testing.assert_series_equal(panel[col], snapshot[col], check_names=False)


def test_skips_pairs_when_columns_absent():
    """Missing macro values shouldn't crash the batch — only the
    available pairs get computed. This happens when FRED + yfinance both
    fail and the macro dict has Nones for everything; we test the
    column-missing case (which the pipeline guards against by always
    setting the keys, but be defensive in the function too)."""
    panel = pd.DataFrame({
        "symbol": ["AAPL"],
        "momentum_20d": [0.05],
        # No vix_level / spy_return_5d / yield_curve_spread at all.
    })
    n_before = len(panel.columns)
    add_macro_interactions(panel)
    assert len(panel.columns) == n_before  # nothing added
    assert "momentum_20d_x_vix_level" not in panel.columns


def test_nan_inputs_propagate_to_nan_outputs():
    panel = _panel()
    panel.loc[0, "vix_level"] = np.nan
    add_macro_interactions(panel)
    # First row has NaN macro → NaN interactions for that row only.
    assert np.isnan(panel.loc[0, "momentum_20d_x_vix_level"])
    assert not np.isnan(panel.loc[1, "momentum_20d_x_vix_level"])
