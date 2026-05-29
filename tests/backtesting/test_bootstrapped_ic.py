"""Tests for the bootstrap-resampled IC helper.

The math is sensitive to two regressions:
  1. Point estimate diverging from `information_coefficient` (different
     code path, must stay calibrated).
  2. LCB collapsing to 0 silently when group-level resampling has too few
     groups — the gate would let in any model in that case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zeus.backtesting.metrics import bootstrapped_ic, information_coefficient


def _rng_pair(rho: float, n: int = 200, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Synthesize predictions, actuals with target Spearman correlation."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    y = rho * x + np.sqrt(max(1 - rho * rho, 1e-6)) * noise
    return pd.Series(x), pd.Series(y)


def test_point_estimate_matches_information_coefficient():
    preds, actuals = _rng_pair(rho=0.15, n=500)
    stats = bootstrapped_ic(preds, actuals, n_resamples=200)
    ic = information_coefficient(preds, actuals)
    assert abs(stats["ic"] - ic) < 1e-9


def test_lcb_is_below_point_estimate_for_positive_ic():
    preds, actuals = _rng_pair(rho=0.10, n=500)
    stats = bootstrapped_ic(preds, actuals, n_resamples=500, lcb_percentile=5.0)
    assert stats["ic_lcb"] < stats["ic"], (
        "5th-pct of bootstrap dist must lie below the point estimate"
    )


def test_lcb_negative_for_noise_only_data():
    """When predictions are pure noise, the 5th-percentile bootstrap IC
    should generally land below zero — that's exactly the population of
    models the LCB gate is supposed to filter out."""
    preds, actuals = _rng_pair(rho=0.0, n=300)
    stats = bootstrapped_ic(preds, actuals, n_resamples=500)
    # Allow a small tolerance; rho=0 means the dist is centered around 0 so
    # the 5th-pct is approximately -1.65 * std_err.
    assert stats["ic_lcb"] < 0.0


def test_grouped_resampling_returns_finite_lcb():
    """Per-date resampling: the helper picks groups (dates) with
    replacement. A handful of synthesized dates should still produce a
    finite LCB rather than NaN."""
    rng = np.random.default_rng(42)
    dates = pd.Series(
        np.repeat(pd.date_range("2026-01-01", periods=30), 20).astype("datetime64[ns]")
    )
    n = len(dates)
    preds = pd.Series(rng.standard_normal(n))
    actuals = 0.20 * preds + pd.Series(rng.standard_normal(n) * 0.8)

    stats = bootstrapped_ic(preds, actuals, group=dates, n_resamples=400)
    assert np.isfinite(stats["ic_lcb"])
    assert np.isfinite(stats["ic"])
    assert stats["ic_lcb"] < stats["ic"]


def test_deterministic_with_fixed_seed():
    """Same seed → same LCB, so promotion gates are reproducible across
    re-runs of the trainer."""
    preds, actuals = _rng_pair(rho=0.12, n=200)
    a = bootstrapped_ic(preds, actuals, n_resamples=300, seed=7)
    b = bootstrapped_ic(preds, actuals, n_resamples=300, seed=7)
    assert a["ic_lcb"] == b["ic_lcb"]
    assert a["ic_mean"] == b["ic_mean"]


def test_empty_or_tiny_input_returns_zeros_not_nan():
    """If the caller passes < 2 paired observations, we still return a
    well-formed dict — the trainer relies on these fields being numeric
    when it builds the metrics row."""
    stats = bootstrapped_ic(pd.Series([1.0]), pd.Series([0.5]), n_resamples=50)
    assert stats["ic_lcb"] == 0.0
    assert stats["ic_mean"] == 0.0
    assert np.isfinite(stats["ic"])
