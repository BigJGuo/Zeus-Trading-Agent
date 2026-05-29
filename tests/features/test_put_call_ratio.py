"""Tests for the CBOE Put/Call ratio alt-data fetch.

These don't hit the network in CI — they monkeypatch the fetchers so the
contract (Series shape, dict keys, fallback chain) is what's verified.
A live integration test belongs in a separate `tests/integration/` tree
that the user runs manually when adding a new alt-data source.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

import zeus.features.macro as macro


@pytest.fixture(autouse=True)
def _disable_cache(monkeypatch):
    """Force the macro cache to a no-op so each test stands alone."""
    monkeypatch.setattr(macro, "_get_cache", lambda: None)


def _zero_macro_calls(monkeypatch):
    """Stub every other macro fetch so the test exercises only the P/C path."""
    monkeypatch.setattr(macro, "_fetch_from_fred", lambda *a, **k: None)
    monkeypatch.setattr(macro, "_fetch_yfinance_series", lambda *a, **k: None)


def test_compute_macro_includes_put_call_keys(monkeypatch):
    """Even when both stooq + yfinance fall through, the dict must include
    the put/call keys (with None values) — downstream code reads them by
    name and a KeyError would break the pipeline."""
    _zero_macro_calls(monkeypatch)
    monkeypatch.setattr(macro, "_fetch_put_call_series", lambda *a, **k: None)

    out = macro.compute_macro_features(date(2026, 5, 1))
    assert "put_call_ratio" in out
    assert "put_call_ratio_5d_change" in out
    assert out["put_call_ratio"] is None
    assert out["put_call_ratio_5d_change"] is None


def test_put_call_populates_when_series_available(monkeypatch):
    _zero_macro_calls(monkeypatch)

    fake_series = pd.Series(
        [0.85, 0.90, 0.92, 0.88, 0.95, 1.05, 1.10],
        index=pd.date_range("2026-04-23", periods=7),
        name="Close",
    )
    monkeypatch.setattr(macro, "_fetch_put_call_series", lambda *a, **k: fake_series)

    out = macro.compute_macro_features(date(2026, 5, 1))
    assert out["put_call_ratio"] == pytest.approx(1.10)
    # 5d change: last (1.10) - 5-back (index 1 = 0.90). Helper uses -6
    # which on a 7-element series is index 1.
    assert out["put_call_ratio_5d_change"] == pytest.approx(1.10 - 0.90)


def test_5d_change_none_when_series_too_short(monkeypatch):
    _zero_macro_calls(monkeypatch)

    short_series = pd.Series(
        [0.85, 0.90, 0.92],
        index=pd.date_range("2026-04-29", periods=3),
        name="Close",
    )
    monkeypatch.setattr(macro, "_fetch_put_call_series", lambda *a, **k: short_series)

    out = macro.compute_macro_features(date(2026, 5, 1))
    assert out["put_call_ratio"] == pytest.approx(0.92)
    assert out["put_call_ratio_5d_change"] is None
