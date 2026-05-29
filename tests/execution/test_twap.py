"""Tests for the TWAP slicing helper.

Slice-sizing is the part that's easy to get subtly wrong (integer
rounding errors, off-by-one share-count drift across slices). The
scheduler-integration path is exercised separately at the integration
level — these tests focus on the deterministic math.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from zeus.execution.twap import (
    DEFAULT_NUM_SLICES,
    DEFAULT_WINDOW_MINUTES,
    MIN_SHARES_FOR_TWAP,
    compute_twap_slices,
)


def test_slices_sum_to_total_shares():
    """The fundamental invariant — we never lose or gain a share to
    rounding across the slice plan."""
    for total in [50, 73, 100, 251, 1000, 9999]:
        slices = compute_twap_slices(total, num_slices=5)
        assert sum(s.shares for s in slices) == total, f"drift at total={total}"


def test_small_orders_fall_through_to_single_fill():
    """Below the threshold, TWAP is more overhead than alpha — emit a single
    slice and let the caller treat it like a normal market submit."""
    slices = compute_twap_slices(MIN_SHARES_FOR_TWAP - 1, num_slices=5)
    assert len(slices) == 1
    assert slices[0].shares == MIN_SHARES_FOR_TWAP - 1
    assert slices[0].total_slices == 1


def test_zero_or_negative_shares_returns_empty():
    assert compute_twap_slices(0) == []
    assert compute_twap_slices(-10) == []


def test_first_slice_carries_the_remainder():
    """The remainder rides on slice 1 so the bulk of the fill lands first.
    If we put the remainder on the *last* slice, a partial-day halt mid-window
    leaves more unfilled qty than necessary."""
    # 103 shares / 5 slices = 20 each + 3 remainder → slice 1 = 23, others 20.
    slices = compute_twap_slices(103, num_slices=5)
    assert len(slices) == 5
    assert slices[0].shares == 23
    for s in slices[1:]:
        assert s.shares == 20


def test_slice_timestamps_span_the_full_window():
    """Slice 1 at start, slice N at start + window. Equal spacing in between."""
    start = datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc)
    slices = compute_twap_slices(500, num_slices=5, window_minutes=12, start=start)
    assert len(slices) == 5
    assert slices[0].submit_at == start
    # window_minutes=12, 5 slices → step = 3 min, last slice at 09:42
    assert slices[-1].submit_at == start + timedelta(minutes=12)
    # Even spacing
    deltas = [
        (slices[i + 1].submit_at - slices[i].submit_at).total_seconds()
        for i in range(len(slices) - 1)
    ]
    assert all(abs(d - 180.0) < 1e-6 for d in deltas), f"uneven slice gaps: {deltas}"


def test_single_slice_when_num_slices_is_one():
    slices = compute_twap_slices(500, num_slices=1)
    assert len(slices) == 1
    assert slices[0].shares == 500
    assert slices[0].slice_index == 0
    assert slices[0].total_slices == 1


def test_slice_indices_are_sequential():
    slices = compute_twap_slices(500, num_slices=5)
    assert [s.slice_index for s in slices] == [0, 1, 2, 3, 4]
    assert all(s.total_slices == 5 for s in slices)
