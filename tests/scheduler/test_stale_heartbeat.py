"""Tests for `stale_heartbeat_check_job`.

The job's contract:
  - emits a CRITICAL `heartbeat_stale` RiskEvent when the most recent
    `trading_loop` heartbeat is older than 10 minutes.
  - throttles re-alerts so a wedged trading_loop doesn't flood the
    risk_events log with the same warning every minute.
  - is a no-op on first boot (no heartbeats yet).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from zeus.data.storage.database import Heartbeat, RiskEvent


def _seed_heartbeat(session, *, age_minutes: float, component: str = "trading_loop"):
    session.add(Heartbeat(
        ts=datetime.now(timezone.utc) - timedelta(minutes=age_minutes),
        component=component,
        status="ok",
        details=None,
    ))
    session.commit()


def test_no_heartbeats_first_boot_is_silent(monkeypatch, patched_session_factory):
    """First boot — heartbeats table is empty. Job must NOT alert: the
    next heartbeat tick will write one, and we don't want to spam."""
    from zeus.scheduler import jobs

    jobs.stale_heartbeat_check_job()

    with patched_session_factory() as s:
        events = s.execute(select(RiskEvent)).scalars().all()
        assert events == []


def test_fresh_heartbeat_no_alert(monkeypatch, patched_session_factory):
    from zeus.scheduler import jobs

    with patched_session_factory() as s:
        _seed_heartbeat(s, age_minutes=2)

    jobs.stale_heartbeat_check_job()

    with patched_session_factory() as s:
        events = s.execute(select(RiskEvent)).scalars().all()
        assert events == []


def test_stale_heartbeat_emits_critical_risk_event(monkeypatch, patched_session_factory):
    from zeus.scheduler import jobs

    with patched_session_factory() as s:
        _seed_heartbeat(s, age_minutes=15)

    jobs.stale_heartbeat_check_job()

    with patched_session_factory() as s:
        events = s.execute(select(RiskEvent)).scalars().all()
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "heartbeat_stale"
        assert ev.severity == "CRITICAL"
        assert "stale" in (ev.description or "").lower()


def test_throttles_repeat_alerts_within_window(monkeypatch, patched_session_factory):
    """A wedged trading_loop would otherwise generate 10 alerts in 10
    minutes. Verify the throttle: a recent stale-heartbeat alert means
    the next tick is a no-op."""
    from zeus.scheduler import jobs

    with patched_session_factory() as s:
        _seed_heartbeat(s, age_minutes=15)

    # First tick — writes the alert.
    jobs.stale_heartbeat_check_job()
    # Second tick — should NOT write a second alert because one exists
    # within the throttle window.
    jobs.stale_heartbeat_check_job()

    with patched_session_factory() as s:
        events = s.execute(select(RiskEvent)).scalars().all()
        assert len(events) == 1, (
            f"expected throttle to suppress repeat alert, got {len(events)} events"
        )


def test_non_trading_loop_heartbeat_does_not_count(monkeypatch, patched_session_factory):
    """A 'scheduler' or 'data_refresh' heartbeat shouldn't substitute for a
    trading_loop heartbeat — those are different components and trading_loop
    can wedge while the scheduler thread keeps writing its own beats."""
    from zeus.scheduler import jobs

    with patched_session_factory() as s:
        _seed_heartbeat(s, age_minutes=2, component="data_refresh")
        # Trading loop heartbeat is missing entirely → behaves like first boot,
        # *not* like a stale alert (no prior trading_loop beat to be stale
        # relative to).
    jobs.stale_heartbeat_check_job()

    with patched_session_factory() as s:
        events = s.execute(select(RiskEvent)).scalars().all()
        assert events == []
