"""Exercise the `_startup_catchup` path.

This is the safety net for a fresh process boot mid-morning on a weekday:
the scheduler won't re-fire the 19:00/21:30/9:00 windows that already
passed. `_startup_catchup` synchronously runs them so the morning session
inherits fresh research + plans. A bug here silently zeros out the whole
overnight pipeline on the first day after a restart.
"""
from __future__ import annotations

import datetime as _dt
from datetime import date
from types import SimpleNamespace

import pytest


class _FakePlan:
    def __init__(self, plan_date: date):
        self.plan_date = plan_date
        self.entries: list = []
        self.exits: list = []


class _FakeContext:
    def __init__(self, plan=None):
        self.current_plan = plan


class _FakeSM:
    def __init__(self, ctxs):
        self._ctxs = ctxs

    def strategy_ids(self):
        return list(self._ctxs.keys())

    def context(self, sid):
        return self._ctxs[sid]


class _FakeLoop:
    def __init__(self, sm=None):
        self._strategy_mgr = sm
        self.run_next_session_calls = 0

    def run_next_session_planning(self):
        self.run_next_session_calls += 1


def _fake_orch():
    return SimpleNamespace(
        bundle=None, strategy_manager=None, regime_fn=None,
        current_regime=lambda: "normal",
    )


def _freeze_clock(monkeypatch, when: _dt.datetime) -> None:
    """Make every `datetime.now(tz)` in the catchup return `when`."""
    class _Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return when if tz is None else when.replace(tzinfo=tz)
    monkeypatch.setattr(_dt, "datetime", _Frozen)


def _stub_agent_jobs(monkeypatch, counter: dict) -> None:
    names = (
        "swing_research_eod_job",
        "long_term_deep_dive_or_review_job",
        "day_research_premarket_job",
        "trader_overnight_plan_job",
        "day_trader_morning_plan_job",
    )
    for n in names:
        counter[n] = 0

    def _factory(name):
        # Jobs are now parameterless (they fetch orch via the registry);
        # the stub just bumps a counter so the catchup paths can be
        # exercised without spinning up real agents.
        def _impl():
            counter[name] += 1
        return _impl

    for n in names:
        monkeypatch.setattr(f"zeus.scheduler.agent_jobs.{n}", _factory(n))


def test_weekend_early_return(monkeypatch):
    """Sat/Sun should leave everything to the regular cadence."""
    import zeus.scheduler.runner as r

    counter: dict = {}
    _stub_agent_jobs(monkeypatch, counter)

    loop = _FakeLoop(sm=_FakeSM({"day": _FakeContext()}))
    # 2026-04-18 is a Saturday
    _freeze_clock(monkeypatch, _dt.datetime(2026, 4, 18, 7, 0))

    r._startup_catchup(loop, _fake_orch())

    assert loop.run_next_session_calls == 0
    assert all(v == 0 for v in counter.values()), (
        f"agent jobs should not fire on a weekend: {counter}"
    )


def test_plan_missing_fires_model_plan_and_agent_jobs(monkeypatch):
    """Pre-09:30 weekday with no `current_plan` — catchup must build the
    model plan AND fire all overnight agent windows."""
    import zeus.scheduler.runner as r

    counter: dict = {}
    _stub_agent_jobs(monkeypatch, counter)

    loop = _FakeLoop(sm=_FakeSM({
        "day": _FakeContext(plan=None),
        "swing": _FakeContext(plan=None),
    }))
    # 2026-04-20 Monday 07:00 ET
    _freeze_clock(monkeypatch, _dt.datetime(2026, 4, 20, 7, 0))

    r._startup_catchup(loop, _fake_orch())

    assert loop.run_next_session_calls == 1
    for name, n in counter.items():
        assert n == 1, f"{name} should have fired exactly once, got {n}"


def test_fresh_plan_skips_model_rebuild_but_runs_briefs(monkeypatch):
    """Plan is already dated for today — don't rerun the model, but do run
    the premarket briefs (journal rows for today haven't been written yet)."""
    import zeus.scheduler.runner as r

    counter: dict = {}
    _stub_agent_jobs(monkeypatch, counter)

    today = date(2026, 4, 20)
    loop = _FakeLoop(sm=_FakeSM({
        "day": _FakeContext(plan=_FakePlan(today)),
        "swing": _FakeContext(plan=_FakePlan(today)),
    }))
    _freeze_clock(monkeypatch, _dt.datetime(2026, 4, 20, 8, 0))

    r._startup_catchup(loop, _fake_orch())

    assert loop.run_next_session_calls == 0
    assert counter["day_research_premarket_job"] == 1


def test_no_orch_still_builds_model_plan(monkeypatch):
    """Agent system disabled (no Anthropic key) — catchup still runs the
    model-plan refresh, then exits without touching any agent job."""
    import zeus.scheduler.runner as r

    counter: dict = {}
    _stub_agent_jobs(monkeypatch, counter)

    loop = _FakeLoop(sm=_FakeSM({"day": _FakeContext(plan=None)}))
    _freeze_clock(monkeypatch, _dt.datetime(2026, 4, 20, 7, 0))

    r._startup_catchup(loop, orch=None)

    assert loop.run_next_session_calls == 1
    assert all(v == 0 for v in counter.values())
