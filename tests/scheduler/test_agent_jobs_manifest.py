"""Verify the 7-agent scheduler manifest is well-formed + registrable.

Failure in any of these tests means a production boot would crash at
`build_scheduler(loop)` before the market ever opens. Cheap insurance.
"""
from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone

from zeus.scheduler.agent_jobs import AGENT_JOB_MANIFEST

ET = timezone("America/New_York")


def test_manifest_entries_have_required_keys():
    for entry in AGENT_JOB_MANIFEST:
        assert "func" in entry and callable(entry["func"])
        assert "trigger" in entry and entry["trigger"] in ("cron", "interval")


def test_every_job_registers_with_unique_id():
    """Multiple rows can share the same function name (premarket scan fires
    3×). The runner must produce a unique APScheduler id per row."""
    scheduler = BackgroundScheduler(timezone=ET)
    seen_ids = set()
    for row in AGENT_JOB_MANIFEST:
        params = dict(row)
        trigger_type = params.pop("trigger")
        fn = params.pop("func")
        if trigger_type == "cron":
            trigger = CronTrigger(timezone=ET, **params)
        else:
            trigger = IntervalTrigger(**params)
        job_id = f"{fn.__name__}_{trigger}"
        assert job_id not in seen_ids, (
            f"duplicate job id: {job_id} — the runner loop would crash"
        )
        seen_ids.add(job_id)
        # Jobs are parameterless after the SQLAlchemyJobStore refactor —
        # they fetch `orch` from `zeus.scheduler.context` at call time.
        scheduler.add_job(
            fn, trigger=trigger,
            id=job_id, misfire_grace_time=300, coalesce=True,
        )
    # Never `.start()` — just build so we know APScheduler accepts every
    # trigger spec. Shutdown is a no-op for a never-started scheduler.
    assert len(scheduler.get_jobs()) == len(AGENT_JOB_MANIFEST)


def test_critical_cadences_present():
    """Smoke: the jobs the user actually relies on for overnight prep must
    all appear in the manifest. If any goes missing, tomorrow's session runs
    blind."""
    required_names = {
        "swing_research_eod_job",
        "long_term_deep_dive_or_review_job",
        "overseer_daily_aggregate_job",
        "trader_overnight_plan_job",
        "day_research_premarket_job",
        "day_trader_morning_plan_job",
        # overseer_realtime_monitor_job was folded into intraday_monitor_job
        # (JOB_MANIFEST), so it no longer appears in AGENT_JOB_MANIFEST.
        "day_research_postmarket_wrap_job",
        "overseer_weekly_review_job",
    }
    names = {row["func"].__name__ for row in AGENT_JOB_MANIFEST}
    missing = required_names - names
    assert not missing, f"missing critical cadences: {missing}"


def test_trader_plan_job_fires_after_model_plan():
    """The 21:00 model-based plan must precede the 21:30 LLM trader refine.
    Otherwise the LLM receives stale `ctx.current_plan.entries` (or None)."""
    overnight = [
        r for r in AGENT_JOB_MANIFEST
        if r["func"].__name__ == "trader_overnight_plan_job"
    ]
    assert len(overnight) == 1
    entry = overnight[0]
    # trigger should be a cron with hour >= 21
    assert entry["trigger"] == "cron"
    assert entry.get("hour", 0) >= 21
