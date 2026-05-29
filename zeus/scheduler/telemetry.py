"""APScheduler telemetry listener — populates scheduler_jobs and
job_executions so the dashboard's "Today's schedule" panel has data.

Wiring:

    from zeus.scheduler.telemetry import attach_scheduler_telemetry
    attach_scheduler_telemetry(scheduler)

Events handled:
  * EVENT_JOB_ADDED / EVENT_JOB_MODIFIED — upsert SchedulerJob row,
    refresh next_run_time + trigger_repr.
  * EVENT_JOB_SUBMITTED — insert JobExecution(status='submitted').
  * EVENT_JOB_EXECUTED  — close the matching JobExecution with
    status='success' and update SchedulerJob.last_*.
  * EVENT_JOB_ERROR     — close as 'error' with traceback.
  * EVENT_JOB_MISSED    — append JobExecution(status='missed').

All writes are best-effort; any DB failure is logged and swallowed so a
broken telemetry path never takes down a trading job.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from apscheduler.events import (
    EVENT_JOB_ADDED,
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    EVENT_JOB_MODIFIED,
    EVENT_JOB_SUBMITTED,
)
from sqlalchemy import select, update

from zeus.data.storage.database import (
    JobExecution,
    SchedulerJob,
    get_session_factory,
)

log = structlog.get_logger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _upsert_scheduler_job(scheduler, job_id: str) -> None:
    """Refresh one row in scheduler_jobs from APScheduler's view of the job."""
    try:
        job = scheduler.get_job(job_id)
    except Exception:
        job = None
    if job is None:
        return

    trigger_repr = str(job.trigger) if job.trigger is not None else None
    func_name = getattr(job.func, "__name__", str(job.func))
    next_run = _as_utc(job.next_run_time)

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        existing = session.get(SchedulerJob, job_id)
        if existing is None:
            row = SchedulerJob(
                job_id=job_id,
                func_name=func_name,
                trigger_repr=trigger_repr,
                next_run_time=next_run,
                updated_at=_now_utc(),
            )
            session.add(row)
        else:
            existing.func_name = func_name
            existing.trigger_repr = trigger_repr
            existing.next_run_time = next_run
            existing.updated_at = _now_utc()
        session.commit()
    except Exception as e:
        session.rollback()
        log.warning("telemetry_upsert_job_failed", job_id=job_id, error=str(e))
    finally:
        session.close()


def _record_submission(scheduler, event) -> None:
    job_id = event.job_id
    scheduled_ts = _as_utc(getattr(event, "scheduled_run_time", None))
    try:
        job = scheduler.get_job(job_id)
    except Exception:
        job = None
    func_name = getattr(getattr(job, "func", None), "__name__", job_id) if job else job_id

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        row = JobExecution(
            job_id=job_id,
            func_name=func_name,
            scheduled_ts=scheduled_ts,
            started_ts=_now_utc(),
            status="submitted",
        )
        session.add(row)
        session.commit()
    except Exception as e:
        session.rollback()
        log.warning("telemetry_record_submission_failed", job_id=job_id, error=str(e))
    finally:
        session.close()


def _find_open_submission(session, job_id: str, scheduled_ts: Optional[datetime]):
    """Find the most recent 'submitted' JobExecution for this job that hasn't
    been closed out yet. We match on job_id + scheduled_ts when available so
    overlapping fires don't clobber each other."""
    stmt = select(JobExecution).where(
        JobExecution.job_id == job_id,
        JobExecution.status == "submitted",
    )
    if scheduled_ts is not None:
        stmt = stmt.where(JobExecution.scheduled_ts == scheduled_ts)
    stmt = stmt.order_by(JobExecution.started_ts.desc()).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def _close_submission(
    scheduler,
    event,
    *,
    success: bool,
    error_text: Optional[str] = None,
) -> None:
    job_id = event.job_id
    scheduled_ts = _as_utc(getattr(event, "scheduled_run_time", None))
    now = _now_utc()

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        row = _find_open_submission(session, job_id, scheduled_ts)
        duration_ms: Optional[int] = None
        if row is None:
            # No submission row — either we missed the SUBMITTED event or the
            # listener registered after startup. Synthesize one.
            started = scheduled_ts or now
            row = JobExecution(
                job_id=job_id,
                func_name=job_id,
                scheduled_ts=scheduled_ts,
                started_ts=started,
                finished_ts=now,
                status="success" if success else "error",
                error=error_text,
            )
            if row.started_ts and row.finished_ts:
                duration_ms = int(
                    (row.finished_ts - row.started_ts).total_seconds() * 1000
                )
            row.duration_ms = duration_ms
            session.add(row)
        else:
            row.finished_ts = now
            row.status = "success" if success else "error"
            row.error = error_text
            if row.started_ts is not None:
                started_utc = _as_utc(row.started_ts)
                if started_utc is not None:
                    duration_ms = int((now - started_utc).total_seconds() * 1000)
                    row.duration_ms = duration_ms

        # Also reflect outcome in scheduler_jobs.
        job = session.get(SchedulerJob, job_id)
        try:
            scheduler_job = scheduler.get_job(job_id)
        except Exception:
            scheduler_job = None
        next_run = _as_utc(scheduler_job.next_run_time) if scheduler_job else None
        func_name = (
            getattr(scheduler_job.func, "__name__", job_id)
            if scheduler_job else (job.func_name if job else job_id)
        )
        if job is None:
            job = SchedulerJob(
                job_id=job_id,
                func_name=func_name,
                trigger_repr=(
                    str(scheduler_job.trigger) if scheduler_job is not None else None
                ),
                next_run_time=next_run,
                updated_at=now,
            )
            session.add(job)
        else:
            job.func_name = func_name
            if scheduler_job is not None and scheduler_job.trigger is not None:
                job.trigger_repr = str(scheduler_job.trigger)
            job.next_run_time = next_run
            job.updated_at = now
        job.last_run_time = now
        job.last_duration_ms = duration_ms
        job.last_success = success
        job.last_error = error_text

        session.commit()
    except Exception as e:
        session.rollback()
        log.warning("telemetry_close_submission_failed", job_id=job_id, error=str(e))
    finally:
        session.close()


def _record_missed(scheduler, event) -> None:
    job_id = event.job_id
    scheduled_ts = _as_utc(getattr(event, "scheduled_run_time", None))
    try:
        job = scheduler.get_job(job_id)
    except Exception:
        job = None
    func_name = getattr(getattr(job, "func", None), "__name__", job_id) if job else job_id
    now = _now_utc()

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        row = JobExecution(
            job_id=job_id,
            func_name=func_name,
            scheduled_ts=scheduled_ts,
            started_ts=scheduled_ts or now,
            finished_ts=now,
            duration_ms=0,
            status="missed",
            error="misfire",
        )
        session.add(row)
        session.commit()
    except Exception as e:
        session.rollback()
        log.warning("telemetry_record_missed_failed", job_id=job_id, error=str(e))
    finally:
        session.close()


def attach_scheduler_telemetry(scheduler) -> None:
    """Register listeners + seed scheduler_jobs with all currently-registered
    jobs. Call this AFTER add_job() calls but BEFORE scheduler.start()."""

    def on_event(event):
        etype = event.code
        try:
            if etype in (EVENT_JOB_ADDED, EVENT_JOB_MODIFIED):
                _upsert_scheduler_job(scheduler, event.job_id)
            elif etype == EVENT_JOB_SUBMITTED:
                _record_submission(scheduler, event)
            elif etype == EVENT_JOB_EXECUTED:
                _close_submission(scheduler, event, success=True)
            elif etype == EVENT_JOB_ERROR:
                exc = getattr(event, "exception", None)
                tb = getattr(event, "traceback", None)
                err = str(exc) if exc is not None else (tb or "error")
                _close_submission(scheduler, event, success=False, error_text=err[:4000])
            elif etype == EVENT_JOB_MISSED:
                _record_missed(scheduler, event)
        except Exception as e:
            log.warning("telemetry_listener_failed", event_type=etype, error=str(e))

    mask = (
        EVENT_JOB_ADDED | EVENT_JOB_MODIFIED | EVENT_JOB_SUBMITTED
        | EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
    )
    scheduler.add_listener(on_event, mask)

    # Seed: upsert every currently-registered job so the checklist endpoint
    # has rows to render even before the first fire.
    for job in scheduler.get_jobs():
        _upsert_scheduler_job(scheduler, job.id)
    log.info(
        "scheduler_telemetry_attached", seeded=len(scheduler.get_jobs()),
    )
