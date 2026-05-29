"""Scheduled job definitions for the ZEUS trading system.

Schedule (all times ET):
  06:00  — pre-market data refresh (overnight gaps, news)
  08:00  — pre-market summary to Telegram
  09:30  — market open: execute planned trades
  (intraday every 15 min)
  15:30  — closing: force-exit positions needing to close today
  16:00  — EOD reconciliation, daily P&L, Telegram summary
  16:30  — data refresh (today's bars)
  17:00  — feature engineering for today
  18:00  — post-trade attribution + drift checks
  21:00  — next-session signal generation + plan
  22:00  — knowledge artifacts save
  Sunday 02:00 — full model retraining pipeline

Jobs are parameterless and fetch the live `TradingLoop` from
`zeus.scheduler.context.get_loop()` at call time. This keeps APScheduler's
pickled args trivial (`()`) so the `SQLAlchemyJobStore` can persist them
across restarts — see `zeus/scheduler/context.py` for the rationale.
"""
from __future__ import annotations

import structlog

from zeus.scheduler.context import get_loop

log = structlog.get_logger(__name__)


# ─── Job implementations (each is a thin wrapper around TradingLoop methods) ──

def premarket_summary_job() -> None:
    loop = get_loop()
    if loop is None:
        log.warning("job_skipped_no_loop", name="premarket_summary")
        return
    log.info("job_start", name="premarket_summary")
    try:
        loop.run_premarket_summary()
    except Exception as e:
        log.error("job_failed", name="premarket_summary", error=str(e))


def market_open_job() -> None:
    loop = get_loop()
    if loop is None:
        log.warning("job_skipped_no_loop", name="market_open")
        return
    log.info("job_start", name="market_open")
    # Hard halt: never submit orders against stale prices. The 5-08
    # incident was caused by 16 days of silent SIP-embargo failures —
    # this gate makes that visible the *first* day it happens.
    from zeus.data.freshness import assert_fresh_or_halt
    if not assert_fresh_or_halt("market_open"):
        log.error("job_aborted_stale_data", name="market_open")
        return
    try:
        loop.run_manual_trim()
    except Exception as e:
        log.error("job_failed", name="market_open_manual_trim", error=str(e))
    try:
        loop.execute_entries()
    except Exception as e:
        log.error("job_failed", name="market_open", error=str(e))


def intraday_monitor_job() -> None:
    """Single market-hours risk-scan cron — folds in the deterministic
    portion of the previous `overseer_realtime_monitor_job`.

    Why merged: both jobs were reading broker positions + the journal on
    overlapping cadences (5 min and 15 min). One cron, one set of DB
    reads, same coverage. The realtime monitor's only side effects are
    deterministic (decoupled-fill detection → halt issuance, kill-switch
    cascade); folding them in here costs nothing extra and saves a
    duplicate broker round-trip every 5 minutes during market hours.
    """
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="intraday_monitor")
    try:
        loop.check_stops_and_risk()
    except Exception as e:
        log.error("job_failed", name="intraday_monitor", error=str(e))

    # Folded-in agent-side realtime check (decoupled fills + kill-switch
    # cascade). Only runs when the 7-agent system is enabled; the
    # parameterless lookup means the dispatch is cheap when it isn't.
    from zeus.scheduler.context import get_orch
    orch = get_orch()
    if orch is None:
        return
    try:
        orch.bundle.overseer.run_realtime_monitor()
    except Exception as e:
        log.error("job_failed", name="intraday_monitor_overseer_realtime", error=str(e))


def closing_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="closing")
    try:
        loop.run_closing()
    except Exception as e:
        log.error("job_failed", name="closing", error=str(e))


def eod_report_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="eod_report")
    try:
        loop.run_eod_report()
    except Exception as e:
        log.error("job_failed", name="eod_report", error=str(e))


def data_refresh_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="data_refresh")
    try:
        loop.run_data_refresh()
    except Exception as e:
        log.error("job_failed", name="data_refresh", error=str(e))
        return
    # Self-check: if the refresh just ran but the anchor table is still
    # stale, surface it as a CRITICAL risk event. This was the silent
    # failure mode in the 4-22 → 5-07 incident.
    from zeus.data.freshness import check_ohlcv_freshness, write_staleness_risk_event
    report = check_ohlcv_freshness()
    if report.is_stale:
        log.error(
            "data_refresh_left_data_stale",
            latest_date=str(report.latest_date),
            age_trading_days=report.age_trading_days,
            reason=report.reason,
        )
        write_staleness_risk_event(report, action_taken="alert_only")


def feature_engineering_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="feature_engineering")
    # Skip-and-alert if upstream data is stale rather than silently
    # computing features off an old snapshot.
    from zeus.data.freshness import assert_fresh_or_halt
    if not assert_fresh_or_halt("feature_engineering"):
        log.error("job_aborted_stale_data", name="feature_engineering")
        return
    try:
        loop.run_feature_engineering()
    except Exception as e:
        log.error("job_failed", name="feature_engineering", error=str(e))


def next_session_planning_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="next_session_planning")
    try:
        loop.run_next_session_planning()
    except Exception as e:
        log.error("job_failed", name="next_session_planning", error=str(e))


def nightly_retrain_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="nightly_retrain")
    try:
        loop.run_nightly_retrain()
    except Exception as e:
        log.error("job_failed", name="nightly_retrain", error=str(e))


def company_research_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="company_research")
    try:
        loop.run_company_research()
    except Exception as e:
        log.error("job_failed", name="company_research", error=str(e))


def heartbeat_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    try:
        loop.emit_heartbeat()
    except Exception as e:
        log.error("job_failed", name="heartbeat", error=str(e))


# Stale-heartbeat threshold. Heartbeats fire every 5 minutes; missing two
# consecutive emissions means either the trading loop is wedged or the
# whole scheduler process is gone — both warrant a CRITICAL risk event.
_HEARTBEAT_STALENESS_THRESHOLD_MIN = 10


def stale_heartbeat_check_job() -> None:
    """Watch the heartbeats table — if no `trading_loop` row has landed
    in the last 10 minutes, write a CRITICAL `risk_event` so the
    monitor API surfaces it.

    Two-failure window: with heartbeats every 5 min, a 10-min threshold
    means we alert on the *second* missed beat. One miss can be a slow
    broker round-trip; two means something is wrong.

    Idempotency: re-alerting every minute while the heartbeat stays
    stale would spam the events log, so we throttle by checking whether
    the most-recent `system_warning_heartbeat_stale` row already exists
    within the last `_HEARTBEAT_STALENESS_THRESHOLD_MIN` minutes.
    """
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import desc, select
    from zeus.data.storage.database import Heartbeat, RiskEvent, get_session_factory

    SessionLocal = get_session_factory()
    now = datetime.now(timezone.utc)
    threshold = timedelta(minutes=_HEARTBEAT_STALENESS_THRESHOLD_MIN)
    try:
        with SessionLocal() as session:
            latest = session.execute(
                select(Heartbeat)
                .where(Heartbeat.component == "trading_loop")
                .order_by(desc(Heartbeat.ts))
                .limit(1)
            ).scalar_one_or_none()

            if latest is None:
                # First-boot path — no heartbeat yet. Don't alert; the
                # next heartbeat tick will write one.
                return

            latest_ts = latest.ts
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=timezone.utc)
            age = now - latest_ts
            if age <= threshold:
                return

            # Throttle — don't write a new risk event if there's already
            # an open stale-heartbeat alert in the throttle window.
            recent_alert = session.execute(
                select(RiskEvent)
                .where(RiskEvent.event_type == "heartbeat_stale")
                .where(RiskEvent.ts >= now - threshold)
                .limit(1)
            ).scalar_one_or_none()
            if recent_alert is not None:
                return

            session.add(RiskEvent(
                event_type="heartbeat_stale",
                severity="CRITICAL",
                description=(
                    f"trading_loop heartbeat is "
                    f"{int(age.total_seconds() // 60)} min stale "
                    f"(last beat at {latest_ts.isoformat()}); "
                    "scheduler process or trading loop may be wedged"
                ),
                action_taken="alert_only",
            ))
            session.commit()
            log.critical(
                "heartbeat_stale_alert",
                last_beat=latest_ts.isoformat(),
                age_minutes=age.total_seconds() / 60,
            )
    except Exception as e:
        log.error("job_failed", name="stale_heartbeat_check", error=str(e))


def intraday_backfill_job() -> None:
    loop = get_loop()
    if loop is None:
        return
    log.info("job_start", name="intraday_backfill")
    try:
        loop.run_intraday_backfill(lookback_minutes=120)
    except Exception as e:
        log.error("job_failed", name="intraday_backfill", error=str(e))


def backup_db_job() -> None:
    """Nightly pg_dump → ./data/backups (mounted) + optional off-machine
    upload when `BACKUP_REMOTE_CMD` is set in the environment.

    Runs the same `scripts/backup_db.sh` as the manual `docker compose run
    zeus-backup` path, just from inside the scheduler container so the
    backup is visible in `job_executions` like every other scheduled job.
    A failure here writes a CRITICAL log line but does not crash the
    scheduler — the next night's run is the recovery path.
    """
    import os
    import shutil
    import subprocess

    if shutil.which("pg_dump") is None:
        log.error(
            "backup_db_pg_dump_missing",
            hint="install postgresql-client in the image or run docker compose run zeus-backup",
        )
        return

    script_path = "/app/scripts/backup_db.sh"
    if not os.path.exists(script_path):
        log.error("backup_db_script_missing", path=script_path)
        return

    # Local dump dir inside the container — the host volume mount lives at
    # ./data/backups in the project root.
    env = os.environ.copy()
    env.setdefault("BACKUP_DIR", "/app/data/backups")
    env.setdefault("DB_HOST", "postgres")
    env.setdefault("DB_USER", "zeus")
    env.setdefault("DB_NAME", "zeus_db")
    os.makedirs(env["BACKUP_DIR"], exist_ok=True)

    log.info("job_start", name="backup_db")
    try:
        proc = subprocess.run(
            ["bash", script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min hard cap — dumps should be much faster.
        )
    except subprocess.TimeoutExpired:
        log.error("backup_db_timeout")
        return
    if proc.returncode != 0:
        log.error(
            "job_failed",
            name="backup_db",
            returncode=proc.returncode,
            stderr=proc.stderr[-2000:],
        )
        return
    log.info("backup_db_complete", stdout_tail=proc.stdout[-400:])


def reload_models_job() -> None:
    """Poll model_versions every 30s and hot-swap any strategy whose newest
    staging/production row differs from the in-memory loaded version.

    Decouples model promotion from process restarts — Sunday retrain writes
    a row, this job picks it up within 30s, the next predict() uses it. No
    restart, no missed market days while sitting on last week's model.
    """
    loop = get_loop()
    if loop is None:
        return
    try:
        from zeus.scheduler.runner import reload_models_if_promoted
        swaps = reload_models_if_promoted(loop)
        if swaps:
            log.info("models_hot_reloaded", swaps=swaps)
    except Exception as e:
        log.error("job_failed", name="reload_models", error=str(e))


# Manifest of all jobs for the runner to register
JOB_MANIFEST: list[dict] = [
    # cron: minute / hour / day_of_week
    {"func": premarket_summary_job,       "trigger": "cron", "hour": 8,  "minute": 0,  "day_of_week": "mon-fri"},
    {"func": market_open_job,             "trigger": "cron", "hour": 9,  "minute": 30, "second": 0, "day_of_week": "mon-fri"},
    {"func": intraday_monitor_job,        "trigger": "cron", "hour": "9-15", "minute": "*/15", "day_of_week": "mon-fri"},
    {"func": intraday_backfill_job,       "trigger": "cron", "hour": "9-15", "minute": "*/5",  "day_of_week": "mon-fri"},
    {"func": closing_job,                 "trigger": "cron", "hour": 15, "minute": 30, "day_of_week": "mon-fri"},
    {"func": eod_report_job,              "trigger": "cron", "hour": 16, "minute": 15, "day_of_week": "mon-fri"},
    {"func": data_refresh_job,            "trigger": "cron", "hour": 16, "minute": 30, "day_of_week": "mon-fri"},
    {"func": feature_engineering_job,     "trigger": "cron", "hour": 17, "minute": 0,  "day_of_week": "mon-fri"},
    {"func": next_session_planning_job,   "trigger": "cron", "hour": 21, "minute": 0,  "day_of_week": "mon-fri"},
    {"func": company_research_job,        "trigger": "cron", "hour": 21, "minute": 30, "day_of_week": "mon-fri"},
    {"func": nightly_retrain_job,         "trigger": "cron", "hour": 2,  "minute": 0,  "day_of_week": "sun"},
    {"func": backup_db_job,               "trigger": "cron", "hour": 3,  "minute": 0,  "day_of_week": "*"},
    # 5-min heartbeat (was 30 min) so a wedge is detectable within ~10
    # minutes via stale_heartbeat_check_job instead of half an hour.
    {"func": heartbeat_job,               "trigger": "interval", "minutes": 5},
    {"func": stale_heartbeat_check_job,   "trigger": "interval", "minutes": 1},
    {"func": reload_models_job,           "trigger": "interval", "seconds": 30},
]
