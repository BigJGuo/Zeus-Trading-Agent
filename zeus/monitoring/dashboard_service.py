"""Read-only query helpers backing the live dashboard endpoints.

All functions open short-lived SQLAlchemy sessions, return plain dict/list
payloads (JSON-serializable), and degrade gracefully when tables are empty
(dashboard should render "—" rather than error out).
"""
from __future__ import annotations

import csv
import json
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, cast

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from zeus.config.settings import get_settings
from zeus.data.storage.database import (
    AgentDailySpend,
    AgentJournal,
    Heartbeat,
    JobExecution,
    Order,
    Position,
    RiskEvent,
    RoadmapTask,
    SchedulerJob,
    SystemMetrics,
    get_session_factory,
)

_TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}

_BROKER = None


def _get_broker():
    """Lazy-init an AlpacaBroker so dashboard endpoints can pull live data
    every poll. Any failure falls back to DB-only mode silently."""
    global _BROKER
    if _BROKER is not None:
        return _BROKER
    try:
        from zeus.execution.alpaca_broker import AlpacaBroker
        s = get_settings()
        if not s.alpaca_api_key or not s.alpaca_secret_key:
            return None
        _BROKER = AlpacaBroker(
            api_key=s.alpaca_api_key,
            secret_key=s.alpaca_secret_key,
            paper=(s.environment != "live"),
        )
        return _BROKER
    except Exception:
        return None


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _open_session() -> Session:
    return get_session_factory()()


_INITIAL_DEPOSIT_CACHE: dict[str, float] = {}


def _initial_deposit() -> Optional[float]:
    """Return the earliest recorded portfolio value as the inception
    baseline. Cached after the first successful read since it's
    immutable for a given paper account (Alpaca paper starts fresh at
    $100k and stays that way unless the user resets)."""
    if "value" in _INITIAL_DEPOSIT_CACHE:
        return _INITIAL_DEPOSIT_CACHE["value"]
    try:
        session = _open_session()
        try:
            row = session.execute(
                select(SystemMetrics)
                .order_by(SystemMetrics.ts.asc())
                .limit(1)
            ).scalar_one_or_none()
        finally:
            session.close()
        if row is None or row.portfolio_value is None:
            return None
        v = float(row.portfolio_value)
        _INITIAL_DEPOSIT_CACHE["value"] = v
        return v
    except Exception:
        return None


def get_summary() -> dict[str, Any]:
    s = get_settings()
    payload: dict[str, Any] = {
        "environment": s.environment,
        "portfolio_value": None,
        "cash_balance": None,
        "unrealized_pnl": None,
        "all_time_pnl_usd": None,
        "all_time_pnl_pct": None,
        "initial_deposit": None,
        "realized_pnl_daily": None,
        "drawdown_usd": None,
        "drawdown_pct": None,
        "open_position_count": None,
        "regime": None,
        "model_ic_rolling20d": None,
        "metrics_ts": None,
        "heartbeat_ts": None,
        "heartbeat_age_s": None,
        "heartbeat_status": None,
        "kill_switch_active": None,
        "live": False,
    }

    # Live path — always prefer fresh Alpaca numbers so dashboard reflects
    # real-time portfolio state, not the last intraday_monitor snapshot.
    broker = _get_broker()
    if broker is not None:
        try:
            acct = broker.get_account()
            positions = broker.get_positions()
            pv = float(acct.portfolio_value or 0)
            cash = float(acct.cash or 0)
            unrealized = sum(float(getattr(p, "unrealized_pl", 0.0)) for p in positions)
            payload.update(
                portfolio_value=pv,
                cash_balance=cash,
                unrealized_pnl=unrealized,
                open_position_count=len(positions),
                metrics_ts=datetime.now(timezone.utc).isoformat(),
                live=True,
            )
        except Exception:
            pass  # fall through to DB-derived values below

    session = _open_session()
    try:
        m = session.execute(
            select(SystemMetrics).order_by(desc(SystemMetrics.ts)).limit(1)
        ).scalar_one_or_none()
        if m is not None:
            if payload["portfolio_value"] is None:
                payload["portfolio_value"] = m.portfolio_value
            if payload["cash_balance"] is None:
                payload["cash_balance"] = m.cash_balance
            if payload["unrealized_pnl"] is None:
                payload["unrealized_pnl"] = m.unrealized_pnl
            payload["realized_pnl_daily"] = m.realized_pnl_daily
            # Drawdown values still come from the tracker (peak is DB-tracked).
            payload["drawdown_usd"] = m.drawdown_from_peak_usd
            payload["drawdown_pct"] = m.drawdown_from_peak_pct
            if payload["open_position_count"] is None:
                payload["open_position_count"] = m.open_position_count
            payload["regime"] = m.regime
            payload["model_ic_rolling20d"] = m.model_ic_rolling20d
            if not payload["live"]:
                payload["metrics_ts"] = _iso(m.ts)
        hb = session.execute(
            select(Heartbeat).order_by(desc(Heartbeat.ts)).limit(1)
        ).scalar_one_or_none()
        if hb is not None:
            ts = hb.ts if hb.ts.tzinfo else hb.ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            payload.update(
                heartbeat_ts=_iso(hb.ts),
                heartbeat_age_s=round(age, 1),
                heartbeat_status=hb.status,
            )
            details = hb.details or {}
            if "kill_switch_active" in details:
                payload["kill_switch_active"] = bool(details["kill_switch_active"])
            if payload["regime"] is None and "regime" in details:
                payload["regime"] = details["regime"]
        if payload["open_position_count"] is None:
            count = session.execute(
                select(Position).where(Position.environment == s.environment)
            ).all()
            payload["open_position_count"] = len(count)
    finally:
        session.close()

    plan = get_today_plan()
    if plan is not None:
        payload["model_version"] = plan.get("model_version")
        if payload["regime"] is None:
            payload["regime"] = plan.get("regime")

    # All-time P&L: portfolio_value − initial deposit. Reads the earliest
    # SystemMetrics.portfolio_value row as the inception baseline. Replaces
    # the misleading "Unrealized" tile that only shows open-position drift
    # and ignores realized losses on closed trades.
    init_dep = _initial_deposit()
    pv = payload.get("portfolio_value")
    if init_dep is not None and pv is not None and init_dep > 0:
        payload["initial_deposit"] = init_dep
        payload["all_time_pnl_usd"] = float(pv) - init_dep
        payload["all_time_pnl_pct"] = (float(pv) - init_dep) / init_dep
    return payload


def get_positions() -> list[dict[str, Any]]:
    # Live-first: pull live positions from Alpaca (includes mark-to-market
    # current_price) so PnL reflects real-time quotes.
    broker = _get_broker()
    if broker is not None:
        try:
            live = broker.get_positions()
            stop_map: dict[str, dict[str, Any]] = {}
            session = _open_session()
            try:
                for row in session.execute(select(Position)).scalars().all():
                    stop_map[row.symbol] = {
                        "hard_stop": row.hard_stop,
                        "peak_price": row.peak_price,
                    }
            finally:
                session.close()
            out = []
            for p in live:
                sym = p.symbol
                qty = int(p.qty)
                avg_entry = float(p.avg_entry_price)
                cur = float(p.current_price) if p.current_price is not None else None
                mv = float(getattr(p, "market_value", qty * (cur or avg_entry)))
                upnl = float(getattr(p, "unrealized_pl", 0.0))
                upct = float(getattr(p, "unrealized_plpc", 0.0))
                stops = stop_map.get(sym, {})
                out.append({
                    "symbol": sym,
                    "qty": qty,
                    "avg_entry_price": avg_entry,
                    "current_price": cur,
                    "market_value": mv,
                    "unrealized_pnl": upnl,
                    "unrealized_pnl_pct": upct,
                    "hard_stop": stops.get("hard_stop"),
                    "peak_price": stops.get("peak_price"),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            out.sort(key=lambda r: r["market_value"] or 0, reverse=True)
            return out
        except Exception:
            pass  # fall through to DB

    session = _open_session()
    try:
        rows = session.execute(
            select(Position).order_by(desc(Position.market_value))
        ).scalars().all()
        return [
            {
                "symbol": r.symbol,
                "qty": r.qty,
                "avg_entry_price": r.avg_entry_price,
                "current_price": r.current_price,
                "market_value": r.market_value,
                "unrealized_pnl": r.unrealized_pnl,
                "unrealized_pnl_pct": r.unrealized_pnl_pct,
                "hard_stop": r.hard_stop,
                "peak_price": r.peak_price,
                "updated_at": _iso(r.updated_at),
            }
            for r in rows
        ]
    finally:
        session.close()


def get_pending_orders() -> list[dict[str, Any]]:
    # Live-first: pull open orders straight from Alpaca every poll so
    # fills/cancellations are visible without waiting for the 15-min tick.
    broker = _get_broker()
    if broker is not None:
        try:
            live = broker.get_open_orders()
            out = []
            for o in live:
                out.append({
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": str(o.side).lower().split(".")[-1],
                    "order_type": str(o.order_type).lower().split(".")[-1],
                    "qty": int(float(o.qty or 0)),
                    "filled_qty": int(o.filled_qty) if o.filled_qty else 0,
                    "limit_price": float(o.limit_price) if o.limit_price else None,
                    "status": str(o.status).lower().split(".")[-1],
                    "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                })
            out.sort(key=lambda r: r["submitted_at"] or "", reverse=True)
            return out
        except Exception:
            pass  # fall through to DB

    session = _open_session()
    try:
        rows = session.execute(
            select(Order).order_by(desc(Order.submitted_at))
        ).scalars().all()
        return [
            {
                "id": r.id,
                "symbol": r.symbol,
                "side": r.side,
                "order_type": r.order_type,
                "qty": r.qty,
                "filled_qty": r.filled_qty or 0,
                "limit_price": r.limit_price,
                "status": r.status,
                "submitted_at": _iso(r.submitted_at),
            }
            for r in rows
            if (r.status or "").lower() not in _TERMINAL_STATUSES
        ]
    finally:
        session.close()


def get_today_plan() -> Optional[dict[str, Any]]:
    """Merge today's per-strategy plans into a single payload for the
    dashboard. The multi-strategy migration writes
    `session_plans/{date}_{strategy_id}.json` (one file per strategy);
    the legacy single-file `{date}.json` isn't produced anymore, so we
    glob per-strategy files first and only fall back to the legacy file
    for old runs.
    """
    s = get_settings()
    today_iso = date.today().isoformat()
    base = Path(s.artifacts_path) / "knowledge" / "session_plans"
    per_strategy = sorted(base.glob(f"{today_iso}_*.json"))
    plans: list[dict[str, Any]] = []
    for p in per_strategy:
        try:
            with p.open() as f:
                d = json.load(f)
            # Filename suffix is the source of truth for strategy_id —
            # the JSON body sometimes omits it at the top level.
            sid = p.stem[len(today_iso) + 1 :]
            d.setdefault("strategy_id", sid)
            plans.append(d)
        except Exception:
            continue

    if not plans:
        legacy = base / f"{today_iso}.json"
        if not legacy.exists():
            return None
        try:
            with legacy.open() as f:
                return json.load(f)
        except Exception:
            return None

    merged_entries: list[dict[str, Any]] = []
    merged_exits: list[str] = []
    merged_holds: list[Any] = []
    regime: Optional[str] = None
    model_version: Optional[str] = None
    for d in plans:
        sid = d.get("strategy_id", "?")
        for e in d.get("entries", []) or []:
            row = dict(e)
            row.setdefault("strategy_id", sid)
            merged_entries.append(row)
        for x in d.get("exits", []) or []:
            # Exits are usually plain symbol strings; tag with strategy
            # so the UI can show provenance without breaking string-based
            # consumers (the dashboard JS joins on ", ").
            sym = x if isinstance(x, str) else x.get("symbol")
            if sym:
                merged_exits.append(f"{sym} ({sid})")
        for h in d.get("holds", []) or []:
            merged_holds.append(h)
        if regime is None:
            regime = d.get("regime")
        if model_version is None:
            model_version = d.get("model_version")

    return {
        "plan_date": today_iso,
        "regime": regime,
        "model_version": model_version,
        "entries": merged_entries,
        "exits": merged_exits,
        "holds": merged_holds,
        "strategies": [d.get("strategy_id", "?") for d in plans],
    }


def tail_fills(n: int = 50) -> list[dict[str, Any]]:
    """Return the most-recent n fills with FIFO-matched realized P&L.

    The CSV only stores raw broker fills (no P&L column), so we walk the
    full file chronologically per symbol, maintaining open buy lots in
    a FIFO queue. When a sell crosses, we pop matched shares from the
    queue and compute pnl = (sell_price - lot_price) * matched_shares.

    Caveats: this is symbol-level FIFO across ALL strategies (the CSV
    doesn't carry strategy_id), so if two strategies hold the same
    symbol simultaneously, lots are blended. Good enough for the
    dashboard's visual P&L column; for true per-strategy realized P&L,
    consult the trades table.
    """
    s = get_settings()
    path = Path(s.logs_path) / "trading_log.csv"
    if not path.exists():
        return []
    all_rows: list[dict[str, Any]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(dict(row))

    open_lots: dict[str, deque[tuple[float, float]]] = {}
    for row in all_rows:
        try:
            shares = float(row.get("shares") or 0)
            price = float(row.get("fill_price") or 0)
        except (TypeError, ValueError):
            row["pnl"] = None
            continue
        symbol = row.get("symbol") or ""
        side = (row.get("side") or "").lower()
        if shares <= 0 or price <= 0 or not symbol:
            row["pnl"] = None
            continue

        lots = open_lots.setdefault(symbol, deque())
        if side == "buy":
            lots.append((shares, price))
            row["pnl"] = None
        elif side == "sell":
            remaining = shares
            realized = 0.0
            matched_any = False
            while remaining > 0 and lots:
                lot_shares, lot_price = lots[0]
                take = min(remaining, lot_shares)
                realized += (price - lot_price) * take
                remaining -= take
                if take >= lot_shares:
                    lots.popleft()
                else:
                    lots[0] = (lot_shares - take, lot_price)
                matched_any = True
            row["pnl"] = round(realized, 2) if matched_any else None
        else:
            row["pnl"] = None

    tail = all_rows[-max(1, n):]
    tail.reverse()
    return tail


def get_risk_events(n: int = 20) -> list[dict[str, Any]]:
    session = _open_session()
    try:
        rows = session.execute(
            select(RiskEvent).order_by(desc(RiskEvent.ts)).limit(n)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "ts": _iso(r.ts),
                "event_type": r.event_type,
                "severity": r.severity,
                "description": r.description,
                "drawdown_usd": r.drawdown_usd,
                "drawdown_pct": r.drawdown_pct,
                "action_taken": r.action_taken,
            }
            for r in rows
        ]
    finally:
        session.close()


_AGENT_IDS = [
    "day", "swing", "long_term",
    "day_research", "swing_research", "long_term_research",
    "overseer",
]

_AGENT_ROLES = {
    "day": "trader", "swing": "trader", "long_term": "trader",
    "day_research": "research", "swing_research": "research",
    "long_term_research": "research",
    "overseer": "overseer",
}

_AGENT_MODEL_TIER = {
    "day_research": "haiku",
    "swing_research": "sonnet",
    "long_term_research": "opus",
    "day": "sonnet",
    "swing": "sonnet",
    "long_term": "sonnet",
    "overseer": "opus",
}


def get_scheduler_jobs() -> list[dict[str, Any]]:
    """All registered APScheduler jobs with next-run/last-run state.

    Rows come from `scheduler_jobs` (upserted by the telemetry listener).
    Dashboard uses `seconds_until_next` to render a live countdown; when
    that value is negative or null the job is either currently running
    or the listener hasn't fired yet.
    """
    now = datetime.now(timezone.utc)
    session = _open_session()
    try:
        rows = session.execute(
            select(SchedulerJob).order_by(SchedulerJob.next_run_time.asc().nullslast())
        ).scalars().all()
        out = []
        for r in rows:
            nxt = r.next_run_time
            if nxt is not None and nxt.tzinfo is None:
                nxt = nxt.replace(tzinfo=timezone.utc)
            secs = None
            if nxt is not None:
                secs = int((nxt - now).total_seconds())
            out.append({
                "job_id": r.job_id,
                "func_name": r.func_name,
                "trigger": r.trigger_repr,
                "next_run_time": _iso(r.next_run_time),
                "seconds_until_next": secs,
                "last_run_time": _iso(r.last_run_time),
                "last_duration_ms": r.last_duration_ms,
                "last_success": r.last_success,
                "last_error": r.last_error,
                "updated_at": _iso(r.updated_at),
            })
        return out
    finally:
        session.close()


def get_scheduler_daily_checklist() -> list[dict[str, Any]]:
    """Today's-schedule view: one row per registered job with a state chip.

    For each SchedulerJob we look at:
      - JobExecution rows with started_ts in today's CT window,
      - the job's SchedulerJob.next_run_time.

    Buckets (priority order — first matching wins):
      error      — today saw at least one execution with status='error'
      running    — most recent today has status='submitted' + null finished_ts
      done       — ≥1 successful run today AND next_run_time is not today
      recurring  — ≥1 successful run today AND next_run_time is later today
                   (cron */N jobs like intraday_backfill)
      pending    — no runs today AND next_run_time is later today
      not_today  — no runs today AND next_run_time is tomorrow-or-later/null

    The list is sorted: errors first, running, recurring/pending by next_run,
    done by last_run_time, not_today last.
    """
    from zoneinfo import ZoneInfo

    ct = ZoneInfo("America/Chicago")
    now_utc = datetime.now(timezone.utc)
    now_ct = now_utc.astimezone(ct)
    today_ct = now_ct.date()
    today_start_ct = datetime.combine(today_ct, datetime.min.time(), tzinfo=ct)
    today_end_ct = today_start_ct + timedelta(days=1)
    today_start_utc = today_start_ct.astimezone(timezone.utc)
    today_end_utc = today_end_ct.astimezone(timezone.utc)

    def _as_utc(ts: Optional[datetime]) -> Optional[datetime]:
        if ts is None:
            return None
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)

    def _ct_label(ts: Optional[datetime]) -> Optional[str]:
        if ts is None:
            return None
        ts_utc = _as_utc(ts)
        assert ts_utc is not None  # `_as_utc` only returns None when ts is None
        ts_ct = ts_utc.astimezone(ct)
        hm = ts_ct.strftime('%I:%M %p').lstrip('0')
        if ts_ct.date() == today_ct:
            return f"today {hm} CT"
        if ts_ct.date() == today_ct + timedelta(days=1):
            return f"tomorrow {hm} CT"
        return f"{ts_ct.strftime('%a')} {hm} CT"

    session = _open_session()
    try:
        jobs = session.execute(select(SchedulerJob)).scalars().all()
        todays = session.execute(
            select(JobExecution)
            .where(JobExecution.started_ts >= today_start_utc)
            .where(JobExecution.started_ts < today_end_utc)
            .order_by(JobExecution.started_ts)
        ).scalars().all()
        by_job: dict[str, list[JobExecution]] = {}
        for e in todays:
            by_job.setdefault(e.job_id, []).append(e)

        rows: list[dict[str, Any]] = []
        for j in jobs:
            execs = by_job.get(j.job_id, [])
            runs_total = len(execs)
            runs_ok = sum(1 for e in execs if e.status == "success")
            runs_err = sum(1 for e in execs if e.status == "error")
            runs_missed = sum(1 for e in execs if e.status == "missed")
            last_exec = execs[-1] if execs else None
            last_err_exec = next(
                (e for e in reversed(execs) if e.status == "error"), None
            )

            next_run = _as_utc(j.next_run_time)
            next_run_today = (
                next_run is not None
                and today_start_utc <= next_run < today_end_utc
                and next_run >= now_utc
            )

            is_running = (
                last_exec is not None
                and last_exec.status == "submitted"
                and last_exec.finished_ts is None
            )

            if runs_err > 0:
                status = "error"
            elif is_running:
                status = "running"
            elif runs_ok > 0 and next_run_today:
                status = "recurring"
            elif runs_ok > 0:
                status = "done"
            elif next_run_today:
                status = "pending"
            else:
                status = "not_today"

            last_run_utc = _as_utc(j.last_run_time)
            last_run_today = (
                last_run_utc is not None
                and today_start_utc <= last_run_utc < today_end_utc
            )

            rows.append({
                "job_id": j.job_id,
                "func_name": j.func_name,
                "trigger": j.trigger_repr,
                "status": status,
                "runs_today": runs_total,
                "runs_ok_today": runs_ok,
                "runs_err_today": runs_err,
                "runs_missed_today": runs_missed,
                "next_run_time": _iso(j.next_run_time),
                "next_run_label": _ct_label(j.next_run_time) if next_run else None,
                "next_run_today": next_run_today,
                "seconds_until_next": (
                    int((next_run - now_utc).total_seconds()) if next_run else None
                ),
                "last_run_time": _iso(j.last_run_time),
                "last_run_today": last_run_today,
                "last_run_label": _ct_label(j.last_run_time),
                "last_duration_ms": j.last_duration_ms,
                "last_success": j.last_success,
                "last_error": (
                    (last_err_exec.error if last_err_exec else j.last_error) or None
                ),
            })

        # Sort: errors → running → pending/recurring by next_run → done by
        # last_run → not_today at the bottom.
        status_rank = {
            "error": 0, "running": 1, "pending": 2,
            "recurring": 3, "done": 4, "not_today": 5,
        }

        def _sort_key(r: dict[str, Any]) -> tuple:
            secs = r.get("seconds_until_next")
            # None ranks after any numeric within same status bucket.
            return (
                status_rank.get(r["status"], 9),
                secs if secs is not None else 10**9,
                r.get("last_run_time") or "",
                r["job_id"],
            )

        rows.sort(key=_sort_key)
        return rows
    finally:
        session.close()


def get_recent_job_executions(n: int = 25) -> list[dict[str, Any]]:
    """Most recent N executions across all jobs — the activity feed."""
    session = _open_session()
    try:
        rows = session.execute(
            select(JobExecution).order_by(desc(JobExecution.started_ts)).limit(n)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "job_id": r.job_id,
                "func_name": r.func_name,
                "started_ts": _iso(r.started_ts),
                "finished_ts": _iso(r.finished_ts),
                "duration_ms": r.duration_ms,
                "status": r.status,
                "error": r.error,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_scheduler_health() -> dict[str, Any]:
    """Aggregate health for the scheduler panel's top banner."""
    now = datetime.now(timezone.utc)
    session = _open_session()
    try:
        jobs = session.execute(select(SchedulerJob)).scalars().all()
        job_count = len(jobs)
        upcoming = [j for j in jobs if j.next_run_time is not None]
        # `next_run_time` is filtered non-None above; cast satisfies the sort key.
        upcoming.sort(key=lambda j: cast(datetime, j.next_run_time))
        next_job = upcoming[0] if upcoming else None
        next_ts = next_job.next_run_time if next_job else None
        if next_ts is not None and next_ts.tzinfo is None:
            next_ts = next_ts.replace(tzinfo=timezone.utc)

        # Stats over last 24h: success vs error count, last error.
        since = now - timedelta(hours=24)
        execs = session.execute(
            select(JobExecution)
            .where(JobExecution.started_ts >= since)
            .order_by(desc(JobExecution.started_ts))
        ).scalars().all()
        succ = sum(1 for e in execs if e.status == "success")
        err = sum(1 for e in execs if e.status == "error")
        missed = sum(1 for e in execs if e.status == "missed")
        last_err = next((e for e in execs if e.status == "error"), None)
        return {
            "now": now.isoformat(),
            "job_count": job_count,
            "next_job_id": next_job.job_id if next_job else None,
            "next_func": next_job.func_name if next_job else None,
            "next_run_time": _iso(next_job.next_run_time) if next_job else None,
            "seconds_until_next": (
                int((next_ts - now).total_seconds()) if next_ts else None
            ),
            "successes_24h": succ,
            "errors_24h": err,
            "missed_24h": missed,
            "last_error_func": last_err.func_name if last_err else None,
            "last_error_msg": (last_err.error or "")[:240] if last_err else None,
            "last_error_ts": _iso(last_err.started_ts) if last_err else None,
        }
    finally:
        session.close()


def get_agent_status_rollup() -> list[dict[str, Any]]:
    """One tile per agent: role, model tier, last journal entry, journal
    counts today, today's LLM spend."""
    today = date.today()
    today_start = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
    session = _open_session()
    try:
        spend_rows = session.execute(
            select(AgentDailySpend).where(AgentDailySpend.date == today)
        ).scalars().all()
        spend_map = {r.agent_id: r for r in spend_rows}

        out = []
        for agent_id in _AGENT_IDS:
            last = session.execute(
                select(AgentJournal)
                .where(AgentJournal.agent_id == agent_id)
                .order_by(desc(AgentJournal.ts))
                .limit(1)
            ).scalar_one_or_none()
            today_count = session.execute(
                select(AgentJournal)
                .where(AgentJournal.agent_id == agent_id)
                .where(AgentJournal.ts >= today_start)
            ).all()
            spend = spend_map.get(agent_id)
            last_ts = last.ts if last else None
            if last_ts is not None and last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            age_s = None
            if last_ts is not None:
                age_s = int((datetime.now(timezone.utc) - last_ts).total_seconds())
            out.append({
                "agent_id": agent_id,
                "role": _AGENT_ROLES.get(agent_id),
                "model_tier": _AGENT_MODEL_TIER.get(agent_id),
                "last_entry_ts": _iso(last.ts) if last else None,
                "last_entry_age_s": age_s,
                "last_entry_kind": last.kind if last else None,
                "last_entry_title": last.title if last else None,
                "entries_today": len(today_count),
                "calls_today": spend.calls if spend else 0,
                "input_tokens_today": int(spend.input_tokens) if spend else 0,
                "output_tokens_today": int(spend.output_tokens) if spend else 0,
                "cost_usd_today": float(spend.cost_usd) if spend else 0.0,
            })
        return out
    finally:
        session.close()


def get_program_info() -> dict[str, Any]:
    """Static/semi-static program context for the dashboard header:
    environment, process uptime, git sha (best-effort), python version."""
    import os
    import platform
    import sys
    s = get_settings()
    uptime_s = None
    try:
        # /proc/1/stat isn't reliable across distros; use /proc/self/stat
        # for the process's start-time ticks, then divide by clock tick.
        with open("/proc/self/stat") as f:
            fields = f.read().split()
        start_ticks = int(fields[21])
        # os.sysconf is POSIX-only; guarded by the outer try (raises on Windows).
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])  # type: ignore[attr-defined]
        with open("/proc/uptime") as f:
            sys_uptime = float(f.read().split()[0])
        proc_uptime = sys_uptime - (start_ticks / hz)
        uptime_s = int(proc_uptime)
    except Exception:
        pass

    git_sha = None
    for env_key in ("GIT_SHA", "GIT_COMMIT", "SOURCE_COMMIT"):
        if os.environ.get(env_key):
            git_sha = os.environ[env_key][:12]
            break

    return {
        "environment": s.environment,
        "now": datetime.now(timezone.utc).isoformat(),
        "uptime_s": uptime_s,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "hostname": platform.node(),
        "git_sha": git_sha,
    }


def get_portfolio_history(days: int = 30) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    session = _open_session()
    try:
        rows = session.execute(
            select(SystemMetrics)
            .where(SystemMetrics.ts >= since)
            .order_by(SystemMetrics.ts)
        ).scalars().all()
        return [
            {
                "ts": _iso(r.ts),
                "portfolio_value": r.portfolio_value,
                "drawdown_pct": r.drawdown_from_peak_pct,
                "open_position_count": r.open_position_count,
            }
            for r in rows
        ]
    finally:
        session.close()


# ─── Roadmap (improvement-plan tracker) ──────────────────────────────────────


_VALID_ROADMAP_STATUSES = ("not_started", "in_progress", "blocked", "complete")
_VALID_ROADMAP_CATEGORIES = (
    "infrastructure", "learning", "decision", "training", "backtest", "safety",
)


def _normalize_array(value: Any) -> list[str]:
    """Postgres ARRAY round-trips as list, SQLite JSON variant as list-or-None
    or stringified JSON. Coerce so the API always returns a list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _serialize_task(row: RoadmapTask) -> dict[str, Any]:
    return {
        "id": row.id,
        "week": row.week,
        "title": row.title,
        "category": row.category,
        "description": row.description,
        "acceptance_criteria": row.acceptance_criteria,
        "dependencies": _normalize_array(row.dependencies),
        "deliverables": _normalize_array(row.deliverables),
        "status": row.status,
        "progress_pct": int(row.progress_pct or 0),
        "blocked_reason": row.blocked_reason,
        "notes": row.notes,
        "started_at": _iso(row.started_at),
        "completed_at": _iso(row.completed_at),
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def get_roadmap() -> dict[str, Any]:
    """Return all roadmap tasks plus a summary for the dashboard tab."""
    session = _open_session()
    try:
        rows = session.execute(
            select(RoadmapTask).order_by(RoadmapTask.week.asc(), RoadmapTask.id.asc())
        ).scalars().all()
        tasks = [_serialize_task(r) for r in rows]
    finally:
        session.close()

    total = len(tasks)
    by_status = {s: 0 for s in _VALID_ROADMAP_STATUSES}
    for t in tasks:
        by_status[t["status"]] = by_status.get(t["status"], 0) + 1

    weeks_summary: dict[int, dict[str, int]] = {}
    for t in tasks:
        w = weeks_summary.setdefault(t["week"], {"total": 0, "complete": 0})
        w["total"] += 1
        if t["status"] == "complete":
            w["complete"] += 1

    weeks_list = [
        {"week": w, "total": d["total"], "complete": d["complete"]}
        for w, d in sorted(weeks_summary.items())
    ]

    return {
        "tasks": tasks,
        "summary": {
            "total": total,
            "complete": by_status["complete"],
            "in_progress": by_status["in_progress"],
            "blocked": by_status["blocked"],
            "not_started": by_status["not_started"],
            "weeks_summary": weeks_list,
        },
    }


def get_roadmap_task(task_id: str) -> Optional[dict[str, Any]]:
    session = _open_session()
    try:
        row = session.execute(
            select(RoadmapTask).where(RoadmapTask.id == task_id)
        ).scalar_one_or_none()
        return _serialize_task(row) if row is not None else None
    finally:
        session.close()


class RoadmapTaskError(Exception):
    """Patch-validation failure: bad status, missing blocked_reason,
    incomplete dependencies, or unknown task id. Caller (API layer)
    converts to a 400/404 response."""

    def __init__(self, message: str, *, http_status: int = 400):
        super().__init__(message)
        self.http_status = http_status


def patch_roadmap_task(
    task_id: str,
    *,
    status: Optional[str] = None,
    progress_pct: Optional[int] = None,
    blocked_reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Apply a partial update. Stamps `started_at` on the first transition
    out of `not_started`; stamps `completed_at` on transition to `complete`.

    Enforces:
      - status must be one of `_VALID_ROADMAP_STATUSES`
      - moving to `blocked` requires a non-empty `blocked_reason`
      - moving out of `not_started` requires every id in `dependencies`
        to already be `complete` (the dependency gate)
      - progress_pct ∈ [0, 100]
    """
    if status is not None and status not in _VALID_ROADMAP_STATUSES:
        raise RoadmapTaskError(
            f"invalid status {status!r}; expected one of {list(_VALID_ROADMAP_STATUSES)}"
        )
    if progress_pct is not None and not (0 <= progress_pct <= 100):
        raise RoadmapTaskError("progress_pct must be in [0, 100]")

    session = _open_session()
    try:
        row = session.execute(
            select(RoadmapTask).where(RoadmapTask.id == task_id)
        ).scalar_one_or_none()
        if row is None:
            raise RoadmapTaskError(f"unknown task_id {task_id!r}", http_status=404)

        # Dependency gate — only when transitioning AWAY from not_started.
        if (
            status is not None
            and status != "not_started"
            and row.status == "not_started"
        ):
            deps = _normalize_array(row.dependencies)
            if deps:
                missing = session.execute(
                    select(RoadmapTask.id).where(
                        RoadmapTask.id.in_(deps),
                        RoadmapTask.status != "complete",
                    )
                ).scalars().all()
                if missing:
                    raise RoadmapTaskError(
                        f"cannot leave not_started: dependencies still open: {sorted(missing)}"
                    )

        if status == "blocked" and not (blocked_reason and blocked_reason.strip()):
            raise RoadmapTaskError("status=blocked requires a non-empty blocked_reason")

        now = datetime.now(timezone.utc)

        if status is not None:
            # First transition out of not_started → stamp started_at.
            if row.status == "not_started" and status != "not_started" and row.started_at is None:
                row.started_at = now
            # Transition into complete → stamp completed_at + progress=100.
            if status == "complete":
                row.completed_at = now
                if progress_pct is None:
                    row.progress_pct = 100
            # If we re-open a completed task, clear completed_at.
            if status != "complete" and row.completed_at is not None:
                row.completed_at = None
            row.status = status
            # Clear blocked_reason if leaving the blocked state.
            if status != "blocked":
                row.blocked_reason = None

        if blocked_reason is not None and (status == "blocked" or row.status == "blocked"):
            row.blocked_reason = blocked_reason

        if progress_pct is not None:
            row.progress_pct = progress_pct

        if notes is not None:
            row.notes = notes

        session.commit()
        session.refresh(row)
        return _serialize_task(row)
    finally:
        session.close()


def upsert_roadmap_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Idempotent upsert used by `scripts/seed_roadmap.py`. Inserts on
    first run, updates the immutable-ish content fields on re-run, but
    leaves agent-mutable state (`status`, `progress_pct`,
    `blocked_reason`, `notes`, `started_at`, `completed_at`) untouched
    on rows that already exist — re-running the seed mustn't reset
    progress."""
    required = ("id", "week", "title", "category", "description", "acceptance_criteria")
    for k in required:
        if not payload.get(k):
            raise RoadmapTaskError(f"seed payload missing required field {k!r}")
    if payload["category"] not in _VALID_ROADMAP_CATEGORIES:
        raise RoadmapTaskError(
            f"invalid category {payload['category']!r}; expected one of "
            f"{list(_VALID_ROADMAP_CATEGORIES)}"
        )

    session = _open_session()
    try:
        row = session.execute(
            select(RoadmapTask).where(RoadmapTask.id == payload["id"])
        ).scalar_one_or_none()
        if row is None:
            row = RoadmapTask(
                id=payload["id"],
                week=int(payload["week"]),
                title=payload["title"],
                category=payload["category"],
                description=payload["description"],
                acceptance_criteria=payload["acceptance_criteria"],
                dependencies=list(payload.get("dependencies") or []),
                deliverables=list(payload.get("deliverables") or []),
            )
            session.add(row)
        else:
            # Refresh content fields only — preserve agent's progress state.
            row.week = int(payload["week"])
            row.title = payload["title"]
            row.category = payload["category"]
            row.description = payload["description"]
            row.acceptance_criteria = payload["acceptance_criteria"]
            row.dependencies = list(payload.get("dependencies") or [])
            row.deliverables = list(payload.get("deliverables") or [])
        session.commit()
        session.refresh(row)
        return _serialize_task(row)
    finally:
        session.close()
