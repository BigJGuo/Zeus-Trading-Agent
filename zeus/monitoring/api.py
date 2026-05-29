"""
FastAPI health, metrics, and live dashboard endpoint for ZEUS.
Runs on port 8000.

Endpoints:
  GET /              – live dashboard (static HTML + JS)
  GET /static/*      – dashboard assets
  GET /health        – aggregated health summary (JSONResponse)
  GET /metrics       – pushed metrics store
  GET /ping          – liveness
  GET /api/summary   – header strip data
  GET /api/positions – current open positions
  GET /api/orders/pending
  GET /api/plan/today
  GET /api/fills/recent
  GET /api/risk/events
  GET /api/portfolio/history
  GET /api/health/detail
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, conint
from fastapi.staticfiles import StaticFiles

from zeus.monitoring import dashboard_service

app = FastAPI(title="ZEUS Monitor", version="1.1.0")

_health_monitor = None
_metrics_store: dict = {}

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def set_health_monitor(monitor) -> None:
    global _health_monitor
    _health_monitor = monitor


def update_metrics(metrics: dict) -> None:
    global _metrics_store
    _metrics_store.update(metrics)
    _metrics_store["last_updated"] = datetime.utcnow().isoformat()


@app.get("/")
async def root():
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "dashboard not built"}, status_code=503)
    return FileResponse(
        index,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/health")
async def health():
    if _health_monitor is None:
        return JSONResponse({"healthy": True, "message": "monitor not initialized"})
    summary = _health_monitor.summary()
    status_code = 200 if summary["healthy"] else 503
    return JSONResponse(summary, status_code=status_code)


@app.get("/metrics")
async def metrics():
    return JSONResponse(_metrics_store)


@app.get("/ping")
async def ping():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─── Dashboard JSON API ──────────────────────────────────────────────────────


def _safe(fn, *args, **kwargs):
    try:
        return JSONResponse(fn(*args, **kwargs))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/summary")
def api_summary():
    return _safe(dashboard_service.get_summary)


@app.get("/api/positions")
def api_positions():
    return _safe(dashboard_service.get_positions)


@app.get("/api/orders/pending")
def api_pending_orders():
    return _safe(dashboard_service.get_pending_orders)


@app.get("/api/plan/today")
def api_plan_today():
    plan = dashboard_service.get_today_plan()
    if plan is None:
        # 204 No Content must have an empty body — JSONResponse({}, 204)
        # writes b"{}" while uvicorn forces Content-Length: 0 for 204
        # responses, which then trips its "body longer than Content-Length"
        # guard. Return a bare Response with no body instead.
        return Response(status_code=204)
    return JSONResponse(plan)


@app.get("/api/fills/recent")
def api_recent_fills(n: int = Query(50, ge=1, le=500)):
    return _safe(dashboard_service.tail_fills, n)


@app.get("/api/risk/events")
def api_risk_events(n: int = Query(20, ge=1, le=200)):
    return _safe(dashboard_service.get_risk_events, n)


@app.get("/api/portfolio/history")
def api_portfolio_history(days: int = Query(30, ge=1, le=365)):
    return _safe(dashboard_service.get_portfolio_history, days)


@app.get("/api/scheduler/jobs")
def api_scheduler_jobs():
    return _safe(dashboard_service.get_scheduler_jobs)


@app.get("/api/scheduler/executions")
def api_scheduler_executions(n: int = Query(25, ge=1, le=200)):
    return _safe(dashboard_service.get_recent_job_executions, n)


@app.get("/api/scheduler/health")
def api_scheduler_health():
    return _safe(dashboard_service.get_scheduler_health)


@app.get("/api/scheduler/daily_checklist")
def api_scheduler_daily_checklist():
    return _safe(dashboard_service.get_scheduler_daily_checklist)


@app.get("/api/agents/status")
def api_agents_status():
    return _safe(dashboard_service.get_agent_status_rollup)


@app.get("/api/program/info")
def api_program_info():
    return _safe(dashboard_service.get_program_info)


@app.get("/api/health/detail")
def api_health_detail():
    if _health_monitor is None:
        return JSONResponse({"healthy": True, "checks": [], "message": "monitor not initialized"})
    return JSONResponse(_health_monitor.summary())


# ─── Per-agent endpoints (7-agent trading system) ────────────────────────────


@app.get("/api/agents")
def api_agents(window_days: int = Query(28, ge=1, le=365)):
    """Summary of every known agent: metrics + halt status + last activity."""
    from zeus.agents.journal import KNOWN_AGENTS, TRADER_AGENTS
    from zeus.agents.metrics import compute_research_metrics, compute_trader_metrics

    out = []
    for aid in sorted(KNOWN_AGENTS):
        try:
            if aid in TRADER_AGENTS:
                m = compute_trader_metrics(aid, window_days=window_days)
                out.append({
                    "agent_id": aid,
                    "role": "trader",
                    "n_trades": m.n_trades,
                    "hit_rate": m.hit_rate,
                    "sharpe": m.sharpe_annualized,
                    "max_dd": m.max_drawdown_pct,
                    "net_pnl": m.net_pnl,
                    "mandate_violations": m.mandate_violations,
                })
            elif aid.endswith("_research"):
                r = compute_research_metrics(aid, window_days=window_days)
                out.append({
                    "agent_id": aid,
                    "role": "research",
                    "n_briefs": r.n_briefs,
                    "n_memos": r.n_memos,
                    "n_alerts": r.n_alerts,
                    "avg_confidence": r.avg_confidence,
                    "decoupled_fills": r.trader_decoupled_trades,
                })
            else:
                out.append({"agent_id": aid, "role": "overseer"})
        except Exception as e:
            out.append({"agent_id": aid, "error": str(e)})
    return JSONResponse({"agents": out, "window_days": window_days})


@app.get("/api/agents/{agent_id}/journal")
def api_agent_journal(
    agent_id: str,
    kind: str = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Recent journal entries for one agent."""
    from zeus.agents.journal import KNOWN_AGENTS, query_journal
    if agent_id not in KNOWN_AGENTS:
        return JSONResponse({"error": f"unknown agent_id: {agent_id}"}, status_code=404)
    entries = query_journal(agent_id, kind=kind, limit=limit)
    return JSONResponse({
        "agent_id": agent_id,
        "entries": [
            {
                "id": e.id,
                "ts": e.ts.isoformat() if e.ts else None,
                "kind": e.kind,
                "symbol": e.symbol,
                "title": e.title,
                "body": e.body,
                "confidence": e.confidence,
                "tags": e.tags,
            } for e in entries
        ],
    })


@app.get("/api/agents/{agent_id}/metrics")
def api_agent_metrics(agent_id: str, window_days: int = Query(28, ge=1, le=365)):
    """Full metrics payload for one agent."""
    from dataclasses import asdict
    from zeus.agents.journal import KNOWN_AGENTS, TRADER_AGENTS
    from zeus.agents.metrics import compute_research_metrics, compute_trader_metrics

    if agent_id not in KNOWN_AGENTS:
        return JSONResponse({"error": f"unknown agent_id: {agent_id}"}, status_code=404)
    try:
        if agent_id in TRADER_AGENTS:
            m = compute_trader_metrics(agent_id, window_days=window_days)
        elif agent_id.endswith("_research"):
            m = compute_research_metrics(agent_id, window_days=window_days)
        else:
            return JSONResponse({"agent_id": agent_id, "role": "overseer", "metrics": None})
        return JSONResponse(asdict(m))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Roadmap (improvement-plan tracker) ──────────────────────────────────────


class _RoadmapPatchBody(BaseModel):
    """Partial update for a roadmap task. Every field optional — the
    handler only touches what's set. Validated by FastAPI before the
    service layer runs its dependency-and-status gates."""
    status: Optional[str] = Field(
        default=None,
        description="One of: not_started, in_progress, blocked, complete",
    )
    progress_pct: Optional[conint(ge=0, le=100)] = None  # type: ignore[valid-type]
    blocked_reason: Optional[str] = Field(default=None, max_length=2000)
    notes: Optional[str] = Field(default=None, max_length=10_000)


@app.get("/api/roadmap")
def api_roadmap_list():
    return _safe(dashboard_service.get_roadmap)


@app.get("/api/roadmap/{task_id}")
def api_roadmap_task(task_id: str):
    task = dashboard_service.get_roadmap_task(task_id)
    if task is None:
        return JSONResponse(
            {"error": f"unknown task_id {task_id!r}"}, status_code=404,
        )
    return JSONResponse(task)


@app.patch("/api/roadmap/{task_id}")
def api_roadmap_patch(task_id: str, body: _RoadmapPatchBody = Body(...)):
    try:
        updated = dashboard_service.patch_roadmap_task(
            task_id,
            status=body.status,
            progress_pct=body.progress_pct,
            blocked_reason=body.blocked_reason,
            notes=body.notes,
        )
    except dashboard_service.RoadmapTaskError as e:
        return JSONResponse({"error": str(e)}, status_code=e.http_status)
    return JSONResponse(updated)


def _build_default_health_monitor():
    """Register a minimal set of checks so /api/health/detail is useful."""
    try:
        from zeus.data.storage.database import get_engine
        from zeus.monitoring.health import (
            HealthMonitor,
            make_db_check,
            make_data_freshness_check,
            make_disk_space_check,
        )

        hm = HealthMonitor()
        hm.register(make_db_check(get_engine()))
        from zeus.data.storage.database import get_session_factory
        hm.register(make_data_freshness_check(get_session_factory(), max_age_hours=72.0))
        hm.register(make_disk_space_check(path="/app", min_gb=2.0))
        return hm
    except Exception:
        return None


def run_api(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    if _health_monitor is None:
        hm = _build_default_health_monitor()
        if hm is not None:
            set_health_monitor(hm)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_api()
