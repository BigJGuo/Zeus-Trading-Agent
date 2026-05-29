"""Integration tests for the Roadmap-tab API.

Exercises the service layer (upsert + patch + dependency gate) and the
FastAPI endpoints (list, get-one, PATCH) end-to-end against the SQLite
in-memory fixture. Status-transition gates and dependency validation
are the critical bits — they're what keep an agent from marking a
Week 3 task complete while Week 1 tasks are still in progress.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from zeus.data.storage.database import RoadmapTask


@pytest.fixture()
def patched_db(sqlite_engine, patched_session_factory):
    """Create the RoadmapTask table on the in-memory SQLite engine and
    ensure every module-level `get_session_factory` import resolves to
    the patched in-memory session factory."""
    RoadmapTask.__table__.create(bind=sqlite_engine)
    return patched_session_factory


@pytest.fixture()
def client(patched_db):
    from zeus.monitoring.api import app
    return TestClient(app)


def _seed(client, **kwargs):
    """Helper: invoke the service-level upsert directly (the API only
    exposes PATCH on existing rows, since rows are created by the seed
    script)."""
    from zeus.monitoring.dashboard_service import upsert_roadmap_task
    base = {
        "id": "test-task",
        "week": 1,
        "title": "Test",
        "category": "infrastructure",
        "description": "desc",
        "acceptance_criteria": "ac",
        "dependencies": [],
        "deliverables": [],
    }
    base.update(kwargs)
    return upsert_roadmap_task(base)


# ─── Listing + retrieval ─────────────────────────────────────────────────────


def test_empty_roadmap_returns_zero_summary(client):
    resp = client.get("/api/roadmap")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tasks"] == []
    assert body["summary"]["total"] == 0
    assert body["summary"]["weeks_summary"] == []


def test_seeded_tasks_are_grouped_by_week(client):
    _seed(client, id="w1-a", week=1)
    _seed(client, id="w1-b", week=1)
    _seed(client, id="w2-a", week=2)

    body = client.get("/api/roadmap").json()
    assert body["summary"]["total"] == 3
    weeks = {w["week"]: w for w in body["summary"]["weeks_summary"]}
    assert weeks[1]["total"] == 2
    assert weeks[2]["total"] == 1


def test_get_single_task_404_on_unknown(client):
    resp = client.get("/api/roadmap/does-not-exist")
    assert resp.status_code == 404
    assert "unknown" in resp.json()["error"].lower()


def test_get_single_task_round_trips(client):
    _seed(client, id="task-one", title="Custom title",
          description="What this does", acceptance_criteria="When it's done")
    body = client.get("/api/roadmap/task-one").json()
    assert body["id"] == "task-one"
    assert body["title"] == "Custom title"
    assert body["status"] == "not_started"
    assert body["progress_pct"] == 0


# ─── Status transitions ─────────────────────────────────────────────────────


def test_patch_to_in_progress_stamps_started_at(client):
    _seed(client, id="t1")
    body = client.patch("/api/roadmap/t1", json={"status": "in_progress"}).json()
    assert body["status"] == "in_progress"
    assert body["started_at"] is not None
    assert body["completed_at"] is None


def test_patch_to_complete_stamps_completed_at_and_sets_progress_100(client):
    _seed(client, id="t1")
    client.patch("/api/roadmap/t1", json={"status": "in_progress"})
    body = client.patch("/api/roadmap/t1", json={"status": "complete"}).json()
    assert body["status"] == "complete"
    assert body["completed_at"] is not None
    assert body["progress_pct"] == 100


def test_reopening_completed_task_clears_completed_at(client):
    _seed(client, id="t1")
    client.patch("/api/roadmap/t1", json={"status": "in_progress"})
    client.patch("/api/roadmap/t1", json={"status": "complete"})
    body = client.patch("/api/roadmap/t1", json={"status": "in_progress"}).json()
    assert body["status"] == "in_progress"
    assert body["completed_at"] is None


def test_invalid_status_rejected(client):
    _seed(client, id="t1")
    resp = client.patch("/api/roadmap/t1", json={"status": "wonky"})
    assert resp.status_code == 400


def test_patching_unknown_task_returns_404(client):
    resp = client.patch("/api/roadmap/nope", json={"status": "in_progress"})
    assert resp.status_code == 404


# ─── Blocked status ─────────────────────────────────────────────────────────


def test_blocked_requires_reason(client):
    _seed(client, id="t1")
    resp = client.patch("/api/roadmap/t1", json={"status": "blocked"})
    assert resp.status_code == 400
    assert "blocked_reason" in resp.json()["error"]


def test_blocked_with_reason_persists(client):
    _seed(client, id="t1")
    body = client.patch(
        "/api/roadmap/t1",
        json={"status": "blocked", "blocked_reason": "waiting on data"},
    ).json()
    assert body["status"] == "blocked"
    assert body["blocked_reason"] == "waiting on data"


def test_leaving_blocked_clears_reason(client):
    _seed(client, id="t1")
    client.patch(
        "/api/roadmap/t1",
        json={"status": "blocked", "blocked_reason": "x"},
    )
    body = client.patch("/api/roadmap/t1", json={"status": "in_progress"}).json()
    assert body["status"] == "in_progress"
    assert body["blocked_reason"] is None


# ─── Dependency gate ────────────────────────────────────────────────────────


def test_dependency_gate_blocks_transition_when_dep_not_complete(client):
    _seed(client, id="dep", week=1)
    _seed(client, id="child", week=2, dependencies=["dep"])

    resp = client.patch("/api/roadmap/child", json={"status": "in_progress"})
    assert resp.status_code == 400
    assert "dependencies" in resp.json()["error"].lower()


def test_dependency_gate_allows_transition_when_deps_complete(client):
    _seed(client, id="dep", week=1)
    _seed(client, id="child", week=2, dependencies=["dep"])

    # Mark dep complete via the API (exercises the chain end-to-end).
    client.patch("/api/roadmap/dep", json={"status": "in_progress"})
    client.patch("/api/roadmap/dep", json={"status": "complete"})

    resp = client.patch("/api/roadmap/child", json={"status": "in_progress"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "in_progress"


def test_dependency_gate_does_not_block_progress_pct_or_notes(client):
    """Progress and notes updates should always work — dependencies
    only gate the status transition itself."""
    _seed(client, id="dep", week=1)
    _seed(client, id="child", week=2, dependencies=["dep"])

    body = client.patch(
        "/api/roadmap/child", json={"progress_pct": 10, "notes": "early planning"}
    ).json()
    assert body["progress_pct"] == 10
    assert body["notes"] == "early planning"
    # Status stayed put.
    assert body["status"] == "not_started"


def test_dependency_gate_only_applies_to_first_transition(client):
    """Once a task is in_progress, the user can move it forward even if
    a dep regressed (e.g. an upstream task was re-opened) — the gate
    only fires on the not_started → * boundary."""
    _seed(client, id="dep", week=1)
    _seed(client, id="child", week=2, dependencies=["dep"])
    client.patch("/api/roadmap/dep", json={"status": "in_progress"})
    client.patch("/api/roadmap/dep", json={"status": "complete"})
    client.patch("/api/roadmap/child", json={"status": "in_progress"})

    # Now regress the dep.
    client.patch("/api/roadmap/dep", json={"status": "in_progress"})
    # Child can still be advanced.
    resp = client.patch("/api/roadmap/child", json={"status": "complete"})
    assert resp.status_code == 200


# ─── Progress + notes ───────────────────────────────────────────────────────


def test_progress_pct_out_of_range_rejected(client):
    _seed(client, id="t1")
    assert client.patch("/api/roadmap/t1", json={"progress_pct": -1}).status_code == 422
    assert client.patch("/api/roadmap/t1", json={"progress_pct": 101}).status_code == 422


def test_notes_round_trip(client):
    _seed(client, id="t1")
    body = client.patch("/api/roadmap/t1", json={"notes": "found a regression"}).json()
    assert body["notes"] == "found a regression"


# ─── Upsert idempotency ─────────────────────────────────────────────────────


def test_seed_idempotency_preserves_agent_state(client):
    """Re-running the seed script must NOT reset progress made by the
    agent. Content fields refresh; status/progress/notes/timestamps
    don't."""
    _seed(client, id="t1", title="Old title")
    client.patch("/api/roadmap/t1", json={"status": "in_progress"})
    client.patch("/api/roadmap/t1", json={"progress_pct": 40, "notes": "halfway"})

    # Re-seed with updated content.
    _seed(client, id="t1", title="New title", description="updated desc")

    body = client.get("/api/roadmap/t1").json()
    assert body["title"] == "New title"
    assert body["description"] == "updated desc"
    # Agent state preserved.
    assert body["status"] == "in_progress"
    assert body["progress_pct"] == 40
    assert body["notes"] == "halfway"
    assert body["started_at"] is not None
