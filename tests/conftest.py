"""Shared pytest fixtures.

Provides an in-memory SQLite session factory for tests that need to exercise
ORM interactions (StrategyManager, Reconciler). Only creates the subset of
tables the tests touch, so the postgres-specific UUID columns on Trade/Order
don't trip SQLite.
"""
from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import zeus.data.storage.database as db_module
from zeus.data.storage.database import (
    AgentJournal,
    Heartbeat,
    Position,
    RiskEvent,
    Trade,
)


@pytest.fixture()
def sqlite_engine():
    # `StaticPool` + `check_same_thread=False` lets the SQLite in-memory
    # database be shared across threads. FastAPI's TestClient runs request
    # handlers in worker threads, so without this every request sees a
    # fresh empty `:memory:` database and any tables we created in setup
    # vanish.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # UUID-bearing tables (Trade / Order.trade_id / AgentJournal.related_trade_id)
    # use SA 2.0's cross-dialect Uuid type so they round-trip under SQLite for
    # tests. Heartbeat + RiskEvent are needed for the stale-heartbeat job tests.
    Position.__table__.create(bind=eng)
    AgentJournal.__table__.create(bind=eng)
    Trade.__table__.create(bind=eng)
    Heartbeat.__table__.create(bind=eng)
    RiskEvent.__table__.create(bind=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def sqlite_session_factory(sqlite_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=sqlite_engine)


@pytest.fixture()
def patched_session_factory(
    monkeypatch: pytest.MonkeyPatch, sqlite_session_factory
) -> sessionmaker:
    """Patch every module-level alias of get_session_factory so code-under-test
    picks up the in-memory SQLite factory regardless of which module imported
    it. Each module that does `from ... import get_session_factory` gets its
    own binding that a single `setattr(db_module, ...)` wouldn't reach."""
    import importlib

    fake = lambda: sqlite_session_factory
    monkeypatch.setattr(db_module, "get_session_factory", fake)

    for mod_name in (
        "zeus.agents.journal",
        "zeus.agents.metrics",
        "zeus.execution.reconciler",
        "zeus.live.strategy_manager",
        "zeus.agents.overseer",
        "zeus.monitoring.dashboard_service",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "get_session_factory"):
            monkeypatch.setattr(mod, "get_session_factory", fake)

    return sqlite_session_factory


@pytest.fixture()
def session(sqlite_session_factory) -> Iterator[Session]:
    s = sqlite_session_factory()
    try:
        yield s
    finally:
        s.close()
