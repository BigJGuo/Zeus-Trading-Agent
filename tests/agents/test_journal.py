"""Smoke tests for AgentJournal round-trip + search."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from zeus.agents.journal import (
    AgentJournal,
    KNOWN_AGENTS,
    audit_research_coupling,
    query_journal,
)
from zeus.data.storage.database import AgentJournal as JournalRow


def test_unknown_agent_id_raises(patched_session_factory):
    with pytest.raises(ValueError, match="unknown agent_id"):
        AgentJournal(agent_id="rogue")


def test_record_and_recall_rationales_in_order(patched_session_factory):
    j = AgentJournal(agent_id="day")
    for i in range(10):
        j.record_rationale(
            symbol=f"SYM{i:02d}",
            title=f"entry #{i}",
            body=f"bought SYM{i:02d} because signal score={i/10:.1f}",
            confidence=0.5,
        )
    recent = j.recent(kind="trade_rationale", limit=5)
    assert len(recent) == 5
    # Newest first → SYM09 ... SYM05
    assert [e.symbol for e in recent] == [f"SYM{i:02d}" for i in (9, 8, 7, 6, 5)]


def test_record_all_known_kinds(patched_session_factory):
    j = AgentJournal(agent_id="overseer")
    j.record_decision(title="daily", body="reallocate swing from 0.40 to 0.45")
    j.record_memo(title="weekly", body="all 3 traders above Sharpe 1")
    j.record_alert(title="spike", body="day trader 4 losing trades in a row")
    j.record_lesson(title="tbd", body="avoid over-adjusting during CPI week")

    assert len(j.recent()) == 4
    assert len(j.recent(kind="decision")) == 1
    assert len(j.recent(kind="memo")) == 1
    assert len(j.recent(kind="alert")) == 1


def test_reject_blank_title_or_body(patched_session_factory):
    j = AgentJournal(agent_id="day")
    with pytest.raises(ValueError, match="non-empty"):
        j.record_rationale(symbol="AAPL", title="", body="body")
    with pytest.raises(ValueError, match="non-empty"):
        j.record_rationale(symbol="AAPL", title="t", body="")


def test_reject_unknown_kind_via_internal_write(patched_session_factory):
    j = AgentJournal(agent_id="day")
    with pytest.raises(ValueError, match="unknown kind"):
        j._write(kind="gossip", title="t", body="b")


def test_reject_confidence_out_of_range(patched_session_factory):
    j = AgentJournal(agent_id="day_research")
    with pytest.raises(ValueError, match="confidence"):
        j.record_brief(symbol="AAPL", title="t", body="b", confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        j.record_brief(symbol="AAPL", title="t", body="b", confidence=-0.1)


def test_by_symbol_filters_and_window(patched_session_factory, sqlite_session_factory):
    j = AgentJournal(agent_id="swing_research")
    j.record_brief(symbol="AAPL", title="flag pattern", body="AAPL forming flag")
    j.record_brief(symbol="MSFT", title="pullback", body="MSFT pullback to 20-MA")
    # An ancient AAPL brief outside the window
    with sqlite_session_factory() as s:
        old = JournalRow(
            ts=datetime.now(timezone.utc) - timedelta(days=30),
            agent_id="swing_research", kind="brief",
            symbol="AAPL", title="stale", body="stale body",
        )
        s.add(old)
        s.commit()

    aapl = j.by_symbol("AAPL", window_days=7)
    assert [e.title for e in aapl] == ["flag pattern"]  # old brief excluded
    msft = j.by_symbol("MSFT", window_days=7)
    assert [e.title for e in msft] == ["pullback"]


def test_search_finds_substring(patched_session_factory):
    j = AgentJournal(agent_id="long_term_research")
    j.record_memo(title="NVDA thesis v3", body="positioning for datacenter growth")
    j.record_memo(title="AMD thesis", body="datacenter is secondary here")
    j.record_memo(title="random", body="nothing about semis")

    hits = j.search("datacenter")
    assert len(hits) == 2
    titles = [h.title for h in hits]
    assert "NVDA thesis v3" in titles
    assert "AMD thesis" in titles


def test_query_journal_cross_agent(patched_session_factory):
    AgentJournal(agent_id="day").record_rationale(
        symbol="AAPL", title="r1", body="b1",
    )
    AgentJournal(agent_id="swing").record_rationale(
        symbol="AAPL", title="r2", body="b2",
    )

    day_only = query_journal("day", kind="trade_rationale")
    assert [e.title for e in day_only] == ["r1"]

    swing_only = query_journal("swing", kind="trade_rationale")
    assert [e.title for e in swing_only] == ["r2"]


def test_audit_research_coupling_success(patched_session_factory):
    rj = AgentJournal(agent_id="day_research")
    rj.record_brief(symbol="NVDA", title="pre-mkt gap", body="gap + high RVOL")

    trade_ts = datetime.now(timezone.utc) + timedelta(minutes=5)
    linked = audit_research_coupling(
        trader_agent_id="day", symbol="NVDA", trade_ts=trade_ts, window_hours=24,
    )
    assert linked is not None
    assert linked.agent_id == "day_research"
    assert linked.symbol == "NVDA"


def test_audit_research_coupling_missing(patched_session_factory):
    AgentJournal(agent_id="day_research").record_brief(
        symbol="AAPL", title="unrelated", body="AAPL setup",
    )
    linked = audit_research_coupling(
        trader_agent_id="day",
        symbol="NVDA",   # no brief exists for NVDA
        trade_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
        window_hours=24,
    )
    assert linked is None


def test_audit_research_coupling_paired_trader_required(patched_session_factory):
    with pytest.raises(ValueError, match="paired research agent"):
        audit_research_coupling(
            trader_agent_id="overseer",
            symbol="AAPL",
            trade_ts=datetime.now(timezone.utc),
        )


def test_known_agents_set_stable():
    # Guard against accidental expansion — overseer audits rely on this.
    # research_library is a pseudo-agent for the curated paper ingestion
    # pipeline (write-only by the ingest script, read-only by every live
    # agent); it lives in this set so `record_*` write-validation accepts
    # the ingestion writes.
    assert KNOWN_AGENTS == {
        "day", "swing", "long_term",
        "day_research", "swing_research", "long_term_research",
        "overseer",
        "research_library",
    }
