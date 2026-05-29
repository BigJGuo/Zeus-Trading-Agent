"""LLM shim test: run an AgentLoop end-to-end against a mocked
AnthropicClient, assert tool round-trip + journal persistence work."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from zeus.agents.base import AgentLoop, AgentRunResult
from zeus.agents.journal import AgentJournal, query_journal
from zeus.llm.client import AnthropicClient, LLMResponse, LLMUsage


class _FakeLLM(AnthropicClient):
    """An AnthropicClient that skips real HTTP and plays back canned responses."""

    def __init__(self, scripted_responses):
        # Skip parent __init__ side effects (which resolves an api key etc.)
        self._tier = "sonnet"
        self._daily_budget = 999.0
        self._spend = {}
        self._script = list(scripted_responses)
        self._calls = []
        from datetime import datetime, timezone
        self._now = lambda: datetime.now(timezone.utc)

    @property
    def model_id(self) -> str:
        return "mocked-model"

    def run(
        self, *, agent_id, system_prompt, messages, tools=None, tool_handlers=None,
        max_tokens=4096, temperature=0.2, cache_system=True,
    ) -> LLMResponse:
        self._calls.append({
            "agent_id": agent_id, "messages": messages, "tools": tools or [],
        })
        # Play scripted tool-use + final text if provided, resolving handlers.
        tool_calls, tool_results = [], []
        script = self._script.pop(0) if self._script else {"text": ""}
        for call in script.get("tool_calls", []):
            name = call["name"]
            inp = call.get("input", {})
            handler = (tool_handlers or {}).get(name)
            if handler is None:
                out = f"ERROR: no handler for {name}"
            else:
                out = handler(**inp)
            tool_calls.append({"id": f"toolu_{name}", "name": name, "input": inp})
            tool_results.append({"id": f"toolu_{name}", "name": name, "output": out})
        return LLMResponse(
            text=script.get("text", ""),
            tool_calls=tool_calls,
            tool_results=tool_results,
            usage=LLMUsage(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
            n_turns=1,
        )


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_research_run_persists_brief(patched_session_factory):
    llm = _FakeLLM([
        {"text": "NVDA — earnings beat, premkt gap +4%. Support at VWAP."}
    ])
    loop = AgentLoop("day_research", llm=llm)
    result = loop.run(
        user_message="Build a pre-market brief for NVDA.",
        persist_output_as="brief",
        persist_title="NVDA pre-market",
        persist_symbol="NVDA",
    )

    assert isinstance(result, AgentRunResult)
    assert result.agent_id == "day_research"
    assert "NVDA" in result.text
    assert len(result.journal_ids_written) == 1

    # And it landed in the journal as a 'brief'
    entries = AgentJournal("day_research").recent(kind="brief", limit=5)
    assert len(entries) == 1
    assert entries[0].symbol == "NVDA"
    assert "earnings beat" in entries[0].body


def test_trader_run_emits_proposal_and_rationale_journal(patched_session_factory):
    # Pre-seed the paired research journal so the trader's query_journal
    # tool round-trip has something to find.
    AgentJournal("day_research").record_brief(
        symbol="NVDA", title="pre-market", body="gap+beat, confidence 4",
    )

    llm = _FakeLLM([
        {
            "tool_calls": [
                {"name": "query_journal",
                 "input": {"agent_id": "day_research",
                           "kind": "brief", "since_hours": 24}},
                {"name": "propose_trade",
                 "input": {"action": "enter", "symbol": "NVDA",
                           "shares": 10, "target_price": 900.0, "stop_price": 870.0,
                           "rationale": "Matches research brief, model conf high.",
                           "brief_id": 1}},
            ],
            "text": "Proposed: 1 entry on NVDA based on day_research brief id=1.",
        },
    ])
    loop = AgentLoop("day", llm=llm)
    result = loop.run(
        user_message="Here are today's top model candidates: NVDA, AMD, MSFT.",
        persist_output_as="decision",
        persist_title="EOD day-trader decision",
    )

    # Proposal captured
    assert len(result.proposals) == 1
    p = result.proposals[0]
    assert p["symbol"] == "NVDA" and p["action"] == "enter"

    # Rationale journal written for the proposal + 'decision' entry for the summary
    rationales = AgentJournal("day").recent(kind="trade_rationale", limit=5)
    assert len(rationales) == 1
    assert rationales[0].symbol == "NVDA"
    assert rationales[0].structured["action"] == "enter"

    decisions = AgentJournal("day").recent(kind="decision", limit=5)
    assert len(decisions) == 1
    assert "day_research" in decisions[0].body


def test_trader_query_journal_outside_allowlist_rejected(patched_session_factory):
    """Day trader asking for 'swing' journal should get the allowlist error
    message back from the tool handler — no exception raised."""
    llm = _FakeLLM([
        {
            "tool_calls": [
                {"name": "query_journal",
                 "input": {"agent_id": "swing", "kind": "trade_rationale"}},
            ],
            "text": "Cross-agent read rejected as expected.",
        },
    ])
    loop = AgentLoop("day", llm=llm)
    result = loop.run(user_message="try to read swing's journal")
    # The error is returned to the model by the handler; it's in the tool_results.
    tool_outputs = [tr["output"] for tr in result.llm_response.tool_results]
    assert any("allowlist" in str(o) for o in tool_outputs)


def test_budget_exhausted_alerts_and_raises(patched_session_factory):
    class _BroeLLM(_FakeLLM):
        def run(self, **kw):
            from zeus.llm.client import BudgetExceededError
            raise BudgetExceededError("fake budget blown")

    loop = AgentLoop("swing_research", llm=_BroeLLM([]))
    with pytest.raises(Exception) as excinfo:
        loop.run(user_message="build briefs")
    assert "budget" in str(excinfo.value).lower()

    # An alert should have been journaled.
    alerts = AgentJournal("swing_research").recent(kind="alert", limit=5)
    assert len(alerts) == 1
    assert alerts[0].title == "llm_budget_exhausted"


def test_overseer_role_sees_all_agents(patched_session_factory):
    # Seed entries from every agent to verify overseer's allowlist = all agents.
    AgentJournal("day_research").record_brief(symbol="NVDA", title="t", body="b")
    AgentJournal("day").record_rationale(symbol="NVDA", title="t", body="b")
    AgentJournal("long_term_research").record_memo(title="t", body="b", symbol="AMZN")

    llm = _FakeLLM([
        {
            "tool_calls": [
                {"name": "query_journal",
                 "input": {"agent_id": "day_research", "kind": "brief"}},
                {"name": "query_journal",
                 "input": {"agent_id": "day", "kind": "trade_rationale"}},
            ],
            "text": "Audit OK: day_research emitted 1 brief; day trader filled 1 with linked rationale.",
        },
    ])
    loop = AgentLoop("overseer", llm=llm)
    result = loop.run(
        user_message="Audit research-trader coupling for today.",
        persist_output_as="decision",
        persist_title="daily audit",
    )
    assert "Audit OK" in result.text
    decs = AgentJournal("overseer").recent(kind="decision", limit=1)
    assert len(decs) == 1
