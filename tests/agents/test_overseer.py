"""OverseerAgent: circuit-breakers, weekly review, LLM-driven reallocation."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from zeus.agents.metrics import AgentMetrics, ResearchMetrics
from zeus.agents.overseer import (
    MAX_DD_HALT_PCT,
    OverseerAgent,
    _extract_weight_proposal,
)
from zeus.config.strategies import GlobalLimits, StrategyConfig
from zeus.llm.client import AnthropicClient, LLMResponse, LLMUsage
from zeus.risk.engine import RiskEngine
from zeus.risk.limits import RiskLimits
from zeus.risk.strategy_allocator import OverseerStrategyAllocator


# ─── Test doubles ─────────────────────────────────────────────────────────────


class _FakeLLM(AnthropicClient):
    """Skip parent init; play back a canned response."""

    def __init__(self, text: str = "ok"):
        self._tier = "opus"
        self._daily_budget = 999.0
        self._spend = {}
        self._text = text
        from datetime import datetime, timezone
        self._now = lambda: datetime.now(timezone.utc)

    @property
    def model_id(self) -> str:
        return "mocked-opus"

    def run(self, **_) -> LLMResponse:
        return LLMResponse(
            text=self._text,
            tool_calls=[],
            tool_results=[],
            usage=LLMUsage(input_tokens=100, output_tokens=50),
            stop_reason="end_turn",
            n_turns=1,
        )


def _strategies() -> List[StrategyConfig]:
    return [
        StrategyConfig(
            id="day", model_name="cross_horizon_day", horizon_days=3,
            max_hold_days=5, weight=0.30, max_positions=5,
            max_position_pct=0.10, max_exposure_pct=0.30,
        ),
        StrategyConfig(
            id="swing", model_name="cross_horizon_swing", horizon_days=5,
            max_hold_days=15, weight=0.40, max_positions=8,
            max_position_pct=0.12, max_exposure_pct=0.40,
        ),
        StrategyConfig(
            id="long_term", model_name="cross_horizon_long_term", horizon_days=20,
            max_hold_days=180, weight=0.30, max_positions=10,
            max_position_pct=0.15, max_exposure_pct=0.40,
        ),
    ]


def _build(llm: AnthropicClient | None = None) -> tuple[OverseerAgent, RiskEngine, OverseerStrategyAllocator]:
    risk_engine = RiskEngine(
        limits=RiskLimits(),
        starting_capital=100_000.0,
    )
    allocator = OverseerStrategyAllocator(
        strategies=_strategies(),
        globals_=GlobalLimits(),
        max_weekly_shift=0.10,
        risk_engine=risk_engine,
    )
    agent = OverseerAgent(
        llm=llm or _FakeLLM("ok"),
        risk_engine=risk_engine,
        allocator=allocator,
    )
    return agent, risk_engine, allocator


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_extract_weight_proposal_happy_path():
    text = (
        "## Review\nGreat week.\n\n"
        '```json\n{"day": 0.25, "swing": 0.45, "long_term": 0.30, '
        '"rationale": "swing is hot"}\n```\nDone.'
    )
    p = _extract_weight_proposal(text)
    assert p is not None
    assert p["day"] == 0.25
    assert p["rationale"] == "swing is hot"


def test_extract_weight_proposal_no_json():
    assert _extract_weight_proposal("nothing here") is None


def test_extract_weight_proposal_rejects_non_strategy_dict():
    # A fenced JSON block that contains only unknown keys → None.
    text = '```json\n{"foo": 0.5, "bar": 0.5}\n```'
    assert _extract_weight_proposal(text) is None


def test_daily_circuit_breaker_halts_on_drawdown(monkeypatch, patched_session_factory):
    agent, risk_engine, _ = _build()

    # Stub _gather_metrics to inject a blown drawdown.
    def fake_gather(self, window_days, environment):
        return {
            "day": AgentMetrics(
                agent_id="day", window_days=window_days, n_trades=10,
                hit_rate=0.50, sharpe_annualized=0.5,
                max_drawdown_pct=MAX_DD_HALT_PCT + 0.01,  # over the cap
            ),
            "swing": AgentMetrics(agent_id="swing", window_days=window_days),
            "long_term": AgentMetrics(agent_id="long_term", window_days=window_days),
            "day_research": ResearchMetrics(agent_id="day_research", window_days=window_days),
            "swing_research": ResearchMetrics(agent_id="swing_research", window_days=window_days),
            "long_term_research": ResearchMetrics(agent_id="long_term_research", window_days=window_days),
        }
    monkeypatch.setattr(OverseerAgent, "_gather_metrics", fake_gather)
    monkeypatch.setattr(OverseerAgent, "_scan_decoupled_fills",
                        lambda self, *, environment, since=None: [])

    report = agent.run_daily_aggregate()
    assert ("day", pytest.approx) or True  # sanity placeholder
    assert any(aid == "day" for aid, _ in report.halts_issued)
    assert risk_engine.is_agent_halted("day")


def test_daily_aggregate_no_halts_when_clean(monkeypatch, patched_session_factory):
    agent, risk_engine, _ = _build()

    def fake_gather(self, window_days, environment):
        return {
            aid: AgentMetrics(
                agent_id=aid, window_days=window_days, n_trades=5, hit_rate=0.65,
                sharpe_annualized=1.2, max_drawdown_pct=0.05,
            ) for aid in ("day", "swing", "long_term")
        } | {
            aid: ResearchMetrics(agent_id=aid, window_days=window_days, n_briefs=3)
            for aid in ("day_research", "swing_research", "long_term_research")
        }
    monkeypatch.setattr(OverseerAgent, "_gather_metrics", fake_gather)
    monkeypatch.setattr(OverseerAgent, "_scan_decoupled_fills",
                        lambda self, *, environment, since=None: [])

    report = agent.run_daily_aggregate()
    assert report.halts_issued == []
    assert not risk_engine.is_agent_halted("day")
    assert report.journal_id is not None


def test_weekly_review_applies_proposed_reallocation(monkeypatch, patched_session_factory):
    llm = _FakeLLM(
        text=(
            "Post-mortem: swing led the book this week.\n\n"
            '```json\n{"day": 0.25, "swing": 0.45, "long_term": 0.30, '
            '"rationale": "swing outperformed"}\n```'
        ),
    )
    agent, _, allocator = _build(llm=llm)
    monkeypatch.setattr(
        OverseerAgent, "_gather_metrics",
        lambda self, window_days, environment: {
            aid: AgentMetrics(agent_id=aid, window_days=window_days)
            for aid in ("day", "swing", "long_term")
        } | {
            aid: ResearchMetrics(agent_id=aid, window_days=window_days)
            for aid in ("day_research", "swing_research", "long_term_research")
        },
    )

    review = agent.run_weekly_review()
    assert review.allocation_update is not None
    assert review.allocation_update.rejected_reason is None
    w = allocator.effective_weights()
    assert pytest.approx(w["day"], abs=1e-6) == 0.25
    assert pytest.approx(w["swing"], abs=1e-6) == 0.45


def test_weekly_review_rejects_out_of_bounds_proposal(monkeypatch, patched_session_factory):
    llm = _FakeLLM(
        text=(
            '```json\n{"day": 0.05, "swing": 0.65, "long_term": 0.30, '
            '"rationale": "go big"}\n```'
        ),
    )
    agent, _, allocator = _build(llm=llm)
    monkeypatch.setattr(
        OverseerAgent, "_gather_metrics",
        lambda self, window_days, environment: {
            aid: AgentMetrics(agent_id=aid, window_days=window_days)
            for aid in ("day", "swing", "long_term")
        } | {
            aid: ResearchMetrics(agent_id=aid, window_days=window_days)
            for aid in ("day_research", "swing_research", "long_term_research")
        },
    )

    review = agent.run_weekly_review()
    assert review.allocation_update is not None
    assert review.allocation_update.rejected_reason is not None
    # Weights unchanged.
    w = allocator.effective_weights()
    assert pytest.approx(w["day"], abs=1e-6) == 0.30


def test_realtime_monitor_halts_on_kill_switch(monkeypatch, patched_session_factory):
    agent, risk_engine, _ = _build()
    monkeypatch.setattr(OverseerAgent, "_scan_decoupled_fills",
                        lambda self, *, environment, since=None: [])
    risk_engine.trip_kill_switch()

    events = agent.run_realtime_monitor()
    halt_events = [e for e in events if e["type"] == "halt"]
    assert any(e["agent_id"] == "day" for e in halt_events)
    assert risk_engine.is_agent_halted("day")


def test_realtime_monitor_halts_on_decoupled_fill(monkeypatch, patched_session_factory):
    agent, risk_engine, _ = _build()

    def fake_scan(self, *, environment, since=None):
        return [{"agent_id": "swing", "symbol": "NVDA", "entry_ts": "2026-04-20T14:30:00Z"}]

    monkeypatch.setattr(OverseerAgent, "_scan_decoupled_fills", fake_scan)
    events = agent.run_realtime_monitor()
    halts = [e for e in events if e["type"] == "halt"]
    assert any(e["agent_id"] == "swing" for e in halts)
    assert risk_engine.is_agent_halted("swing")
