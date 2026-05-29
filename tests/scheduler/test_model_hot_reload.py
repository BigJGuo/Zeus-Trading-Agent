"""Tests for `reload_models_if_promoted` — model hot-swap path.

Verifies that when a newer staging/production row appears in model_versions
matching a strategy's `model_name`, the StrategyContext's `.model` and the
SignalGenerator's cached `_model` reference both get updated in one tick,
without restarting the process.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import pytest


# ─── Minimal stand-ins for the runtime objects the helper inspects ───────────


@dataclass
class _FakeModel:
    """Models are matched by `version_` — the loader sets this attribute."""
    version_: str = "v1"


class _FakeSignalGenerator:
    """SignalGenerator caches a reference to its model; the swap must update
    that reference in addition to ctx.model, or predict() keeps routing
    through the stale model."""

    def __init__(self, model):
        self._model = model


@dataclass
class _FakeStrategyConfig:
    id: str
    model_name: str


class _FakeContext:
    def __init__(self, sid: str, model_name: str, model: _FakeModel):
        self.config = _FakeStrategyConfig(id=sid, model_name=model_name)
        self.model = model
        self.signal_generator = _FakeSignalGenerator(model)


class _FakeManager:
    def __init__(self, contexts):
        self._ctxs = contexts

    def strategy_ids(self):
        return list(self._ctxs.keys())

    def context(self, sid):
        return self._ctxs[sid]


def _make_loop(contexts, after_hours_ctx=None):
    return SimpleNamespace(
        _strategy_mgr=_FakeManager(contexts),
        _after_hours_ctx=after_hours_ctx,
    )


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_no_swap_when_version_unchanged(monkeypatch):
    from zeus.scheduler import runner

    ctx = _FakeContext("day", "cross_horizon_day_h3", _FakeModel(version_="v1"))
    loop = _make_loop({"day": ctx})

    # Loader returns a model with the *same* version → no swap should happen.
    monkeypatch.setattr(runner, "_load_model_by_name", lambda name: _FakeModel(version_="v1"))
    monkeypatch.setattr(runner, "_load_latest_model", lambda: None)

    swaps = runner.reload_models_if_promoted(loop)

    assert swaps == {}
    assert ctx.model.version_ == "v1"
    assert ctx.signal_generator._model.version_ == "v1"


def test_swap_when_newer_version_promoted(monkeypatch):
    from zeus.scheduler import runner

    old = _FakeModel(version_="v1")
    ctx = _FakeContext("day", "cross_horizon_day_h3", old)
    loop = _make_loop({"day": ctx})

    new = _FakeModel(version_="v2")
    monkeypatch.setattr(runner, "_load_model_by_name", lambda name: new)
    monkeypatch.setattr(runner, "_load_latest_model", lambda: None)

    swaps = runner.reload_models_if_promoted(loop)

    assert swaps == {"day": "v2"}
    # Both the context AND the signal generator's cached reference swapped —
    # otherwise predict() would keep routing through the old model.
    assert ctx.model is new
    assert ctx.signal_generator._model is new


def test_swap_handles_multiple_strategies_independently(monkeypatch):
    from zeus.scheduler import runner

    day_old = _FakeModel(version_="v1")
    swing_old = _FakeModel(version_="vA")
    long_term_old = _FakeModel(version_="vX")

    ctxs = {
        "day": _FakeContext("day", "cross_horizon_day_h3", day_old),
        "swing": _FakeContext("swing", "cross_horizon_swing_h5", swing_old),
        "long_term": _FakeContext("long_term", "cross_horizon_long_term_h20", long_term_old),
    }
    loop = _make_loop(ctxs)

    # Only swing has a new version in the registry.
    def _loader(name):
        if name == "cross_horizon_swing_h5":
            return _FakeModel(version_="vB")
        if name == "cross_horizon_day_h3":
            return _FakeModel(version_="v1")
        return _FakeModel(version_="vX")

    monkeypatch.setattr(runner, "_load_model_by_name", _loader)
    monkeypatch.setattr(runner, "_load_latest_model", lambda: None)

    swaps = runner.reload_models_if_promoted(loop)

    assert swaps == {"swing": "vB"}
    assert ctxs["day"].model is day_old
    assert ctxs["swing"].model.version_ == "vB"
    assert ctxs["long_term"].model is long_term_old


def test_loader_returns_none_is_tolerated(monkeypatch):
    """A transient DB blip or artifact-missing path returns None from the
    loader — we must not crash and must leave the existing model in place."""
    from zeus.scheduler import runner

    ctx = _FakeContext("day", "cross_horizon_day_h3", _FakeModel(version_="v1"))
    loop = _make_loop({"day": ctx})

    monkeypatch.setattr(runner, "_load_model_by_name", lambda name: None)
    monkeypatch.setattr(runner, "_load_latest_model", lambda: None)

    swaps = runner.reload_models_if_promoted(loop)
    assert swaps == {}
    assert ctx.model.version_ == "v1"


def test_legacy_model_on_research_ctx_also_swaps(monkeypatch):
    from zeus.scheduler import runner

    after_hours = SimpleNamespace(
        model=_FakeModel(version_="legacy-v1"),
        signal_generator=_FakeSignalGenerator(_FakeModel(version_="legacy-v1")),
    )
    ctx = _FakeContext("day", "cross_horizon_day_h3", _FakeModel(version_="v1"))
    loop = _make_loop({"day": ctx}, after_hours_ctx=after_hours)

    monkeypatch.setattr(runner, "_load_model_by_name", lambda name: _FakeModel(version_="v1"))
    monkeypatch.setattr(runner, "_load_latest_model", lambda: _FakeModel(version_="legacy-v2"))

    swaps = runner.reload_models_if_promoted(loop)

    assert "__legacy__" in swaps
    assert swaps["__legacy__"] == "legacy-v2"
    assert after_hours.model.version_ == "legacy-v2"
    assert after_hours.signal_generator._model.version_ == "legacy-v2"


def test_no_strategy_manager_returns_empty(monkeypatch):
    """Single-model deployments (no StrategyManager attached) must short-circuit
    safely rather than NPEing inside the loop."""
    from zeus.scheduler import runner

    loop = SimpleNamespace(_strategy_mgr=None, _after_hours_ctx=None)
    monkeypatch.setattr(runner, "_load_model_by_name", lambda name: _FakeModel(version_="v2"))
    monkeypatch.setattr(runner, "_load_latest_model", lambda: None)

    swaps = runner.reload_models_if_promoted(loop)
    assert swaps == {}
