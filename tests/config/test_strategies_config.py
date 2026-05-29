"""Tests for zeus.config.strategies — YAML loading + weight validation."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zeus.config.strategies import (
    GlobalLimits,
    StrategyBundle,
    StrategyConfig,
    load_strategies,
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "strategies.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_valid_weights_sum_to_one(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies:
          - id: day
            model_name: cross_horizon_day_h3
            horizon_days: 3
            max_hold_days: 5
            weight: 0.30
            max_positions: 5
            max_position_pct: 0.06
            max_exposure_pct: 0.45
          - id: swing
            model_name: cross_horizon_swing_h5
            horizon_days: 5
            max_hold_days: 10
            weight: 0.40
            max_positions: 8
            max_position_pct: 0.08
            max_exposure_pct: 0.60
          - id: long_term
            model_name: cross_horizon_long_term_h20
            horizon_days: 20
            max_hold_days: 60
            weight: 0.30
            max_positions: 10
            max_position_pct: 0.10
            max_exposure_pct: 0.45
        globals:
          max_total_exposure_pct: 0.95
          global_max_position_pct: 0.18
    """)
    bundle = load_strategies(p)
    assert isinstance(bundle, StrategyBundle)
    assert len(bundle.strategies) == 3
    assert [s.id for s in bundle.strategies] == ["day", "swing", "long_term"]
    assert bundle.globals.max_total_exposure_pct == 0.95
    assert bundle.globals.global_max_position_pct == 0.18


def test_by_id_returns_correct_config(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies:
          - id: day
            model_name: m1
            horizon_days: 3
            max_hold_days: 5
            weight: 0.5
            max_positions: 5
            max_position_pct: 0.06
            max_exposure_pct: 0.45
          - id: swing
            model_name: m2
            horizon_days: 5
            max_hold_days: 10
            weight: 0.5
            max_positions: 8
            max_position_pct: 0.08
            max_exposure_pct: 0.60
    """)
    bundle = load_strategies(p)
    swing = bundle.by_id("swing")
    assert swing.model_name == "m2"
    assert swing.max_hold_days == 10


def test_by_id_unknown_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies:
          - id: only
            model_name: m
            horizon_days: 3
            max_hold_days: 5
            weight: 1.0
            max_positions: 5
            max_position_pct: 0.06
            max_exposure_pct: 0.45
    """)
    bundle = load_strategies(p)
    with pytest.raises(KeyError):
        bundle.by_id("nope")


def test_weight_sum_off_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies:
          - id: day
            model_name: m1
            horizon_days: 3
            max_hold_days: 5
            weight: 0.50
            max_positions: 5
            max_position_pct: 0.06
            max_exposure_pct: 0.45
          - id: swing
            model_name: m2
            horizon_days: 5
            max_hold_days: 10
            weight: 0.30
            max_positions: 8
            max_position_pct: 0.08
            max_exposure_pct: 0.60
    """)
    with pytest.raises(ValueError, match="weights must sum"):
        load_strategies(p)


def test_disabled_strategy_excluded_from_weight_sum(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies:
          - id: day
            model_name: m1
            horizon_days: 3
            max_hold_days: 5
            weight: 1.0
            max_positions: 5
            max_position_pct: 0.06
            max_exposure_pct: 0.45
          - id: swing
            model_name: m2
            horizon_days: 5
            max_hold_days: 10
            weight: 0.50
            max_positions: 8
            max_position_pct: 0.08
            max_exposure_pct: 0.60
            enabled: false
    """)
    bundle = load_strategies(p)
    enabled = bundle.enabled()
    assert len(enabled) == 1
    assert enabled[0].id == "day"


def test_empty_strategies_raises(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies: []
    """)
    with pytest.raises(ValueError, match="no strategies"):
        load_strategies(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_strategies(tmp_path / "nope.yaml")


def test_globals_defaults_when_omitted(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, """
        strategies:
          - id: day
            model_name: m1
            horizon_days: 3
            max_hold_days: 5
            weight: 1.0
            max_positions: 5
            max_position_pct: 0.06
            max_exposure_pct: 0.45
    """)
    bundle = load_strategies(p)
    assert bundle.globals == GlobalLimits()
