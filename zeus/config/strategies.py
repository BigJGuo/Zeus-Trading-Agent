"""Strategy configuration loader.

Parses `config/strategies.yaml` into typed `StrategyConfig` + `GlobalLimits`
objects used by StrategyAllocator / StrategyManager. Kept deliberately thin —
no side effects beyond YAML parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class StrategyConfig:
    id: str                         # 'day' | 'swing' | 'long_term' | ...
    model_name: str                 # model_versions.model_name to load
    horizon_days: int               # target prediction horizon
    max_hold_days: int              # time-exit after this many days
    weight: float                   # 0..1 share of NAV for this strategy
    max_positions: int              # cap on concurrent positions
    max_position_pct: float         # % of NAV per single name
    max_exposure_pct: float         # % of NAV this strategy may deploy
    enabled: bool = True


@dataclass(frozen=True)
class GlobalLimits:
    max_total_exposure_pct: float = 0.95
    global_max_position_pct: float = 0.18


@dataclass(frozen=True)
class StrategyBundle:
    strategies: List[StrategyConfig]
    globals: GlobalLimits

    def by_id(self, strategy_id: str) -> StrategyConfig:
        for s in self.strategies:
            if s.id == strategy_id:
                return s
        raise KeyError(f"unknown strategy_id: {strategy_id}")

    def enabled(self) -> List[StrategyConfig]:
        return [s for s in self.strategies if s.enabled]


def _default_path() -> Path:
    # Project root is two levels up from zeus/config/strategies.py
    return Path(__file__).resolve().parents[2] / "config" / "strategies.yaml"


def load_strategies(path: Optional[Path] = None) -> StrategyBundle:
    """Read and validate the strategies YAML. Raises on weight-sum != 1 or empty."""
    p = Path(path) if path else _default_path()
    if not p.exists():
        raise FileNotFoundError(f"strategies.yaml not found at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    strategies_raw = raw.get("strategies", [])
    if not strategies_raw:
        raise ValueError("strategies.yaml has no strategies")

    strategies = [
        StrategyConfig(
            id=s["id"],
            model_name=s["model_name"],
            horizon_days=int(s["horizon_days"]),
            max_hold_days=int(s["max_hold_days"]),
            weight=float(s["weight"]),
            max_positions=int(s["max_positions"]),
            max_position_pct=float(s["max_position_pct"]),
            max_exposure_pct=float(s["max_exposure_pct"]),
            enabled=bool(s.get("enabled", True)),
        )
        for s in strategies_raw
    ]
    enabled_weight = sum(s.weight for s in strategies if s.enabled)
    if enabled_weight <= 0 or abs(enabled_weight - 1.0) > 0.02:
        raise ValueError(
            f"enabled strategy weights must sum to ~1.0, got {enabled_weight:.4f}"
        )

    globals_raw = raw.get("globals", {})
    glimits = GlobalLimits(
        max_total_exposure_pct=float(globals_raw.get("max_total_exposure_pct", 0.95)),
        global_max_position_pct=float(globals_raw.get("global_max_position_pct", 0.18)),
    )
    return StrategyBundle(strategies=strategies, globals=glimits)


@lru_cache(maxsize=1)
def get_strategies() -> StrategyBundle:
    return load_strategies()
