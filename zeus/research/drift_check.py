"""Information Coefficient (IC) drift detection.

Compares a snapshot of yesterday's predicted returns against today's realized
1-day returns and persists the resulting Spearman IC to KnowledgeStore.

A sustained IC drop (rolling 5-day mean < 0.01) is the gate that triggers a
nightly retrain in `after_hours.run_nightly_retrain`.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from zeus.storage.knowledge_store import KnowledgeStore

log = structlog.get_logger(__name__)

DRIFT_THRESHOLD = 0.01           # rolling-5d mean IC below this = drift
RETRAIN_TRIGGER_DAYS = 5         # consecutive low-IC days


def spearman_ic(predicted: pd.Series, actual: pd.Series) -> float:
    """Spearman rank correlation. Returns 0.0 if either series is degenerate."""
    if len(predicted) != len(actual):
        raise ValueError(f"length mismatch: predicted={len(predicted)} actual={len(actual)}")
    if len(predicted) < 3:
        return 0.0
    df = pd.DataFrame({"p": predicted.values, "a": actual.values}).dropna()
    if len(df) < 3 or df["p"].nunique() < 2 or df["a"].nunique() < 2:
        return 0.0
    return float(df["p"].rank().corr(df["a"].rank()))


def compute_daily_ic(predicted_by_symbol: dict[str, float],
                     actual_by_symbol: dict[str, float]) -> tuple[float, int]:
    """Inner-join predictions and actuals, return (ic, n_paired)."""
    common = set(predicted_by_symbol).intersection(actual_by_symbol)
    if not common:
        return 0.0, 0
    p = pd.Series({k: predicted_by_symbol[k] for k in common})
    a = pd.Series({k: actual_by_symbol[k] for k in common})
    return spearman_ic(p, a), len(common)


def detect_drift(store: KnowledgeStore, model_version: str) -> bool:
    """True if rolling 5-day IC mean is below threshold for the given model."""
    df = store.load_ic_history()
    if df.empty:
        return False
    sub = df[df["model_version"] == model_version].sort_values("as_of").tail(RETRAIN_TRIGGER_DAYS)
    if len(sub) < RETRAIN_TRIGGER_DAYS:
        return False
    mean_ic = float(sub["ic"].mean())
    drifted = mean_ic < DRIFT_THRESHOLD
    log.info(
        "drift_check",
        model_version=model_version,
        rolling_ic=mean_ic,
        drifted=drifted,
        threshold=DRIFT_THRESHOLD,
    )
    return drifted


def record_daily_ic(
    store: KnowledgeStore,
    as_of: date,
    model_version: str,
    predicted_by_symbol: dict[str, float],
    actual_by_symbol: dict[str, float],
) -> Optional[float]:
    """Compute, persist, and return today's IC. Returns None if no overlap."""
    ic, n = compute_daily_ic(predicted_by_symbol, actual_by_symbol)
    if n == 0:
        log.warning("ic_no_overlap", as_of=str(as_of), model_version=model_version)
        return None
    store.append_ic(as_of=as_of, model_version=model_version, ic=ic, n=n)
    return ic
