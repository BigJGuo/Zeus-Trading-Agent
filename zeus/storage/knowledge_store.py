"""KnowledgeStore: thin filesystem persistence for learning artifacts.

All paths live under `artifacts_path` from settings. Parquet for tabular
artifacts (IC history, trade outcomes, regime history); JSON for one-off
session plans / metadata.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

import pandas as pd
import structlog

from zeus.config.settings import get_settings

log = structlog.get_logger(__name__)


class KnowledgeStore:
    def __init__(self, artifacts_path: Optional[str] = None) -> None:
        self._root = artifacts_path or get_settings().artifacts_path

    # ─── IC history ───────────────────────────────────────────────────────────
    def ic_history_path(self) -> str:
        return os.path.join(self._root, "knowledge", "ic_history.parquet")

    def append_ic(self, as_of: date, model_version: str, ic: float, n: int) -> None:
        path = self.ic_history_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = pd.DataFrame(
            [{"as_of": pd.Timestamp(as_of), "model_version": model_version, "ic": ic, "n": n}]
        )
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            df = pd.concat([existing, row], ignore_index=True)
        else:
            df = row
        df.to_parquet(path, engine="pyarrow", index=False)
        log.info("ic_appended", as_of=str(as_of), model_version=model_version, ic=ic, n=n)

    def load_ic_history(self) -> pd.DataFrame:
        path = self.ic_history_path()
        if not os.path.exists(path):
            return pd.DataFrame(columns=["as_of", "model_version", "ic", "n"])
        return pd.read_parquet(path)

    def rolling_ic(self, model_version: str, days: int = 20) -> Optional[float]:
        df = self.load_ic_history()
        if df.empty:
            return None
        sub = df[df["model_version"] == model_version].sort_values("as_of").tail(days)
        if sub.empty:
            return None
        return float(sub["ic"].mean())

    # ─── Trade outcomes (append-only) ─────────────────────────────────────────
    def trade_outcomes_path(self) -> str:
        return os.path.join(self._root, "trade_outcomes", "all_trades.parquet")

    def append_trade_outcome(self, outcome: dict) -> None:
        path = self.trade_outcomes_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        row = pd.DataFrame([outcome])
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            df = pd.concat([existing, row], ignore_index=True)
        else:
            df = row
        df.to_parquet(path, engine="pyarrow", index=False)

    def load_trade_outcomes(self) -> pd.DataFrame:
        path = self.trade_outcomes_path()
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)
