"""Signal filtering by quality thresholds."""
from __future__ import annotations

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class SignalFilter:
    def apply(
        self,
        signals_df: pd.DataFrame,
        min_confidence: float = 0.45,
        min_liquidity_score: float = 0.30,
        min_expected_return: float = 0.002,
        max_vol_annual: float = 0.80,
    ) -> pd.DataFrame:
        n_start = len(signals_df)
        df = signals_df.copy()

        mask_conf = df["confidence"] >= min_confidence
        n_conf = (~mask_conf).sum()
        df = df[mask_conf]

        mask_liq = df["liquidity_score"] >= min_liquidity_score
        n_liq = (~mask_liq).sum()
        df = df[mask_liq]

        mask_ret = df["expected_return"] >= min_expected_return
        n_ret = (~mask_ret).sum()
        df = df[mask_ret]

        mask_vol = df["vol_estimate"] <= max_vol_annual
        n_vol = (~mask_vol).sum()
        df = df[mask_vol]

        log.info(
            "signal_filter_applied",
            n_start=n_start,
            dropped_confidence=int(n_conf),
            dropped_liquidity=int(n_liq),
            dropped_return=int(n_ret),
            dropped_vol=int(n_vol),
            n_final=len(df),
        )
        return df.reset_index(drop=True)
