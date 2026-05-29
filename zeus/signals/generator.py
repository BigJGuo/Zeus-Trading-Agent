"""Signal generation from model predictions."""
from __future__ import annotations

from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


class PredictorProtocol(Protocol):
    def predict(self, X: pd.DataFrame) -> pd.Series: ...


class SignalGenerator:
    def __init__(self, model: PredictorProtocol) -> None:
        self._model = model

    def generate(
        self,
        features_df: pd.DataFrame,
        regime: str,
        feature_version: str,
    ) -> pd.DataFrame:
        meta_cols = {"symbol", "feature_date", "hist_vol_20d", "dollar_volume_rank_universe"}
        feature_cols = [c for c in features_df.columns if c not in meta_cols]

        scores = self._model.predict(features_df[feature_cols])
        score_std = float(scores.std()) or 1.0

        expected_return = scores.to_numpy()
        confidence = np.clip(np.abs(expected_return) / score_std, 0.0, 1.0)

        up_probability = np.where(expected_return >= 0, 0.5 + 0.5 * confidence, 0.5 - 0.5 * confidence)
        down_probability = 1.0 - up_probability

        expected_downside = np.where(expected_return < 0, expected_return, -np.abs(expected_return) * 0.3)

        blended_score = expected_return * confidence

        signal_date = features_df["feature_date"].iloc[0] if "feature_date" in features_df.columns else date.today()
        model_version = getattr(self._model, "version_", "unknown")

        out = pd.DataFrame(
            {
                "symbol": features_df["symbol"].to_numpy(),
                "signal_date": signal_date,
                "expected_return": expected_return,
                "expected_downside": expected_downside,
                "up_probability": up_probability,
                "down_probability": down_probability,
                "confidence": confidence,
                "blended_score": blended_score,
                "price": features_df["close"].to_numpy() if "close" in features_df.columns else np.nan,
                "vol_estimate": features_df["hist_vol_20d"].to_numpy() if "hist_vol_20d" in features_df.columns else np.nan,
                "liquidity_score": features_df["dollar_volume_rank_universe"].to_numpy() if "dollar_volume_rank_universe" in features_df.columns else np.nan,
                "regime": regime,
                "model_version": model_version,
                "feature_version": feature_version,
            }
        )

        out = out.sort_values("expected_return", ascending=False).reset_index(drop=True)

        log.info(
            "signals_generated",
            total=len(out),
            regime=regime,
            feature_version=feature_version,
        )
        return out
