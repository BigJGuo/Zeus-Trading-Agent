"""Rule-based market regime detector."""
from __future__ import annotations

import pandas as pd

REGIME_PARAMS: dict[str, dict] = {
    "BULL":       {"max_positions": 25, "max_exposure_pct": 0.95},
    "RANGE":      {"max_positions": 15, "max_exposure_pct": 0.75},
    "BEAR":       {"max_positions": 10, "max_exposure_pct": 0.40},
    "HIGH_VOL":   {"max_positions": 5,  "max_exposure_pct": 0.20},
    "TRANSITION": {"max_positions": 12, "max_exposure_pct": 0.60},
}


class RuleBasedRegimeDetector:
    def detect(self, spy_close: pd.Series, vix_close: pd.Series) -> str:
        vix = float(vix_close.iloc[-1])
        spy_20d_return = float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) if len(spy_close) >= 21 else 0.0)
        sma_200 = float(spy_close.tail(200).mean()) if len(spy_close) >= 200 else float(spy_close.mean())
        spy_above_200ma = float(spy_close.iloc[-1]) > sma_200
        return self.detect_from_current(vix, spy_20d_return, spy_above_200ma)

    def detect_from_current(
        self,
        vix_level: float,
        spy_20d_return: float,
        spy_above_200ma: bool,
    ) -> str:
        if vix_level > 30:
            return "HIGH_VOL"
        if vix_level > 20 and spy_20d_return < -0.05:
            return "BEAR"
        if spy_above_200ma and spy_20d_return >= 0.03:
            return "BULL"
        if spy_above_200ma and abs(spy_20d_return) < 0.03:
            return "RANGE"
        return "TRANSITION"
