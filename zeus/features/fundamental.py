"""Fundamental feature extraction from yfinance raw info dict."""
from __future__ import annotations

import math
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

# (dict_key, output_feature_name, clip_min, clip_max)
_FUNDAMENTAL_FIELDS: list[tuple[str, str, Optional[float], Optional[float]]] = [
    ("trailingPE", "pe_ratio", None, 500.0),
    ("priceToBook", "pb_ratio", None, 50.0),
    ("priceToSalesTrailing12Months", "ps_ratio", None, 100.0),
    ("enterpriseToEbitda", "ev_ebitda", None, 200.0),
    ("earningsYield", "earnings_yield", -1.0, 1.0),
    ("revenueGrowth", "revenue_growth_yoy", -1.0, 10.0),
    ("earningsGrowth", "earnings_growth_yoy", -5.0, 20.0),
    ("grossMargins", "gross_margin", -1.0, 1.0),
    ("operatingMargins", "operating_margin", -1.0, 1.0),
    ("debtToEquity", "debt_to_equity", None, 1000.0),
    ("currentRatio", "current_ratio", 0.0, 50.0),
    ("returnOnEquity", "roe", -5.0, 5.0),
    ("returnOnAssets", "roa", -2.0, 2.0),
    ("shortRatio", "short_interest_ratio", 0.0, 100.0),
]


def _safe_float(value: Any, clip_min: Optional[float], clip_max: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    if clip_max is not None and v > clip_max:
        return None
    if clip_min is not None and v < clip_min:
        return None
    return v


def compute_fundamental_features(symbol: str, fundamentals: dict) -> dict:
    """Extract and clean fundamental features from a raw yfinance info dict.

    Args:
        symbol: Ticker symbol (used for logging only).
        fundamentals: Raw dict from YFinanceClient.fetch_fundamentals (yfinance .info).

    Returns:
        Dict of 15 clean feature values; missing/invalid values are None.
    """
    features: dict[str, Any] = {}

    for dict_key, feat_name, clip_min, clip_max in _FUNDAMENTAL_FIELDS:
        raw = fundamentals.get(dict_key)
        features[feat_name] = _safe_float(raw, clip_min, clip_max)

    # earnings_yield: sometimes not present directly — derive from PE if needed
    if features.get("earnings_yield") is None and features.get("pe_ratio") is not None:
        pe = features["pe_ratio"]
        if pe and pe > 0:
            features["earnings_yield"] = 1.0 / pe

    # market_cap_log
    market_cap = _safe_float(fundamentals.get("marketCap"), 0.0, None)
    if market_cap and market_cap > 0:
        features["market_cap_log"] = math.log10(market_cap)
    else:
        features["market_cap_log"] = None

    logger.debug("fundamental_features_computed", symbol=symbol, n_non_null=sum(v is not None for v in features.values()))
    return features
