"""Macro feature computation with FRED / yfinance fallback and daily caching."""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

_CACHE_TTL = 86400  # 24 hours in seconds


def _get_cache():
    try:
        from zeus.data.storage.cache import Cache
        return Cache()
    except Exception:
        return None


def _fetch_from_fred(series_id: str, api_key: str, as_of: date) -> Optional[float]:
    try:
        import pandas_datareader.data as web
        start = as_of - timedelta(days=30)
        df = web.DataReader(series_id, "fred", start, as_of, api_key=api_key)
        if df.empty:
            return None
        return float(df.iloc[-1, 0])
    except Exception as exc:
        logger.warning("fred_fetch_failed", series=series_id, error=str(exc))
        return None


def _fetch_put_call_series(min_obs: int = 6) -> Optional[Any]:
    """Daily CBOE Total Put/Call ratio, ~30 trading days back.

    Primary source: stooq.com (no auth, no rate limit on this series).
    yfinance has historically published `^CPC` but the ticker disappears
    intermittently, so it's a fallback rather than the primary.

    Returns a pandas Series indexed by date, or None when both sources
    are unreachable. The pipeline already tolerates None macro fields, so
    a failed lookup here just leaves `put_call_ratio` at None for the day
    — preferable to silently substituting a stale value.
    """
    try:
        import io
        import pandas as pd
        import requests
        # stooq lays out CBOE Total Put/Call as ^cpc, CSV format, columns
        # Date,Open,High,Low,Close. The "Close" value is the published
        # daily ratio.
        resp = requests.get(
            "https://stooq.com/q/d/l/?s=^cpc&i=d",
            timeout=10,
            headers={"User-Agent": "zeus-features/1.0"},
        )
        if resp.ok and resp.text and "Date" in resp.text[:128]:
            df = pd.read_csv(io.StringIO(resp.text))
            if "Close" in df.columns and "Date" in df.columns and len(df) >= min_obs:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").set_index("Date")
                return df["Close"].dropna().tail(30)
    except Exception as exc:
        logger.warning("put_call_stooq_failed", error=str(exc))

    # yfinance fallback — `^CPC` is occasionally available, but the ticker
    # is fragile and may return empty. Keep as best-effort.
    series = _fetch_yfinance_series("^CPC", period="60d")
    if series is not None and not series.empty:
        return series
    return None


def _fetch_yfinance_series(ticker: str, period: str = "30d") -> Optional[Any]:
    try:
        import pandas as pd
        import yfinance as yf
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close[close.columns[0]]
        return close
    except Exception as exc:
        logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
        return None


def compute_macro_features(as_of: date) -> dict:
    """Compute macro features for the given date with daily caching.

    Args:
        as_of: The date for which macro features are needed.

    Returns:
        Dict with keys: vix_level, vix_5d_change, yield_10y, yield_2y,
        yield_curve_spread, fed_funds_rate, spy_return_5d, spy_return_20d.
        Missing values are None.
    """
    cache_key = f"macro:features:{as_of.isoformat()}"
    cache = _get_cache()

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("macro_features_cache_hit", as_of=str(as_of))
            return cached

    features: dict[str, Any] = {
        "vix_level": None,
        "vix_5d_change": None,
        "yield_10y": None,
        "yield_2y": None,
        "yield_curve_spread": None,
        "fed_funds_rate": None,
        "spy_return_5d": None,
        "spy_return_20d": None,
        # CBOE Total Put/Call ratio — fear/complacency proxy. Above 1.0 is
        # heavy hedging / fear, below 0.7 is complacency. Pulled from
        # stooq (no API key, daily) with a yfinance fallback. See the
        # `^CPC` / `^cpc` series below.
        "put_call_ratio": None,
        "put_call_ratio_5d_change": None,
    }

    try:
        from zeus.config.settings import get_settings
        settings = get_settings()
        fred_key = settings.fred_api_key
    except Exception:
        fred_key = None

    # ── VIX ────────────────────────────────────────────────────────────────────
    vix_series = None
    if fred_key:
        vix_val = _fetch_from_fred("VIXCLS", fred_key, as_of)
        if vix_val is not None:
            features["vix_level"] = vix_val
    if features["vix_level"] is None:
        vix_series = _fetch_yfinance_series("^VIX", period="60d")
        if vix_series is not None and not vix_series.empty:
            features["vix_level"] = float(vix_series.iloc[-1])
            if len(vix_series) >= 6:
                features["vix_5d_change"] = float(vix_series.iloc[-1] - vix_series.iloc[-6])

    if features["vix_5d_change"] is None and vix_series is not None and len(vix_series) >= 6:
        features["vix_5d_change"] = float(vix_series.iloc[-1] - vix_series.iloc[-6])

    # ── Treasury Yields ────────────────────────────────────────────────────────
    if fred_key:
        features["yield_10y"] = _fetch_from_fred("DGS10", fred_key, as_of)
        features["yield_2y"] = _fetch_from_fred("DGS2", fred_key, as_of)
        features["fed_funds_rate"] = _fetch_from_fred("FEDFUNDS", fred_key, as_of)
    else:
        tnx = _fetch_yfinance_series("^TNX", period="30d")
        if tnx is not None and not tnx.empty:
            features["yield_10y"] = float(tnx.iloc[-1]) / 100
        tyx = _fetch_yfinance_series("^IRX", period="30d")
        if tyx is not None and not tyx.empty:
            features["yield_2y"] = float(tyx.iloc[-1]) / 100

    if features["yield_10y"] is not None and features["yield_2y"] is not None:
        features["yield_curve_spread"] = features["yield_10y"] - features["yield_2y"]

    # ── SPY Returns ────────────────────────────────────────────────────────────
    spy = _fetch_yfinance_series("SPY", period="60d")
    if spy is not None and not spy.empty:
        if len(spy) >= 6:
            features["spy_return_5d"] = float((spy.iloc[-1] / spy.iloc[-6]) - 1)
        if len(spy) >= 21:
            features["spy_return_20d"] = float((spy.iloc[-1] / spy.iloc[-21]) - 1)

    # ── CBOE Put/Call Ratio (alt-data) ────────────────────────────────────────
    # Sentiment is orthogonal to the OHLCV/fundamentals features the model
    # already has — exactly the kind of signal that's still pricing alpha in
    # 2026 because retail and most quant systems don't condition on it.
    pcr_series = _fetch_put_call_series()
    if pcr_series is not None and not pcr_series.empty:
        features["put_call_ratio"] = float(pcr_series.iloc[-1])
        if len(pcr_series) >= 6:
            features["put_call_ratio_5d_change"] = float(
                pcr_series.iloc[-1] - pcr_series.iloc[-6]
            )

    logger.info(
        "macro_features_computed",
        as_of=str(as_of),
        n_non_null=sum(v is not None for v in features.values()),
    )

    if cache is not None:
        try:
            cache.set(cache_key, features, ttl=_CACHE_TTL)
        except Exception as exc:
            logger.warning("macro_cache_write_failed", error=str(exc))

    return features
