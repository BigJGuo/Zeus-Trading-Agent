"""Technical feature computation from OHLCV data using pandas-ta."""
from __future__ import annotations

import numpy as np
import pandas as pd
try:
    import pandas_ta as ta  # type: ignore[import-not-found]
except ImportError:
    import pandas_ta_classic as ta
import structlog

logger = structlog.get_logger()


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 30 core technical features from OHLCV DataFrame.

    Args:
        df: DataFrame with DatetimeIndex and columns [open, high, low, close, volume].
            Must be sorted ascending by index.

    Returns:
        DataFrame with same index as df, containing all technical features.
        Warmup periods left as NaN — no lookahead bias (rolling uses closed='right').
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    df = df.sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)
    open_ = df["open"]

    out = pd.DataFrame(index=df.index)

    # ── Trend: Simple & Exponential Moving Averages ────────────────────────────
    out["sma_5"] = ta.sma(close, length=5)
    out["sma_20"] = ta.sma(close, length=20)
    out["sma_50"] = ta.sma(close, length=50)
    out["sma_200"] = ta.sma(close, length=200)
    out["ema_12"] = ta.ema(close, length=12)
    out["ema_26"] = ta.ema(close, length=26)

    # ── Price vs MA ────────────────────────────────────────────────────────────
    out["price_vs_sma20_pct"] = (close - out["sma_20"]) / out["sma_20"] * 100
    out["price_vs_sma50_pct"] = (close - out["sma_50"]) / out["sma_50"] * 100
    out["price_vs_sma200_pct"] = (close - out["sma_200"]) / out["sma_200"] * 100

    # ── MACD ───────────────────────────────────────────────────────────────────
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        out["macd"] = macd_df.iloc[:, 0]
        out["macd_signal"] = macd_df.iloc[:, 2]
        out["macd_hist"] = macd_df.iloc[:, 1]
    else:
        out["macd"] = np.nan
        out["macd_signal"] = np.nan
        out["macd_hist"] = np.nan

    # ── ADX ────────────────────────────────────────────────────────────────────
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and not adx_df.empty:
        out["adx_14"] = adx_df.iloc[:, 0]
        out["di_plus"] = adx_df.iloc[:, 1]
        out["di_minus"] = adx_df.iloc[:, 2]
    else:
        out["adx_14"] = np.nan
        out["di_plus"] = np.nan
        out["di_minus"] = np.nan

    # ── Momentum ───────────────────────────────────────────────────────────────
    out["rsi_14"] = ta.rsi(close, length=14)
    out["rsi_5"] = ta.rsi(close, length=5)

    # momentum_Nd = (close / close.shift(N)) - 1, using data at t only
    out["momentum_5d"] = close.pct_change(5)
    out["momentum_10d"] = close.pct_change(10)
    out["momentum_20d"] = close.pct_change(20)
    out["momentum_60d"] = close.pct_change(60)

    roc5 = ta.roc(close, length=5)
    roc20 = ta.roc(close, length=20)
    out["roc_5"] = roc5
    out["roc_20"] = roc20

    # ── Volatility ─────────────────────────────────────────────────────────────
    # ta.atr returns None (not an empty Series) when fewer than 14 rows are
    # available — guard it the same way MACD/ADX/BBands are guarded above so a
    # newly-listed symbol with insufficient history doesn't blow up the batch.
    atr14 = ta.atr(high, low, close, length=14)
    if atr14 is not None and len(atr14) > 0:
        out["atr_14_pct"] = atr14 / close * 100
    else:
        out["atr_14_pct"] = np.nan

    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        bb_lower = bb_df.iloc[:, 0]
        bb_upper = bb_df.iloc[:, 2]
        bb_mid = bb_df.iloc[:, 1]
        out["bb_width_20"] = (bb_upper - bb_lower) / bb_mid
    else:
        out["bb_width_20"] = np.nan

    log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index)
    out["hist_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    out["hist_vol_5d"] = log_ret.rolling(5).std() * np.sqrt(252)
    out["vol_ratio_5_20"] = out["hist_vol_5d"] / out["hist_vol_20d"]

    # ── Volume ─────────────────────────────────────────────────────────────────
    vol_sma5 = volume.rolling(5).mean()
    vol_sma20 = volume.rolling(20).mean()
    out["volume_ratio_5d"] = vol_sma5 / vol_sma20

    obv = ta.obv(close, volume)
    if obv is not None and not obv.empty:
        out["obv_slope_5d"] = obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-10)
    else:
        out["obv_slope_5d"] = np.nan

    # VWAP deviation: rolling 20-day vwap proxy using typical price * volume
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).rolling(20).sum()
    cum_vol = volume.rolling(20).sum()
    vwap_rolling = cum_tp_vol / cum_vol
    out["vwap_deviation"] = (close - vwap_rolling) / vwap_rolling * 100

    # ── Short-horizon block (for 3–5 day predictors) ───────────────────────────
    rsi_3 = ta.rsi(close, length=3)
    if rsi_3 is not None and not rsi_3.empty:
        out["rsi_3"] = rsi_3
    else:
        out["rsi_3"] = np.nan

    sma_3 = ta.sma(close, length=3)
    if sma_3 is not None and not sma_3.empty:
        out["sma_3"] = sma_3
    else:
        out["sma_3"] = np.nan

    sma_8 = ta.sma(close, length=8)
    if sma_8 is not None and not sma_8.empty:
        out["sma_8"] = sma_8
    else:
        out["sma_8"] = np.nan

    out["price_vs_sma8_pct"] = (close - out["sma_8"]) / out["sma_8"] * 100

    # Compute MACD manually instead of ta.macd — pandas_ta_classic.macd
    # segfaults inside _ema_aligned on certain price series with non-default
    # fast/slow/signal windows (Windows access violation, exit 139). Doing it
    # via ta.ema directly avoids that code path entirely.
    _ema_fast_s = ta.ema(close, length=5)
    _ema_slow_s = ta.ema(close, length=13)
    if _ema_fast_s is not None and _ema_slow_s is not None:
        _macd_s = _ema_fast_s - _ema_slow_s
        _signal_s = ta.ema(_macd_s, length=3)
        out["macd_short"] = _macd_s
        out["macd_short_hist"] = _macd_s - _signal_s if _signal_s is not None else np.nan
    else:
        out["macd_short"] = np.nan
        out["macd_short_hist"] = np.nan

    # ── Long-horizon block (for 20-day predictor) ──────────────────────────────
    rsi_21 = ta.rsi(close, length=21)
    if rsi_21 is not None and not rsi_21.empty:
        out["rsi_21"] = rsi_21
    else:
        out["rsi_21"] = np.nan

    ema_50 = ta.ema(close, length=50)
    if ema_50 is not None and not ema_50.empty:
        out["ema_50"] = ema_50
    else:
        out["ema_50"] = np.nan

    out["momentum_40d"] = close.pct_change(40)

    # Same workaround as macd_short above — ta.macd is unsafe with custom windows.
    _ema_fast_l = ta.ema(close, length=19)
    _ema_slow_l = ta.ema(close, length=39)
    if _ema_fast_l is not None and _ema_slow_l is not None:
        _macd_l = _ema_fast_l - _ema_slow_l
        _signal_l = ta.ema(_macd_l, length=9)
        out["macd_slow"] = _macd_l
        out["macd_slow_hist"] = _macd_l - _signal_l if _signal_l is not None else np.nan
    else:
        out["macd_slow"] = np.nan
        out["macd_slow_hist"] = np.nan

    logger.debug("technical_features_computed", n_rows=len(out), n_features=len(out.columns))
    return out
