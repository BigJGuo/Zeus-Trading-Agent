"""Cross-sectional feature computation for panel data (symbol × date)."""
from __future__ import annotations

import pandas as pd
import structlog

logger = structlog.get_logger()

_UNIVERSE_RANK_COLS = {
    "return_5d": "return_5d_rank_universe",
    "return_20d": "return_20d_rank_universe",
    "vol_20d": "vol_20d_rank_universe",
    "rsi_14": "rsi_rank_universe",
    "momentum_20d": "momentum_rank_universe",
    "dollar_volume": "dollar_volume_rank_universe",
}

_SECTOR_RANK_COLS = {
    "return_5d": "return_5d_rank_sector",
    "return_20d": "return_20d_rank_sector",
}


def compute_cross_sectional_features(
    feature_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Add cross-sectional rank features to a panel DataFrame.

    Args:
        feature_df: Panel with columns [symbol, ts, ...features...]. One row per
                    symbol-date pair.
        sector_map: Mapping of symbol -> sector string.

    Returns:
        Copy of feature_df with rank and relative-strength columns appended.
    """
    df = feature_df.copy()

    # Attach sector so we can groupby it
    df["_sector"] = df["symbol"].map(sector_map)

    # dollar_volume proxy: volume * close (if available)
    if "volume" in df.columns and "close" in df.columns:
        df["dollar_volume"] = df["volume"].astype(float) * df["close"].astype(float)

    # ── Universe-wide percentile ranks ─────────────────────────────────────────
    # Use the built-in `groupby().rank()` path instead of `transform()` with a
    # Python lambda. transform(_rank_pct) on ~1.5M rows blows up inside
    # `_transform_general → concat → is_monotonic_increasing` with a Windows
    # access violation under pandas/Python 3.12. `groupby.rank()` is a stable
    # C-only path and equivalent in semantics.
    for src_col, dst_col in _UNIVERSE_RANK_COLS.items():
        if src_col in df.columns:
            df[dst_col] = df.groupby("ts")[src_col].rank(pct=True, method="average")
        else:
            df[dst_col] = float("nan")

    # ── Sector-level percentile ranks ──────────────────────────────────────────
    for src_col, dst_col in _SECTOR_RANK_COLS.items():
        if src_col in df.columns:
            df[dst_col] = df.groupby(["ts", "_sector"])[src_col].rank(pct=True, method="average")
        else:
            df[dst_col] = float("nan")

    # ── Relative strength vs SPY ───────────────────────────────────────────────
    if "return_20d" in df.columns:
        spy_returns = df[df["symbol"] == "SPY"][["ts", "return_20d"]].rename(
            columns={"return_20d": "_spy_ret_20d"}
        )
        if not spy_returns.empty:
            df = df.merge(spy_returns, on="ts", how="left")
            df["rs_vs_spy_20d"] = df["return_20d"] - df["_spy_ret_20d"]
            df.drop(columns=["_spy_ret_20d"], inplace=True)
        else:
            df["rs_vs_spy_20d"] = float("nan")
    else:
        df["rs_vs_spy_20d"] = float("nan")

    # ── Relative strength vs sector index ─────────────────────────────────────
    if "return_20d" in df.columns and "_sector" in df.columns:
        sector_med = (
            df.groupby(["ts", "_sector"])["return_20d"]
            .median()
            .reset_index()
            .rename(columns={"return_20d": "_sector_ret_20d"})
        )
        df = df.merge(sector_med, on=["ts", "_sector"], how="left")
        df["rs_vs_sector_20d"] = df["return_20d"] - df["_sector_ret_20d"]
        df.drop(columns=["_sector_ret_20d"], inplace=True)
    else:
        df["rs_vs_sector_20d"] = float("nan")

    df.drop(columns=["_sector"], inplace=True, errors="ignore")

    logger.debug(
        "cross_sectional_features_computed",
        n_rows=len(df),
        n_symbols=df["symbol"].nunique() if "symbol" in df.columns else 0,
    )
    return df
