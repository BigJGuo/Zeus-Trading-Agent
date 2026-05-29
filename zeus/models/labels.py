"""Forward-return labels for supervised ML training.

Two label schemes:
  - `compute_labels`: raw forward log-return plus MAE — original approach.
  - `compute_triple_barrier_labels`: López de Prado's triple-barrier. Place an
    upper / lower barrier a multiple of realized vol above / below entry, plus
    a time expiry. Label = +1 if upper hit first, -1 if lower hit first, 0 if
    time expired without touching a barrier. The 0-labels are noise — training
    on only ±1 rows gives a much cleaner directional signal, which is the
    single biggest lever we have for pushing hit rate past ~0.55.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_labels(ohlcv_df: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    """Compute forward return labels without lookahead.

    Args:
        ohlcv_df: long-form DataFrame with [symbol, ts, close, high, low]
        horizon_days: label horizon N

    Returns:
        DataFrame with [symbol, ts, fwd_{N}d_return, direction_{N}d, mae_{N}d]
        Future rows (insufficient lookahead) get NaN labels.
    """
    N = horizon_days
    fwd_col = f"fwd_{N}d_return"
    dir_col = f"direction_{N}d"
    mae_col = f"mae_{N}d"

    results = []
    for symbol, grp in ohlcv_df.groupby("symbol", sort=False):
        grp = grp.sort_values("ts").reset_index(drop=True)
        close = grp["close"]
        low = grp["low"]

        # Forward return: log(close[t+N] / close[t])
        fwd_return = np.log(close.shift(-N) / close)

        # Direction bucketed at ±1%
        direction = pd.cut(
            fwd_return,
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=[-1, 0, 1],
        ).astype(float)

        # MAE: (min low over next N days - close) / close
        # Rolling min of future lows — shift by 1 to avoid current day
        mae = pd.Series(index=grp.index, dtype=float)
        for i in range(len(grp) - N):
            future_lows = low.iloc[i + 1 : i + N + 1]
            mae.iloc[i] = (future_lows.min() - close.iloc[i]) / close.iloc[i]
        # Last N rows get NaN (no complete future window)
        mae.iloc[len(grp) - N :] = np.nan

        sub = grp[["symbol", "ts"]].copy()
        sub[fwd_col] = fwd_return
        sub[dir_col] = direction
        sub[mae_col] = mae
        results.append(sub)

    if not results:
        cols = ["symbol", "ts", fwd_col, dir_col, mae_col]
        return pd.DataFrame(columns=cols)

    return pd.concat(results, ignore_index=True)


def compute_triple_barrier_labels(
    ohlcv_df: pd.DataFrame,
    horizon_days: int = 10,
    barrier_sigma_mult: float = 1.0,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Triple-barrier labels.

    For each (symbol, ts) row:
      - realized daily log-return vol σ over trailing `vol_window` days
      - horizon-scaled vol σ_h = σ * sqrt(horizon_days). This is the key detail:
        with daily σ, a k=1 barrier at h=5 is ~0.45 h-sigmas and ~all rows hit
        it trivially. Scaling by √h makes `barrier_sigma_mult` comparable
        across horizons and gives genuine filtering of non-directional moves.
      - place upper/lower barriers at exp(±k * σ_h) * entry
      - label is the first barrier to break within [t+1, t+horizon]

      tb_label       = +1 upper first, -1 lower first, 0 vertical (time) expiry
      tb_return      = log-return at the hit event
      tb_days_to_hit = trading days until the hit

    Output columns: `tb_label_h{N}`, `tb_return_h{N}`, `tb_days_to_hit_h{N}`.
    Rows with label=0 are the noisy, non-directional outcomes — callers typically
    drop these before training a directional model.
    """
    N = horizon_days
    lbl_col = f"tb_label_h{N}"
    ret_col = f"tb_return_h{N}"
    days_col = f"tb_days_to_hit_h{N}"
    horizon_scale = float(np.sqrt(N))

    results = []
    for symbol, grp in ohlcv_df.groupby("symbol", sort=False):
        grp = grp.sort_values("ts").reset_index(drop=True)
        close = grp["close"].astype(float).to_numpy()
        high = grp["high"].astype(float).to_numpy()
        low = grp["low"].astype(float).to_numpy()
        n = len(grp)

        log_ret = np.diff(np.log(close), prepend=np.nan)
        vol = pd.Series(log_ret).rolling(vol_window, min_periods=vol_window // 2).std().to_numpy()

        labels = np.full(n, np.nan, dtype=float)
        returns = np.full(n, np.nan, dtype=float)
        days_to_hit = np.full(n, np.nan, dtype=float)

        # Promote close/high/low to plain Python lists once so the inner loop
        # works with native floats instead of numpy scalars. Each np.log /
        # np.exp inside the hot loop allocates a 0-d numpy object; over
        # ~3000 entry points × 20 inner steps × 501 symbols × 3 horizons
        # the allocator churn on Windows reliably triggers an access
        # violation. math.log / math.exp on plain floats has no allocator.
        close_py = close.tolist()
        high_py = high.tolist()
        low_py = low.tolist()

        for i in range(n - 1):
            sigma_daily = vol[i]
            if not (math.isfinite(sigma_daily) and sigma_daily > 0):
                continue
            sigma_h = sigma_daily * horizon_scale
            entry = close_py[i]
            up_barrier = entry * math.exp(barrier_sigma_mult * sigma_h)
            dn_barrier = entry * math.exp(-barrier_sigma_mult * sigma_h)
            hit_end = min(i + N + 1, n)
            if hit_end <= i + 1:
                continue
            hit_label = 0
            hit_ix = hit_end - 1
            hit_ret = math.log(close_py[hit_ix] / entry) if entry > 0 else 0.0
            for j in range(i + 1, hit_end):
                if high_py[j] >= up_barrier:
                    hit_label = 1
                    hit_ix = j
                    hit_ret = math.log(up_barrier / entry)
                    break
                if low_py[j] <= dn_barrier:
                    hit_label = -1
                    hit_ix = j
                    hit_ret = math.log(dn_barrier / entry)
                    break
            labels[i] = hit_label
            returns[i] = hit_ret
            days_to_hit[i] = hit_ix - i

        sub = grp[["symbol", "ts"]].copy()
        sub[lbl_col] = labels
        sub[ret_col] = returns
        sub[days_col] = days_to_hit
        results.append(sub)

    if not results:
        cols = ["symbol", "ts", lbl_col, ret_col, days_col]
        return pd.DataFrame(columns=cols)

    return pd.concat(results, ignore_index=True)
