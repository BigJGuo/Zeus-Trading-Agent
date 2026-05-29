"""Pure performance and signal quality metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    predictions: pd.Series,
    actuals: pd.Series,
    group: pd.Series | None = None,
) -> float:
    """Spearman rank IC, optionally mean of per-group ICs."""
    if group is None:
        corr = float(stats.spearmanr(predictions, actuals, nan_policy="omit").statistic)  # type: ignore[union-attr]
        return corr if not np.isnan(corr) else 0.0

    per_group = []
    for _, idx in actuals.groupby(group).groups.items():
        p = predictions.loc[idx].dropna()
        a = actuals.loc[idx].dropna()
        common = p.index.intersection(a.index)
        if len(common) < 2:
            continue
        corr = float(stats.spearmanr(p.loc[common], a.loc[common]).statistic)  # type: ignore[union-attr]
        if not np.isnan(corr):
            per_group.append(corr)

    return float(np.mean(per_group)) if per_group else 0.0


def ic_stability(per_date_ic: pd.Series) -> float:
    """IC Information Ratio: mean / std."""
    std = per_date_ic.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(per_date_ic.mean() / std)


def hit_rate(predictions: pd.Series, actuals: pd.Series) -> float:
    """Fraction of samples where predicted and actual direction agree."""
    mask = predictions.notna() & actuals.notna()
    p = predictions[mask]
    a = actuals[mask]
    if len(p) == 0:
        return 0.0
    return float((np.sign(p) == np.sign(a)).mean())


def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Annualised Sharpe ratio."""
    clean = returns.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return 0.0
    return float(clean.mean() / clean.std() * np.sqrt(freq))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a negative fraction."""
    clean = equity.dropna()
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    dd = (clean - peak) / peak
    return float(dd.min())


def sortino_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    downside = clean[clean < 0]
    if len(downside) == 0:
        return float("inf")
    dd_std = downside.std()
    if dd_std == 0:
        return 0.0
    return float(clean.mean() / dd_std * np.sqrt(freq))


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return / absolute max drawdown."""
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    equity = (1 + clean).cumprod()
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    annual_ret = clean.mean() * 252
    return float(annual_ret / abs(mdd))


def win_rate(trade_pnls: pd.Series) -> float:
    """Fraction of trades with positive PnL."""
    clean = trade_pnls.dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profits / gross losses (absolute)."""
    clean = trade_pnls.dropna()
    gross_profit = clean[clean > 0].sum()
    gross_loss = abs(clean[clean < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def bootstrapped_ic(
    predictions: pd.Series,
    actuals: pd.Series,
    group: pd.Series | None = None,
    *,
    n_resamples: int = 1000,
    lcb_percentile: float = 5.0,
    seed: int | None = 42,
) -> dict:
    """Bootstrap the Spearman IC and return point estimate + LCB.

    Gating models on a point-estimate IC accepts a model that beat the bar
    once on a single sample. Resampling the rows (with replacement) and
    re-computing IC gives us the distribution under sampling variation —
    the 5th-percentile lower bound is the IC the model would still clear
    even on a noticeably worse sample, which is the bar we actually care
    about for promotion to production.

    When `group` is provided (typical: per-date), resampling happens at the
    *group* level: each bootstrap sample picks N dates with replacement,
    then computes per-date ICs the same way `information_coefficient`
    does. This is the right unit of resampling for cross-sectional rank
    forecasts — resampling individual rows would understate the variance.

    Returns a dict:
      `ic`            — point estimate (same as information_coefficient)
      `ic_lcb`        — `lcb_percentile`-th percentile of bootstrap samples
      `ic_mean`       — mean across bootstrap samples
      `ic_std`        — std across bootstrap samples
      `n_resamples`   — echo of input
    """
    point = information_coefficient(predictions, actuals, group=group)

    rng = np.random.default_rng(seed)

    if group is not None:
        # Per-group IC distribution. Resample groups (dates) with replacement
        # and take the mean per-group IC each time — matches how the
        # point-estimate is computed.
        per_group_ic: list[float] = []
        for _, idx in actuals.groupby(group).groups.items():
            p = predictions.loc[idx].dropna()
            a = actuals.loc[idx].dropna()
            common = p.index.intersection(a.index)
            if len(common) < 2:
                continue
            corr = float(stats.spearmanr(p.loc[common], a.loc[common]).statistic)  # type: ignore[union-attr]
            if not np.isnan(corr):
                per_group_ic.append(corr)
        if not per_group_ic:
            return {
                "ic": point, "ic_lcb": 0.0, "ic_mean": 0.0, "ic_std": 0.0,
                "n_resamples": n_resamples,
            }
        per_group_arr = np.asarray(per_group_ic)
        samples = rng.choice(per_group_arr, size=(n_resamples, len(per_group_arr)), replace=True)
        boot_means = samples.mean(axis=1)
    else:
        # Row-level resampling — appropriate when the call site has no
        # natural group (e.g. single-cross-section eval).
        mask = predictions.notna() & actuals.notna()
        p = predictions[mask].to_numpy()
        a = actuals[mask].to_numpy()
        if len(p) < 2:
            return {
                "ic": point, "ic_lcb": 0.0, "ic_mean": 0.0, "ic_std": 0.0,
                "n_resamples": n_resamples,
            }
        n = len(p)
        boot_means = np.empty(n_resamples, dtype=float)
        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            corr = float(stats.spearmanr(p[idx], a[idx]).statistic)  # type: ignore[union-attr]
            boot_means[i] = corr if not np.isnan(corr) else 0.0

    lcb = float(np.percentile(boot_means, lcb_percentile))
    return {
        "ic": float(point),
        "ic_lcb": lcb,
        "ic_mean": float(boot_means.mean()),
        "ic_std": float(boot_means.std()),
        "n_resamples": n_resamples,
    }
