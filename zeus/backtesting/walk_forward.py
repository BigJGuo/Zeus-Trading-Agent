"""Walk-forward cross-validation with embargo to prevent leakage."""
from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
import structlog
from dateutil.relativedelta import relativedelta

from zeus.backtesting.metrics import (
    bootstrapped_ic,
    hit_rate,
    information_coefficient,
    sharpe_ratio,
)

logger = structlog.get_logger()

_EMBARGO_TRADING_DAYS = 20


class WalkForwardValidator:
    def __init__(
        self,
        train_window_months: int = 18,
        validation_window_months: int = 3,
        step_months: int = 1,
    ) -> None:
        self.train_window_months = train_window_months
        self.validation_window_months = validation_window_months
        self.step_months = step_months

    def generate_folds(
        self, dates: pd.DatetimeIndex
    ) -> list[tuple[date, date, date, date]]:
        """Return (train_start, train_end, test_start, test_end) for each fold."""
        sorted_dates = sorted(dates.normalize().unique())
        if not sorted_dates:
            return []

        first_date = sorted_dates[0].date()
        last_date = sorted_dates[-1].date()

        folds = []
        train_start = first_date
        while True:
            train_end = train_start + relativedelta(months=self.train_window_months) - relativedelta(days=1)
            # Apply embargo: skip _EMBARGO_TRADING_DAYS after train_end
            trading_dates = [d for d in sorted_dates if d.date() > train_end]
            if len(trading_dates) < _EMBARGO_TRADING_DAYS + 1:
                break
            test_start = trading_dates[_EMBARGO_TRADING_DAYS].date()
            test_end = test_start + relativedelta(months=self.validation_window_months) - relativedelta(days=1)
            if test_end > last_date:
                break
            folds.append((train_start, train_end, test_start, test_end))
            train_start = train_start + relativedelta(months=self.step_months)

        return folds

    def run(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        model_factory: Callable,
    ) -> pd.DataFrame:
        """Run walk-forward validation and return per-fold metrics DataFrame."""
        merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")
        merged = merged.sort_values("ts").reset_index(drop=True)

        dates = pd.DatetimeIndex(merged["ts"])
        folds = self.generate_folds(dates)
        if not folds:
            logger.warning("no_folds_generated")
            return pd.DataFrame()

        label_cols = [c for c in labels_df.columns if c not in ("symbol", "ts")]
        feature_cols = [c for c in features_df.columns if c not in ("symbol", "ts")]
        target_col = label_cols[0] if label_cols else None

        records = []
        for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
            tr_mask = (merged["ts"].dt.date >= tr_start) & (merged["ts"].dt.date <= tr_end)
            te_mask = (merged["ts"].dt.date >= te_start) & (merged["ts"].dt.date <= te_end)

            train = merged[tr_mask].dropna(subset=feature_cols + ([target_col] if target_col else []))
            test = merged[te_mask].dropna(subset=feature_cols)

            if train.empty or test.empty:
                logger.warning("empty_fold", fold=fold_idx)
                continue

            X_train = train[feature_cols]
            y_train = train[target_col] if target_col else pd.Series(dtype=float)
            X_test = test[feature_cols]
            y_test = test[target_col] if target_col and target_col in test.columns else pd.Series(dtype=float)

            model = model_factory()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            group = test["ts"].dt.date if len(test) > 0 else None
            # Bootstrap-resampled IC gives us a 5th-percentile lower bound
            # alongside the point estimate. Gating on the LCB filters out
            # models whose IC point estimate is propped up by a few lucky
            # dates — those collapse on resampling.
            ic_stats = bootstrapped_ic(
                pd.Series(predictions).reset_index(drop=True),
                pd.Series(y_test.values if hasattr(y_test, "values") else y_test).reset_index(drop=True),
                group=pd.Series(group).reset_index(drop=True) if group is not None else None,
            )
            fold_ic = ic_stats["ic"]
            fold_hit = hit_rate(predictions, y_test)

            # Sharpe of equal-weight long-top-quintile strategy
            signal_returns = _compute_signal_returns(predictions, y_test, test["ts"])
            fold_sharpe = sharpe_ratio(signal_returns) if not signal_returns.empty else 0.0

            record = {
                "fold": fold_idx,
                "train_start": tr_start,
                "train_end": tr_end,
                "test_start": te_start,
                "test_end": te_end,
                "n_train": len(train),
                "n_test": len(test),
                "ic": fold_ic,
                "ic_lcb": ic_stats["ic_lcb"],
                "ic_mean": ic_stats["ic_mean"],
                "ic_std": ic_stats["ic_std"],
                "hit_rate": fold_hit,
                "sharpe_on_signals": fold_sharpe,
            }
            records.append(record)
            logger.info("fold_complete", **{k: str(v) if isinstance(v, date) else v for k, v in record.items()})

        return pd.DataFrame(records)


def _compute_signal_returns(
    predictions: pd.Series,
    actuals: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    """Daily equal-weight long top-quintile signal returns."""
    df = pd.DataFrame({"pred": predictions, "actual": actuals, "date": dates.values})
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float)

    daily_returns = []
    for dt, grp in df.groupby("date"):
        if len(grp) < 5:
            continue
        threshold = grp["pred"].quantile(0.8)
        longs = grp[grp["pred"] >= threshold]["actual"]
        if longs.empty:
            continue
        daily_returns.append({"date": dt, "return": longs.mean()})

    if not daily_returns:
        return pd.Series(dtype=float)

    ret_df = pd.DataFrame(daily_returns).set_index("date")
    return ret_df["return"]
