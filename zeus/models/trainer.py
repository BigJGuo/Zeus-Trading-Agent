"""ModelTrainer: build dataset, train, evaluate, and save models."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.orm import Session

from zeus.backtesting.metrics import (
    hit_rate,
    ic_stability,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
)
from zeus.data.storage.database import FeaturesDaily, ModelVersion, OHLCVDaily
from zeus.models.base import BaseModel
from zeus.models.labels import compute_labels, compute_triple_barrier_labels
from zeus.models.return_predictor import XGBReturnPredictor

logger = structlog.get_logger()

_DEFAULT_GATES: dict[str, float] = {
    "information_coefficient": 0.03,
    "ic_stability": 0.30,
    "hit_rate": 0.52,
    "sharpe_on_signals": 0.80,
    "max_drawdown_on_signals": -0.25,
}

_MAX_NAN_FRACTION = 0.20


class ModelTrainer:
    def __init__(self, session: Session, artifacts_path: str) -> None:
        self.session = session
        self.artifacts_path = Path(artifacts_path)

    def build_training_dataset(
        self,
        start: date,
        end: date,
        feature_version: str,
        horizon_days: int = 5,
        label_type: str = "return",
        barrier_sigma_mult: float = 1.0,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load features and compute labels; return (X, y, metadata).

        label_type:
          - "return"    : raw forward log-return (continuous)
          - "tb"        : triple-barrier label ∈ {-1, 0, +1}; rows with label=0
                          (no barrier hit before time expiry) are dropped so
                          training/eval only see clean directional outcomes.
                          The regressor still fits a continuous target; sign()
                          at inference recovers the direction.
        """
        logger.info("building_training_dataset", start=str(start), end=str(end),
                    feature_version=feature_version, horizon_days=horizon_days,
                    label_type=label_type, barrier_sigma_mult=barrier_sigma_mult)

        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)

        feat_rows = (
            self.session.query(FeaturesDaily)
            .filter(
                FeaturesDaily.feature_version == feature_version,
                FeaturesDaily.feature_date >= start_dt,
                FeaturesDaily.feature_date <= end_dt,
            )
            .all()
        )
        if not feat_rows:
            raise ValueError(f"No features found for version={feature_version} in [{start}, {end}]")

        records = []
        for row in feat_rows:
            rec = {"symbol": row.symbol, "ts": row.feature_date}
            rec.update(row.features or {})
            records.append(rec)
        features_df = pd.DataFrame(records)
        features_df["ts"] = pd.to_datetime(features_df["ts"], utc=True)

        symbols = features_df["symbol"].unique().tolist()
        ohlcv_rows = (
            self.session.query(OHLCVDaily)
            .filter(
                OHLCVDaily.symbol.in_(symbols),
                OHLCVDaily.ts >= start_dt,
            )
            .order_by(OHLCVDaily.ts)
            .all()
        )
        ohlcv_df = pd.DataFrame(
            [{"symbol": r.symbol, "ts": r.ts, "close": r.close, "high": r.high, "low": r.low} for r in ohlcv_rows]
        )
        ohlcv_df["ts"] = pd.to_datetime(ohlcv_df["ts"], utc=True)

        if label_type == "tb":
            labels_df = compute_triple_barrier_labels(
                ohlcv_df,
                horizon_days=horizon_days,
                barrier_sigma_mult=barrier_sigma_mult,
            )
            labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)
            target_col = f"tb_label_h{horizon_days}"
            aux_cols = [f"tb_return_h{horizon_days}", f"tb_days_to_hit_h{horizon_days}"]
            merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")
            merged = merged.dropna(subset=[target_col])
            # Drop non-directional (barrier not hit) rows — this is the key step
            # that turns the task into "did the move go up or down, given it
            # moved at all?", which is what actually matters for PnL.
            n_before = len(merged)
            merged = merged[merged[target_col] != 0].reset_index(drop=True)
            logger.info("triple_barrier_filter",
                        n_before=n_before, n_after=len(merged),
                        pct_kept=round(len(merged) / max(n_before, 1), 3))
            feature_cols = [c for c in merged.columns
                            if c not in ("symbol", "ts", target_col, *aux_cols)]
        else:
            labels_df = compute_labels(ohlcv_df, horizon_days=horizon_days)
            labels_df["ts"] = pd.to_datetime(labels_df["ts"], utc=True)

            merged = features_df.merge(labels_df, on=["symbol", "ts"], how="inner")

            target_col = f"fwd_{horizon_days}d_return"
            dir_col = f"direction_{horizon_days}d"
            mae_col = f"mae_{horizon_days}d"
            merged = merged.dropna(subset=[target_col])

            feature_cols = [c for c in merged.columns if c not in ("symbol", "ts", target_col, dir_col, mae_col)]
        merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce")
        nan_fractions = merged[feature_cols].isna().mean(axis=1)
        merged = merged[nan_fractions <= _MAX_NAN_FRACTION]

        metadata = merged[["symbol", "ts"]].reset_index(drop=True)
        X = merged[feature_cols].astype("float32").reset_index(drop=True)
        y = merged[target_col].reset_index(drop=True)

        logger.info("dataset_built", n_rows=len(X), n_features=len(feature_cols))
        return X, y, metadata

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        promotion_gates: dict[str, float] | None = None,
        model_factory: Callable[[], BaseModel] | None = None,
    ) -> dict[str, Any]:
        gates = {**_DEFAULT_GATES, **(promotion_gates or {})}

        sorted_idx = metadata["ts"].argsort().to_numpy()
        n = len(sorted_idx)
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)

        train_idx = sorted_idx[:train_end]
        val_idx = sorted_idx[train_end:val_end]

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        factory: Callable[[], BaseModel] = model_factory or (lambda: XGBReturnPredictor())
        model = factory()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        val_preds = model.predict(X_val)
        val_dates = metadata["ts"].iloc[val_idx]

        ic = information_coefficient(val_preds, y_val, group=val_dates.dt.date)
        per_date_ic = _per_date_ic(val_preds, y_val, val_dates)
        stab = ic_stability(per_date_ic)
        hr = hit_rate(val_preds, y_val)

        signal_returns = _signal_returns(val_preds, y_val, val_dates)
        sharpe = sharpe_ratio(signal_returns)
        equity = (1 + signal_returns).cumprod() if not signal_returns.empty else pd.Series(dtype=float)
        mdd = max_drawdown(equity) if not equity.empty else 0.0

        metrics = {
            "information_coefficient": ic,
            "ic_stability": stab,
            "hit_rate": hr,
            "sharpe_on_signals": sharpe,
            "max_drawdown_on_signals": mdd,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }

        passed_gates = all(
            (metrics.get(k, 0) >= v if v >= 0 else metrics.get(k, 0) >= v)
            for k, v in gates.items()
            if k in metrics
        )

        ts_str = datetime.utcnow().strftime("%Y%m%d")
        param_hash = hashlib.md5(json.dumps(model.get_params(), sort_keys=True).encode()).hexdigest()[:6]
        version_string = f"v{ts_str}_{param_hash}"
        model.version_ = version_string

        logger.info("train_evaluate_complete", metrics=metrics, passed_gates=passed_gates, version=version_string)

        return {
            "model": model,
            "metrics": metrics,
            "passed_gates": passed_gates,
            "version_string": version_string,
            "gates": gates,
        }

    def save_model(
        self,
        model: BaseModel,
        metrics: dict[str, Any],
        metadata: pd.DataFrame,
        passed_gates: bool,
        version_string: str | None = None,
        model_name: str = "xgb_return",
    ) -> str:
        if version_string is None:
            ts_str = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            version_string = f"v{ts_str}"

        artifact_dir = self.artifacts_path / "models" / model_name / version_string
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model.save(artifact_dir / "model.joblib")

        model_cls = type(model)
        model_class_path = f"{model_cls.__module__}:{model_cls.__name__}"
        (artifact_dir / "model_class.txt").write_text(model_class_path)

        with open(artifact_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        config_to_save = {"__model_class__": model_class_path, **model.get_params()}
        with open(artifact_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)

        fi_df = model.get_feature_importance()
        fi_df.to_csv(artifact_dir / "feature_importance.csv", index=False)

        training_meta = {
            "version": version_string,
            "n_samples": len(metadata),
            "date_range": {
                "start": str(metadata["ts"].min()),
                "end": str(metadata["ts"].max()),
            },
            "passed_gates": passed_gates,
        }
        with open(artifact_dir / "training_metadata.json", "w") as f:
            json.dump(training_meta, f, indent=2, default=str)

        status = "staging" if passed_gates else "failed"
        ts_start = metadata["ts"].min()
        ts_end = metadata["ts"].max()

        from sqlalchemy import select

        existing = self.session.execute(
            select(ModelVersion)
            .where(ModelVersion.model_name == model_name)
            .where(ModelVersion.version == version_string)
            .limit(1)
        ).scalar_one_or_none()

        if existing is not None:
            existing.status = status
            existing.metrics = metrics
            existing.config = config_to_save
            existing.artifact_path = str(artifact_dir)
            existing.training_start_date = ts_start if pd.notna(ts_start) else None
            existing.training_end_date = ts_end if pd.notna(ts_end) else None
            existing.n_training_samples = len(metadata)
        else:
            mv = ModelVersion(
                model_name=model_name,
                version=version_string,
                status=status,
                metrics=metrics,
                config=config_to_save,
                artifact_path=str(artifact_dir),
                training_start_date=ts_start if pd.notna(ts_start) else None,
                training_end_date=ts_end if pd.notna(ts_end) else None,
                n_training_samples=len(metadata),
            )
            self.session.add(mv)
        self.session.commit()

        logger.info("model_saved", path=str(artifact_dir), status=status, version=version_string)
        return str(artifact_dir)


def _per_date_ic(preds: pd.Series, actuals: pd.Series, dates: pd.Series) -> pd.Series:
    from scipy import stats

    records = {}
    df = pd.DataFrame({"pred": preds.to_numpy(), "actual": actuals.to_numpy(), "date": dates.to_numpy()})
    for dt, grp in df.groupby("date"):
        grp = grp.dropna()
        if len(grp) < 2:
            continue
        result = stats.spearmanr(grp["pred"], grp["actual"])
        corr = float(result.statistic)  # type: ignore[union-attr]
        if not np.isnan(corr):
            records[dt] = corr
    return pd.Series(records)


def _signal_returns(preds: pd.Series, actuals: pd.Series, dates: pd.Series) -> pd.Series:
    df = pd.DataFrame({"pred": preds.values, "actual": actuals.values, "date": dates.values})
    df = df.dropna()
    daily = []
    for dt, grp in df.groupby("date"):
        if len(grp) < 5:
            continue
        threshold = grp["pred"].quantile(0.8)
        longs = grp[grp["pred"] >= threshold]["actual"]
        if longs.empty:
            continue
        daily.append({"date": dt, "return": longs.mean()})
    if not daily:
        return pd.Series(dtype=float)
    return pd.DataFrame(daily).set_index("date")["return"]
