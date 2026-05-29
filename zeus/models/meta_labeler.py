"""Meta-labeling (Lopez de Prado): a second-stage classifier that predicts
whether the primary model's call is going to be right.

Usage pattern:
  1. Train a primary return predictor (XGB/LGB/ensemble) via walk-forward.
  2. Collect its out-of-sample predictions per (symbol, ts).
  3. For each OOS row, label = 1 if sign(pred) == sign(actual) else 0.
  4. Train this MetaLabeler on (features + primary pred) → label.
  5. At inference, gate primary predictions to zero when the classifier says
     the primary is unlikely to win. Downstream signal logic, filters, and
     portfolio construction stay unchanged — they just see fewer non-zero
     scores with a higher expected hit rate.

This is the textbook path from ~50% hit rate to 55-60% without having to
improve the primary model at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog

from zeus.models.base import BaseModel

logger = structlog.get_logger()


class MetaLabeler:
    """Binary classifier over (features + primary prediction) → primary-wins."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        num_leaves: int = 15,
        learning_rate: float = 0.03,
        min_child_samples: int = 60,
        early_stopping_rounds: int = 50,
        threshold: float = 0.55,
        is_unbalance: bool = False,
    ) -> None:
        self._params: dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            is_unbalance=is_unbalance,
        )
        self._early_stopping_rounds = early_stopping_rounds
        self.threshold_ = threshold
        self._clf: lgb.LGBMClassifier | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self.feature_names_ = list(X.columns)
        self._clf = lgb.LGBMClassifier(**self._params)
        callbacks = []
        if X_val is not None and y_val is not None and self._early_stopping_rounds:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self._early_stopping_rounds, verbose=False)
            )
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        self._clf.fit(X, y, **fit_kwargs)
        logger.info(
            "meta_fit_complete",
            n_train=len(X),
            best_iteration=getattr(self._clf, "best_iteration_", None),
            pos_rate=float(y.mean()),
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        assert self._clf is not None, "MetaLabeler not fitted"
        Xs = X.reindex(columns=self.feature_names_).apply(pd.to_numeric, errors="coerce").astype("float32")
        probs = cast(np.ndarray, self._clf.predict_proba(Xs))[:, 1]
        return pd.Series(probs, index=X.index, name="win_proba")

    def get_feature_importance(self) -> pd.DataFrame:
        assert self._clf is not None, "MetaLabeler not fitted"
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": self._clf.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self._clf,
                "feature_names": self.feature_names_,
                "params": self._params,
                "threshold": self.threshold_,
                "early_stopping_rounds": self._early_stopping_rounds,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabeler":
        payload = joblib.load(path)
        obj = cls(threshold=payload.get("threshold", 0.55))
        obj._params = payload["params"]
        obj._early_stopping_rounds = payload.get("early_stopping_rounds", 50)
        obj._clf = payload["clf"]
        obj.feature_names_ = payload["feature_names"]
        return obj


def build_meta_dataset(
    features_df: pd.DataFrame,
    oos_preds_df: pd.DataFrame,
    include_pred_rank: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Join primary OOS predictions with their corresponding feature rows.

    Returns (X_meta, y_meta, metadata). X_meta contains the original feature
    columns plus `primary_pred` and `primary_pred_abs`, optionally
    `primary_pred_rank_in_date`. y_meta is the 0/1 "did the primary win" label.
    """
    preds = oos_preds_df[["symbol", "ts", "pred", "actual"]].copy()
    preds["ts"] = pd.to_datetime(preds["ts"], utc=True)
    feats = features_df.copy()
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)

    merged = feats.merge(preds, on=["symbol", "ts"], how="inner")
    merged = merged.dropna(subset=["pred", "actual"])

    y = (np.sign(merged["pred"]) == np.sign(merged["actual"])).astype(int)
    y = y.where(merged["actual"] != 0, 0)  # treat exactly-zero actuals as a miss

    merged["primary_pred"] = merged["pred"]
    merged["primary_pred_abs"] = merged["pred"].abs()
    if include_pred_rank:
        merged["primary_pred_rank_in_date"] = (
            merged.groupby(merged["ts"].dt.date)["pred"].rank(pct=True)
        )

    meta_cols_drop = {"symbol", "ts", "pred", "actual"}
    feat_cols = [c for c in merged.columns if c not in meta_cols_drop]
    X = merged[feat_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    metadata = merged[["symbol", "ts", "pred", "actual"]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y, metadata


class MetaGatedPredictor(BaseModel):
    """Wraps a primary regressor and a MetaLabeler; zeroes out predictions
    whose win-probability falls below the learned threshold.

    Implements BaseModel so it can drop into SignalGenerator / trainer / loader
    without any of them knowing meta-labeling exists.
    """

    def __init__(
        self,
        primary: BaseModel,
        meta: MetaLabeler,
        threshold: float | None = None,
    ) -> None:
        self._primary = primary
        self._meta = meta
        self._threshold = threshold if threshold is not None else meta.threshold_
        self.feature_names_: list[str] = list(getattr(primary, "feature_names_", []) or [])
        self.version_ = getattr(primary, "version_", None)

    def fit(self, X, y, **kwargs) -> None:  # pragma: no cover - not used; components pre-fit
        raise NotImplementedError("Fit the primary and meta separately; MetaGatedPredictor composes pre-fit components.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        primary_preds = self._primary.predict(X).astype("float64")
        meta_X = X.copy()
        meta_X["primary_pred"] = primary_preds.values
        meta_X["primary_pred_abs"] = primary_preds.abs().values
        # Cross-sectional rank is date-based; at inference we typically pass one
        # date's rows at once, so rank over the whole input is the right proxy.
        meta_X["primary_pred_rank_in_date"] = primary_preds.rank(pct=True).values
        proba = self._meta.predict_proba(meta_X)
        gated = primary_preds.where(proba >= self._threshold, 0.0)
        gated.name = "predicted_return"
        return gated

    def get_params(self) -> dict[str, Any]:
        return {
            "primary_cls": f"{type(self._primary).__module__}:{type(self._primary).__name__}",
            "primary_params": self._primary.get_params(),
            "meta_params": self._meta._params,
            "meta_threshold": self._threshold,
        }

    def get_feature_importance(self) -> pd.DataFrame:
        try:
            return self._primary.get_feature_importance()
        except Exception:
            return pd.DataFrame(columns=["feature", "importance"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent
        primary_path = parent / "primary.joblib"
        meta_path = parent / "meta.joblib"
        self._primary.save(primary_path)
        self._meta.save(meta_path)
        joblib.dump(
            {
                "primary_cls": f"{type(self._primary).__module__}:{type(self._primary).__name__}",
                "primary_path": str(primary_path),
                "meta_path": str(meta_path),
                "threshold": self._threshold,
                "feature_names": self.feature_names_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MetaGatedPredictor":
        import importlib

        payload = joblib.load(path)
        module_name, cls_name = payload["primary_cls"].split(":", 1)
        primary_cls = getattr(importlib.import_module(module_name), cls_name)
        primary = primary_cls.load(payload["primary_path"])
        meta = MetaLabeler.load(payload["meta_path"])
        obj = cls(primary=primary, meta=meta, threshold=payload["threshold"])
        obj.feature_names_ = payload.get("feature_names", list(getattr(primary, "feature_names_", []) or []))
        return obj
