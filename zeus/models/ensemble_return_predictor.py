"""Ensemble return predictor: averages predictions from multiple base models.

Used for Option D — XGB + LightGBM average. The two gradient-boosting libraries
disagree on different feature subsets (leaf-wise vs level-wise growth, different
categorical handling, different regularization paths), so averaging reliably
reduces variance and lifts IC by 1-3pp without changing the feature set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import structlog

from zeus.models.base import BaseModel

logger = structlog.get_logger()


class EnsembleReturnPredictor(BaseModel):
    def __init__(self, base_models: Sequence[BaseModel], weights: Sequence[float] | None = None) -> None:
        if not base_models:
            raise ValueError("EnsembleReturnPredictor requires at least one base model")
        self._base_models = list(base_models)
        if weights is None:
            self._weights = [1.0 / len(base_models)] * len(base_models)
        else:
            if len(weights) != len(base_models):
                raise ValueError("weights length must match base_models length")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("weights must sum to > 0")
            self._weights = [w / total for w in weights]
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> None:
        self.feature_names_ = list(X.columns)
        for i, m in enumerate(self._base_models):
            logger.info("ensemble_fit_member", idx=i, cls=type(m).__name__)
            m.fit(X, y, X_val=X_val, y_val=y_val)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds_stack = None
        for i, (m, w) in enumerate(zip(self._base_models, self._weights)):
            p = m.predict(X).astype("float64")
            if preds_stack is None:
                preds_stack = p * w
            else:
                preds_stack = preds_stack.add(p * w, fill_value=0.0)
        assert preds_stack is not None
        preds_stack.name = "predicted_return"
        return preds_stack

    def get_params(self) -> dict[str, Any]:
        return {
            "ensemble_members": [
                {"cls": type(m).__name__, "params": m.get_params(), "weight": w}
                for m, w in zip(self._base_models, self._weights)
            ]
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Weighted average of per-member importances, normalised per member first."""
        merged: dict[str, float] = {}
        for m, w in zip(self._base_models, self._weights):
            try:
                fi = m.get_feature_importance()
            except Exception:
                continue
            if fi.empty:
                continue
            total = fi["importance"].sum()
            if total <= 0:
                continue
            for _, row in fi.iterrows():
                key = str(row["feature"])
                merged[key] = merged.get(key, 0.0) + w * (float(row["importance"]) / total)
        if not merged:
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame(
                [{"feature": k, "importance": v} for k, v in merged.items()]
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parent_dir = path.parent
        member_paths = []
        for i, m in enumerate(self._base_models):
            sub = parent_dir / f"member_{i}_{type(m).__name__}.joblib"
            m.save(sub)
            member_paths.append({"cls": type(m).__name__, "module": type(m).__module__, "path": str(sub)})
        payload = {
            "members": member_paths,
            "weights": self._weights,
            "feature_names": self.feature_names_,
        }
        joblib.dump(payload, path)
        logger.info("ensemble_saved", path=str(path), n_members=len(self._base_models))

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleReturnPredictor":
        import importlib

        payload = joblib.load(path)
        base_models = []
        for m in payload["members"]:
            module = importlib.import_module(m["module"])
            klass = getattr(module, m["cls"])
            base_models.append(klass.load(m["path"]))
        obj = cls(base_models=base_models, weights=payload["weights"])
        obj.feature_names_ = payload.get("feature_names", [])
        return obj
