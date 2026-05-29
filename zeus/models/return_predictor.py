"""XGBoost-based return predictor."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import structlog
import xgboost as xgb

from zeus.models.base import BaseModel

logger = structlog.get_logger()


class XGBReturnPredictor(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.01,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            tree_method="hist",
            random_state=42,
        )
        self._model: xgb.XGBRegressor | None = None
        self.feature_names_: list[str] = []
        self.training_metadata_: dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> None:
        self.feature_names_ = list(X.columns)
        early_stopping = self._params["early_stopping_rounds"]

        model_kwargs = {k: v for k, v in self._params.items() if k != "early_stopping_rounds"}
        self._model = xgb.XGBRegressor(**model_kwargs, early_stopping_rounds=early_stopping)

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self._model.fit(
            X,
            y,
            eval_set=eval_set,
            verbose=False,
        )

        self.training_metadata_ = {
            "n_train": len(X),
            "n_val": len(X_val) if X_val is not None else 0,
            "best_iteration": getattr(self._model, "best_iteration", None),
            "trained_at": datetime.utcnow().isoformat(),
        }
        logger.info(
            "xgb_fit_complete",
            n_train=len(X),
            best_iteration=self.training_metadata_["best_iteration"],
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert self._model is not None, "Model not fitted"
        Xs = X.reindex(columns=self.feature_names_).apply(pd.to_numeric, errors="coerce").astype("float32")
        preds = self._model.predict(Xs)
        return pd.Series(preds, index=X.index, name="predicted_return")

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def get_feature_importance(self) -> pd.DataFrame:
        assert self._model is not None, "Model not fitted"
        scores = self._model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "feature_names": self.feature_names_,
            "params": self._params,
            "training_metadata": self.training_metadata_,
        }
        joblib.dump(payload, path)
        logger.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "XGBReturnPredictor":
        import inspect
        payload = joblib.load(path)
        init_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        init_kwargs = {k: v for k, v in payload["params"].items() if k in init_keys}
        obj = cls(**init_kwargs)
        obj._params = dict(payload["params"])
        obj._model = payload["model"]
        obj.feature_names_ = payload["feature_names"]
        obj.training_metadata_ = payload.get("training_metadata", {})
        return obj
