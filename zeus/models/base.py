"""Abstract base class for all ZEUS ML models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class BaseModel(ABC):
    version_: str | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]: ...

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame: ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return joblib.load(path)
