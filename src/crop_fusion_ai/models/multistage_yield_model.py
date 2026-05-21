"""Multistage yield model using image embeddings and weather summaries."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from crop_fusion_ai.models.yield_regressor import (
    DEFAULT_YIELD_UNIT,
    RANDOM_STATE,
    _build_pipeline,
    _split_feature_columns,
)

MULTISTAGE_MODEL_NAME = "mobilenet_weather_multistage_yield_regressor"


class MultiStageYieldModel:
    """Yield regressor for MobileNet features plus weather time-series summaries."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Create or load the multistage yield model."""
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.model_name = MULTISTAGE_MODEL_NAME
        self.unit = DEFAULT_YIELD_UNIT
        if model_path is not None:
            self.load(model_path)

    def train_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """Train from image-feature, weather-summary, crop, and yield columns."""
        if target_column not in dataframe.columns:
            msg = f"Target column '{target_column}' is missing from dataframe"
            raise ValueError(msg)
        if len(dataframe) < 2:
            msg = "At least 2 rows are required for multistage model training"
            raise ValueError(msg)

        train_df = dataframe.dropna(subset=[target_column]).copy()
        features = train_df.drop(columns=[target_column])
        target = train_df[target_column].astype(float)
        self.feature_columns = list(features.columns)

        categorical_columns, numeric_columns = _split_feature_columns(features)
        self.pipeline = _build_pipeline(categorical_columns, numeric_columns)
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.25 if len(train_df) >= 8 else 0.5,
            random_state=RANDOM_STATE,
        )
        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)
        return {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }

    def fit_dataframe(self, dataframe: pd.DataFrame, target_column: str) -> None:
        """Fit the multistage model on all labelled rows without an internal split."""
        if target_column not in dataframe.columns:
            msg = f"Target column '{target_column}' is missing from dataframe"
            raise ValueError(msg)
        if len(dataframe) < 2:
            msg = "At least 2 rows are required for multistage model training"
            raise ValueError(msg)

        train_df = dataframe.dropna(subset=[target_column]).copy()
        if train_df.empty:
            msg = f"No rows remain after dropping missing target '{target_column}'"
            raise ValueError(msg)
        features = train_df.drop(columns=[target_column])
        target = train_df[target_column].astype(float)
        self.feature_columns = list(features.columns)

        categorical_columns, numeric_columns = _split_feature_columns(features)
        self.pipeline = _build_pipeline(categorical_columns, numeric_columns)
        self.pipeline.fit(features, target)

    def predict_from_feature_dict(
        self,
        features: dict[str, float | int | str],
    ) -> float:
        """Predict yield from already-extracted multistage features."""
        if self.pipeline is None:
            msg = "MultiStageYieldModel must be trained or loaded before prediction"
            raise RuntimeError(msg)
        dataframe = pd.DataFrame([features]).reindex(columns=self.feature_columns)
        prediction = float(self.pipeline.predict(dataframe)[0])
        return max(0.0, prediction)

    def save(self, model_path: Path) -> None:
        """Persist the multistage model with joblib."""
        if self.pipeline is None:
            msg = "Cannot save MultiStageYieldModel before training or loading"
            raise RuntimeError(msg)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_columns": self.feature_columns,
                "model_name": self.model_name,
                "unit": self.unit,
            },
            model_path,
        )

    def load(self, model_path: Path) -> None:
        """Load a persisted multistage model."""
        if not model_path.exists():
            msg = f"Multistage yield model file does not exist: {model_path}"
            raise FileNotFoundError(msg)
        payload: dict[str, Any] = joblib.load(model_path)
        self.pipeline = payload["pipeline"]
        self.feature_columns = list(payload["feature_columns"])
        self.model_name = str(payload.get("model_name", MULTISTAGE_MODEL_NAME))
        self.unit = str(payload.get("unit", DEFAULT_YIELD_UNIT))
