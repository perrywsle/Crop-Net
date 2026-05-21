"""Baseline yield prediction model for fused CropNet-style features."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from crop_fusion_ai.config.schemas import YieldInput, YieldPrediction
from crop_fusion_ai.data.fusion_features import create_fusion_feature_dict

DEFAULT_MODEL_NAME = "random_forest_yield_regressor"
DEFAULT_YIELD_UNIT = "tonnes_per_hectare"
RANDOM_STATE = 42


class YieldRegressor:
    """RandomForest baseline for tabular late-fusion yield prediction."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Create the regressor and optionally load a saved model."""
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.model_name = DEFAULT_MODEL_NAME
        self.unit = DEFAULT_YIELD_UNIT

        if model_path is not None:
            self.load(model_path)

    def train_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """Train the baseline regressor from a tabular dataframe."""
        self.fit_dataframe(df, target_column)
        return self.evaluate_dataframe(df, target_column)

    def fit_dataframe(self, df: pd.DataFrame, target_column: str) -> None:
        """Fit the regressor on the provided dataframe without a holdout split."""
        if target_column not in df.columns:
            msg = f"Target column '{target_column}' is missing from dataframe"
            raise ValueError(msg)
        if len(df) < 4:
            msg = "At least 4 rows are required to train the yield regressor"
            raise ValueError(msg)

        train_df = df.dropna(subset=[target_column]).copy()
        if train_df.empty:
            msg = f"No rows remain after dropping missing target '{target_column}'"
            raise ValueError(msg)

        features = train_df.drop(columns=[target_column])
        target = train_df[target_column].astype(float)
        self.feature_columns = list(features.columns)
        categorical_columns, numeric_columns = _split_feature_columns(features)
        self.pipeline = _build_pipeline(categorical_columns, numeric_columns)
        self.pipeline.fit(features, target)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """Evaluate the currently configured regressor on a labelled dataframe."""
        if self.pipeline is None:
            msg = "YieldRegressor must be trained or loaded before evaluation"
            raise RuntimeError(msg)
        if target_column not in df.columns:
            msg = f"Target column '{target_column}' is missing from dataframe"
            raise ValueError(msg)

        eval_df = df.dropna(subset=[target_column]).copy()
        if eval_df.empty:
            msg = f"No rows remain after dropping missing target '{target_column}'"
            raise ValueError(msg)

        features = eval_df.drop(columns=[target_column])
        features = features.reindex(columns=self.feature_columns)
        target = eval_df[target_column].astype(float)
        predictions = self.pipeline.predict(features)

        metrics = {
            "mae": float(mean_absolute_error(target, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
            "r2": float(r2_score(target, predictions)),
        }
        return metrics

    def train_with_holdout_split(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """Train on an internal random holdout split for quick demos and tests."""
        if target_column not in df.columns:
            msg = f"Target column '{target_column}' is missing from dataframe"
            raise ValueError(msg)
        if len(df) < 4:
            msg = "At least 4 rows are required to train/test the yield regressor"
            raise ValueError(msg)

        train_df = df.dropna(subset=[target_column]).copy()
        if train_df.empty:
            msg = f"No rows remain after dropping missing target '{target_column}'"
            raise ValueError(msg)

        features = train_df.drop(columns=[target_column])
        target = train_df[target_column].astype(float)
        self.feature_columns = list(features.columns)
        categorical_columns, numeric_columns = _split_feature_columns(features)
        self.pipeline = _build_pipeline(categorical_columns, numeric_columns)

        test_size = 0.25 if len(train_df) >= 8 else 0.5
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }
        return metrics

    def predict(self, input_data: YieldInput) -> YieldPrediction:
        """Predict crop yield from a validated late-fusion input object."""
        if self.pipeline is None:
            msg = "YieldRegressor must be trained or loaded before prediction"
            raise RuntimeError(msg)

        feature_dict = create_fusion_feature_dict(input_data)
        features = pd.DataFrame([feature_dict])
        features = features.reindex(columns=self.feature_columns)
        predicted_yield = float(self.pipeline.predict(features)[0])

        warnings: list[str] = []
        if predicted_yield < 0.0:
            warnings.append("Model predicted a negative yield; clipped to 0.0.")
            predicted_yield = 0.0

        return YieldPrediction(
            predicted_yield=predicted_yield,
            unit=self.unit,
            model_name=self.model_name,
            warnings=warnings,
        )

    def save(self, model_path: Path) -> None:
        """Persist the trained yield model with joblib."""
        if self.pipeline is None:
            msg = "Cannot save YieldRegressor before training or loading a model"
            raise RuntimeError(msg)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "model_name": self.model_name,
            "unit": self.unit,
        }
        joblib.dump(payload, model_path)

    def load(self, model_path: Path) -> None:
        """Load a persisted yield model from joblib."""
        if not model_path.exists():
            msg = f"Yield model file does not exist: {model_path}"
            raise FileNotFoundError(msg)

        payload: dict[str, Any] = joblib.load(model_path)
        self.pipeline = payload["pipeline"]
        self.feature_columns = list(payload["feature_columns"])
        self.model_name = str(payload.get("model_name", DEFAULT_MODEL_NAME))
        self.unit = str(payload.get("unit", DEFAULT_YIELD_UNIT))


def _split_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split dataframe columns into categorical and numeric feature lists."""
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(str(column))
        else:
            categorical_columns.append(str(column))

    return categorical_columns, numeric_columns


def _build_pipeline(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> Pipeline:
    """Build a preprocessing and RandomForest regression pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    regressor = RandomForestRegressor(
        n_estimators=80,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )
