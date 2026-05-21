"""Tests for the baseline yield regressor."""

from pathlib import Path

import pandas as pd
import pytest

from crop_fusion_ai.config.schemas import (
    CropFeatures,
    ImagePrediction,
    WeatherFeatures,
    YieldInput,
    YieldPrediction,
)
from crop_fusion_ai.models.yield_regressor import YieldRegressor
from crop_fusion_ai.training.train_yield_model import create_synthetic_yield_dataframe


def make_yield_input() -> YieldInput:
    """Create a representative late-fusion input for tests."""
    return YieldInput(
        weather=WeatherFeatures(
            temperature_mean=25.0,
            rainfall_total=140.0,
            humidity_mean=68.0,
        ),
        crop=CropFeatures(
            crop_type="corn",
            region="01003",
            year=2022,
            planting_age_days=95,
        ),
        image_prediction=ImagePrediction(
            disease_class="healthy",
            health_score=0.9,
            confidence=0.82,
        ),
    )


def test_yield_regressor_trains_and_predicts_from_synthetic_dataframe() -> None:
    """The baseline model should train and return a valid YieldPrediction."""
    df = create_synthetic_yield_dataframe(36)
    regressor = YieldRegressor()

    metrics = regressor.train_from_dataframe(df, "yield")
    prediction = regressor.predict(make_yield_input())

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())
    assert isinstance(prediction, YieldPrediction)
    assert prediction.predicted_yield >= 0.0
    assert prediction.unit == "tonnes_per_hectare"


def test_yield_regressor_save_and_load_round_trip(tmp_path: Path) -> None:
    """A trained model should persist and load through joblib."""
    model_path = tmp_path / "yield_regressor.joblib"
    df = create_synthetic_yield_dataframe(40)

    regressor = YieldRegressor()
    regressor.train_from_dataframe(df, "yield")
    regressor.save(model_path)

    loaded_regressor = YieldRegressor(model_path=model_path)
    prediction = loaded_regressor.predict(make_yield_input())

    assert model_path.exists()
    assert prediction.model_name == "random_forest_yield_regressor"
    assert prediction.predicted_yield >= 0.0


def test_yield_regressor_rejects_missing_target_column() -> None:
    """Training should fail clearly when the target column is absent."""
    df = pd.DataFrame({"temperature_mean": [20.0, 21.0, 22.0, 23.0]})
    regressor = YieldRegressor()

    with pytest.raises(ValueError, match="Target column"):
        regressor.train_from_dataframe(df, "yield")


def test_yield_regressor_requires_training_before_prediction() -> None:
    """Prediction before training/loading should fail clearly."""
    regressor = YieldRegressor()

    with pytest.raises(RuntimeError, match="trained or loaded"):
        regressor.predict(make_yield_input())
