"""Tests for multistage MobileNet/weather-style yield modeling."""

from crop_fusion_ai.data.weather_timeseries import summarize_weather_sequence
from crop_fusion_ai.models.multistage_yield_model import MultiStageYieldModel
from crop_fusion_ai.training.train_multistage_demo import (
    create_synthetic_multistage_dataframe,
)


def test_summarize_weather_sequence_creates_fixed_features() -> None:
    """Weather sequence reducer should summarize prior weather records."""
    features = summarize_weather_sequence(
        [
            {"temperature": 20.0, "rainfall": 5.0, "humidity": 70.0},
            {"temperature": 22.0, "rainfall": 0.0, "humidity": 75.0},
        ]
    )

    assert features["weather_steps"] == 2.0
    assert features["temperature_mean"] == 21.0
    assert features["rainfall_sum"] == 5.0
    assert features["humidity_max"] == 75.0


def test_multistage_yield_model_trains_and_predicts() -> None:
    """Multistage model should train on image/weather/yield-shaped rows."""
    dataframe = create_synthetic_multistage_dataframe(
        sample_count=36,
        image_feature_count=8,
    )
    model = MultiStageYieldModel()

    metrics = model.train_from_dataframe(dataframe, "yield")
    prediction = model.predict_from_feature_dict(
        dataframe.drop(columns=["yield"]).iloc[0].to_dict()
    )

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert prediction >= 0.0
