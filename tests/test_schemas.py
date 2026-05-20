"""Tests for Pydantic data-layer schemas."""

import pytest
from pydantic import ValidationError

from crop_fusion_ai.config.schemas import (
    CropFeatures,
    ImagePrediction,
    WeatherFeatures,
    YieldInput,
    YieldPrediction,
)


def test_yield_input_accepts_nested_valid_features() -> None:
    """YieldInput should accept the complete multimodal feature payload."""
    payload = YieldInput(
        weather=WeatherFeatures(
            temperature_mean=27.5,
            temperature_min=22.0,
            temperature_max=33.0,
            rainfall_total=148.2,
            humidity_mean=82.0,
            solar_radiation_mean=18.5,
        ),
        crop=CropFeatures(
            crop_type="rice",
            region="Sarawak",
            year=2026,
            planting_age_days=90,
        ),
        image_prediction=ImagePrediction(
            disease_class="healthy",
            health_score=0.95,
            confidence=0.91,
        ),
    )

    assert payload.weather.rainfall_total == pytest.approx(148.2)
    assert payload.crop.crop_type == "rice"
    assert payload.image_prediction.confidence == pytest.approx(0.91)


def test_image_prediction_rejects_scores_outside_unit_interval() -> None:
    """Image prediction scores must stay between 0 and 1."""
    with pytest.raises(ValidationError):
        ImagePrediction(
            disease_class="leaf_blight",
            health_score=1.2,
            confidence=0.8,
        )


def test_weather_features_reject_negative_rainfall() -> None:
    """Rainfall cannot be negative."""
    with pytest.raises(ValidationError):
        WeatherFeatures(
            temperature_mean=29.0,
            rainfall_total=-1.0,
        )


def test_crop_features_reject_negative_planting_age() -> None:
    """Planting age must be positive when provided."""
    with pytest.raises(ValidationError):
        CropFeatures(
            crop_type="maize",
            year=2026,
            planting_age_days=-5,
        )


def test_yield_prediction_defaults_to_empty_warnings() -> None:
    """YieldPrediction should not require warnings for normal predictions."""
    prediction = YieldPrediction(
        predicted_yield=4.2,
        unit="tonnes_per_hectare",
        model_name="baseline_fusion_demo",
    )

    assert prediction.warnings == []
