"""Tests for the end-to-end fusion inference pipeline."""

from pathlib import Path

import pytest
from PIL import Image

from crop_fusion_ai.config.schemas import CropFeatures, WeatherFeatures
from crop_fusion_ai.inference.pipeline import CropFusionPipeline
from crop_fusion_ai.models import YieldRegressor
from crop_fusion_ai.training.train_yield_model import create_synthetic_yield_dataframe


def save_synthetic_yield_model(model_path: Path) -> None:
    """Train and save a small synthetic yield model for pipeline tests."""
    regressor = YieldRegressor()
    regressor.train_from_dataframe(create_synthetic_yield_dataframe(36), "yield")
    regressor.save(model_path)


def create_temp_image(image_path: Path, color: tuple[int, int, int]) -> None:
    """Create a small RGB test image."""
    Image.new("RGB", (16, 16), color=color).save(image_path)


def test_pipeline_returns_image_and_yield_predictions(tmp_path: Path) -> None:
    """Pipeline should connect placeholder image inference to loaded yield model."""
    image_path = tmp_path / "leaf.jpg"
    model_path = tmp_path / "yield_regressor.joblib"
    create_temp_image(image_path, color=(160, 200, 120))
    save_synthetic_yield_model(model_path)

    pipeline = CropFusionPipeline(yield_model_path=model_path)
    image_prediction, segmentation_result, yield_prediction = (
        pipeline.predict_from_image_and_features(
            image_path=image_path,
            weather=WeatherFeatures(
                temperature_mean=25.0,
                rainfall_total=150.0,
                humidity_mean=70.0,
            ),
            crop=CropFeatures(
                crop_type="corn",
                region="01003",
                year=2022,
                planting_age_days=90,
            ),
        )
    )

    assert image_prediction.disease_class in {
        "healthy",
        "mild_disease",
        "severe_disease",
    }
    assert segmentation_result.overlay_path.exists()
    assert yield_prediction.predicted_yield >= 0.0
    assert any("placeholder inference" in item for item in yield_prediction.warnings)


def test_pipeline_warns_for_low_health_and_low_confidence(tmp_path: Path) -> None:
    """Dark placeholder image should produce stress and confidence warnings."""
    image_path = tmp_path / "dark_leaf.jpg"
    model_path = tmp_path / "yield_regressor.joblib"
    create_temp_image(image_path, color=(20, 20, 20))
    save_synthetic_yield_model(model_path)

    pipeline = CropFusionPipeline(yield_model_path=model_path)
    _, _, yield_prediction = pipeline.predict_from_image_and_features(
        image_path=image_path,
        weather=WeatherFeatures(temperature_mean=29.0, rainfall_total=80.0),
        crop=CropFeatures(crop_type="soybean", region="17019", year=2021),
    )

    assert any("health score is low" in item for item in yield_prediction.warnings)
    assert any("confidence is low" in item for item in yield_prediction.warnings)


def test_pipeline_raises_clear_error_when_yield_model_untrained(tmp_path: Path) -> None:
    """Pipeline should fail clearly when no yield model path is provided."""
    image_path = tmp_path / "leaf.jpg"
    create_temp_image(image_path, color=(160, 200, 120))
    pipeline = CropFusionPipeline()

    with pytest.raises(RuntimeError, match="Yield model is untrained"):
        pipeline.predict_from_image_and_features(
            image_path=image_path,
            weather=WeatherFeatures(temperature_mean=25.0, rainfall_total=120.0),
            crop=CropFeatures(crop_type="corn", year=2022),
        )
