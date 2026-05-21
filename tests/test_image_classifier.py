"""Tests for the plant health image classifier wrapper."""

from pathlib import Path

import pytest
from PIL import Image

from crop_fusion_ai.config.schemas import ImagePrediction
from crop_fusion_ai.models import PlantHealthClassifier


def test_placeholder_predict_returns_valid_image_prediction(tmp_path: Path) -> None:
    """Placeholder inference should return a valid schema for a readable image."""
    image_path = tmp_path / "leaf.jpg"
    Image.new("RGB", (16, 16), color=(120, 180, 80)).save(image_path)

    classifier = PlantHealthClassifier()
    prediction = classifier.predict(image_path)

    assert isinstance(prediction, ImagePrediction)
    assert prediction.disease_class in {
        "healthy",
        "mild_disease",
        "severe_disease",
    }
    assert 0.0 <= prediction.health_score <= 1.0
    assert 0.0 <= prediction.confidence <= 1.0


def test_placeholder_predict_rejects_missing_image(tmp_path: Path) -> None:
    """Missing image files should raise a clear file error."""
    classifier = PlantHealthClassifier()

    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        classifier.predict(tmp_path / "missing.jpg")


def test_classifier_save_and_load_metadata(tmp_path: Path) -> None:
    """Classifier metadata should round-trip through JSON."""
    model_path = tmp_path / "image_model" / "metadata.json"

    classifier = PlantHealthClassifier()
    classifier.save(model_path)
    loaded_classifier = PlantHealthClassifier(model_path=model_path)

    assert loaded_classifier.metadata.model_type == "placeholder"
    assert loaded_classifier.metadata.classes == [
        "healthy",
        "mild_disease",
        "severe_disease",
    ]
