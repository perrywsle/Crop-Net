"""Tests for late-fusion feature engineering helpers."""

import pytest

from crop_fusion_ai.config.schemas import (
    CropFeatures,
    ImagePrediction,
    WeatherFeatures,
    YieldInput,
)
from crop_fusion_ai.data.fusion_features import (
    create_fusion_feature_dict,
    encode_basic_crop_type,
    validate_health_score_mapping,
)


def test_create_fusion_feature_dict_flattens_nested_input() -> None:
    """The fusion dictionary should expose tabular and image-derived features."""
    input_data = YieldInput(
        weather=WeatherFeatures(
            temperature_mean=28.4,
            temperature_min=23.1,
            temperature_max=34.0,
            rainfall_total=180.0,
            humidity_mean=76.0,
            solar_radiation_mean=None,
        ),
        crop=CropFeatures(
            crop_type="Rice",
            region="Sarawak",
            year=2026,
            planting_age_days=105,
        ),
        image_prediction=ImagePrediction(
            disease_class="healthy",
            health_score=0.98,
            confidence=0.93,
        ),
    )

    features = create_fusion_feature_dict(input_data)

    assert features["temperature_mean"] == pytest.approx(28.4)
    assert features["crop_type"] == "Rice"
    assert features["crop_type_encoded"] == 1
    assert features["health_score"] == pytest.approx(0.98)
    assert features["image_confidence"] == pytest.approx(0.93)
    assert features["disease_class_health_score"] == pytest.approx(1.0)
    assert features["region"] == "Sarawak"
    assert "solar_radiation_mean" not in features


@pytest.mark.parametrize(
    ("disease_class", "expected_score"),
    [
        ("healthy", 1.0),
        ("mild_rust", 0.75),
        ("moderate leaf spot", 0.5),
        ("severe-blight", 0.25),
        ("unseen disease label", 0.5),
    ],
)
def test_validate_health_score_mapping_maps_common_labels(
    disease_class: str,
    expected_score: float,
) -> None:
    """Disease labels should map to stable health-score estimates."""
    assert validate_health_score_mapping(disease_class) == pytest.approx(expected_score)


def test_validate_health_score_mapping_rejects_blank_label() -> None:
    """Blank disease labels are invalid."""
    with pytest.raises(ValueError, match="disease_class"):
        validate_health_score_mapping("  ")


@pytest.mark.parametrize(
    ("crop_type", "expected_encoding"),
    [
        ("rice", 1),
        ("Maize", 2),
        ("corn", 2),
        ("soybeans", 4),
        ("dragon fruit", 0),
    ],
)
def test_encode_basic_crop_type_returns_stable_codes(
    crop_type: str,
    expected_encoding: int,
) -> None:
    """Common crop names should map to stable integer codes."""
    assert encode_basic_crop_type(crop_type) == expected_encoding


def test_encode_basic_crop_type_rejects_blank_label() -> None:
    """Blank crop labels are invalid."""
    with pytest.raises(ValueError, match="crop_type"):
        encode_basic_crop_type("")
