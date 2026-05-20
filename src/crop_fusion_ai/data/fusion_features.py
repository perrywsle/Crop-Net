"""Feature engineering helpers for late-fusion yield prediction inputs."""

from crop_fusion_ai.config.schemas import YieldInput

FusionFeatureValue = float | str | int

_CROP_TYPE_ENCODING: dict[str, int] = {
    "rice": 1,
    "maize": 2,
    "corn": 2,
    "wheat": 3,
    "soybean": 4,
    "soybeans": 4,
    "cassava": 5,
    "potato": 6,
    "tomato": 7,
}

_DISEASE_HEALTH_SCORE_MAPPING: dict[str, float] = {
    "healthy": 1.0,
    "no disease": 1.0,
    "none": 1.0,
    "mild": 0.75,
    "moderate": 0.5,
    "severe": 0.25,
    "critical": 0.1,
}


def create_fusion_feature_dict(input_data: YieldInput) -> dict[str, FusionFeatureValue]:
    """Flatten validated multimodal input into model-ready fusion features.

    Optional values are omitted when absent so the returned feature dictionary
    contains only primitive values that can be passed directly to simple tabular
    model pipelines.
    """
    weather = input_data.weather
    crop = input_data.crop
    image_prediction = input_data.image_prediction

    features: dict[str, FusionFeatureValue] = {
        "temperature_mean": weather.temperature_mean,
        "rainfall_total": weather.rainfall_total,
        "crop_type": crop.crop_type,
        "crop_type_encoded": encode_basic_crop_type(crop.crop_type),
        "year": crop.year,
        "disease_class": image_prediction.disease_class,
        "health_score": image_prediction.health_score,
        "disease_class_health_score": validate_health_score_mapping(
            image_prediction.disease_class
        ),
        "image_confidence": image_prediction.confidence,
    }

    if weather.temperature_min is not None:
        features["temperature_min"] = weather.temperature_min
    if weather.temperature_max is not None:
        features["temperature_max"] = weather.temperature_max
    if weather.humidity_mean is not None:
        features["humidity_mean"] = weather.humidity_mean
    if weather.solar_radiation_mean is not None:
        features["solar_radiation_mean"] = weather.solar_radiation_mean
    if crop.region is not None:
        features["region"] = crop.region
    if crop.planting_age_days is not None:
        features["planting_age_days"] = crop.planting_age_days

    return features


def validate_health_score_mapping(disease_class: str) -> float:
    """Map a disease class label to a conservative health score estimate.

    The explicit mapping supports common severity labels. Unknown but non-empty
    disease classes receive a neutral middle score so future classifiers can add
    labels without breaking the demo pipeline.
    """
    normalized_class = disease_class.strip().lower().replace("_", " ").replace("-", " ")
    if not normalized_class:
        msg = "disease_class must not be empty"
        raise ValueError(msg)

    if normalized_class in _DISEASE_HEALTH_SCORE_MAPPING:
        return _DISEASE_HEALTH_SCORE_MAPPING[normalized_class]
    if "healthy" in normalized_class:
        return 1.0
    if "mild" in normalized_class:
        return 0.75
    if "moderate" in normalized_class:
        return 0.5
    if "severe" in normalized_class:
        return 0.25

    return 0.5


def encode_basic_crop_type(crop_type: str) -> int:
    """Encode common crop types with stable integer identifiers.

    Unknown but non-empty crop types are encoded as ``0`` so callers can still
    pass the original ``crop_type`` string alongside the numeric feature.
    """
    normalized_crop = crop_type.strip().lower().replace("_", " ").replace("-", " ")
    if not normalized_crop:
        msg = "crop_type must not be empty"
        raise ValueError(msg)

    return _CROP_TYPE_ENCODING.get(normalized_crop, 0)
