"""Validated data schemas for multimodal crop prediction."""

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    """Prediction produced by the plant health image classifier."""

    disease_class: str = Field(min_length=1)
    health_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class WeatherFeatures(BaseModel):
    """Weather and climate features used by the yield model."""

    temperature_mean: float
    temperature_min: float | None = None
    temperature_max: float | None = None
    rainfall_total: float = Field(ge=0.0)
    humidity_mean: float | None = Field(default=None, ge=0.0, le=100.0)
    solar_radiation_mean: float | None = Field(default=None, ge=0.0)


class CropFeatures(BaseModel):
    """Crop metadata used for yield prediction."""

    crop_type: str = Field(min_length=1)
    region: str | None = None
    year: int = Field(ge=1900)
    planting_age_days: int | None = Field(default=None, ge=0)


class YieldInput(BaseModel):
    """Complete late-fusion input for the yield prediction model."""

    weather: WeatherFeatures
    crop: CropFeatures
    image_prediction: ImagePrediction


class YieldPrediction(BaseModel):
    """Yield model output returned to downstream users or the desktop UI."""

    predicted_yield: float = Field(ge=0.0)
    unit: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
