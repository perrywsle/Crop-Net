"""Schemas for selective CropNet dataset access."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from crop_fusion_ai.config.schemas import WeatherFeatures

CropNetImageType = Literal["AG", "NDVI"]


class CropNetQuery(BaseModel):
    """Small, bounded query for CropNet's official on-demand APIs."""

    crop_type: str = Field(min_length=1)
    fips_codes: list[str] = Field(min_length=1)
    years: list[int] = Field(min_length=1)
    image_type: CropNetImageType = "AG"
    include_usda: bool = True
    include_hrrr: bool = True
    include_sentinel2: bool = False

    @field_validator("crop_type")
    @classmethod
    def normalize_crop_type(cls, value: str) -> str:
        """Normalize crop labels for stable downstream API calls."""
        return value.strip()

    @field_validator("fips_codes")
    @classmethod
    def validate_fips_codes(cls, values: list[str]) -> list[str]:
        """Ensure FIPS codes are five-digit strings."""
        normalized_values = [value.strip() for value in values]
        invalid_values = [
            value
            for value in normalized_values
            if len(value) != 5 or not value.isdigit()
        ]
        if invalid_values:
            msg = "fips_codes must contain five-digit strings"
            raise ValueError(msg)
        return normalized_values

    @field_validator("years")
    @classmethod
    def validate_years(cls, values: list[int]) -> list[int]:
        """Limit CropNet queries to the documented dataset years."""
        invalid_values = [value for value in values if value < 2017 or value > 2022]
        if invalid_values:
            msg = "years must be between 2017 and 2022 for the CropNet dataset"
            raise ValueError(msg)
        return values

    def years_as_strings(self) -> list[str]:
        """Return years in the string format expected by the CropNet package."""
        return [str(year) for year in self.years]


class CropNetSample(BaseModel):
    """Normalized lightweight sample derived from CropNet records."""

    fips_code: str = Field(min_length=5, max_length=5)
    year: int = Field(ge=2017, le=2022)
    crop_type: str = Field(min_length=1)
    yield_value: float = Field(ge=0.0)
    yield_unit: str = Field(default="bushels_per_acre", min_length=1)
    weather: WeatherFeatures
    region: str | None = None
    sentinel2_image_path: str | None = None
