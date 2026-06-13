"""Farmer-friendly labels and groupings for yield model features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureLabel:
    name: str
    label: str
    group: str
    description: str
    unit: str = "value"
    display: str = "decimal"
    hidden: bool = False


FEATURE_GROUPS: tuple[str, ...] = ("canopy", "vegetation", "weather", "season")

FEATURE_LABELS: dict[str, FeatureLabel] = {
    "ag_green_pixel_ratio": FeatureLabel(
        name="ag_green_pixel_ratio",
        label="Green canopy cover",
        group="canopy",
        description="How much of the field image is dominated by healthy green plant cover.",
        unit="percent",
        display="percent",
    ),
    "ag_brown_yellow_pixel_ratio": FeatureLabel(
        name="ag_brown_yellow_pixel_ratio",
        label="Brown or yellow crop cover",
        group="canopy",
        description="Share of the image showing stressed, dry, or yellowing plant cover.",
        unit="percent",
        display="percent",
    ),
    "ag_soil_exposure_ratio": FeatureLabel(
        name="ag_soil_exposure_ratio",
        label="Visible soil exposure",
        group="canopy",
        description="How much bare soil is visible between plants.",
        unit="percent",
        display="percent",
    ),
    "ag_shadow_cloud_ratio": FeatureLabel(
        name="ag_shadow_cloud_ratio",
        label="Shadow or cloud cover",
        group="canopy",
        description="How much of the image is affected by shade or cloud cover.",
        unit="percent",
        display="percent",
    ),
    "ag_mean_brightness": FeatureLabel(
        name="ag_mean_brightness",
        label="Average field brightness",
        group="canopy",
        description="The overall brightness of the crop image.",
    ),
    "ag_texture_entropy": FeatureLabel(
        name="ag_texture_entropy",
        label="Field texture variation",
        group="canopy",
        description="How visually mixed or complex the field surface looks.",
    ),
    "ag_field_uniformity_score": FeatureLabel(
        name="ag_field_uniformity_score",
        label="Field uniformity",
        group="canopy",
        description="How even the field looks across the image.",
    ),
    "ndvi_mean": FeatureLabel(
        name="ndvi_mean",
        label="Average vegetation vigor",
        group="vegetation",
        description="Average NDVI value for the scene.",
    ),
    "ndvi_median": FeatureLabel(
        name="ndvi_median",
        label="Typical vegetation vigor",
        group="vegetation",
        description="Median NDVI value for the scene.",
    ),
    "ndvi_max": FeatureLabel(
        name="ndvi_max",
        label="Peak vegetation vigor",
        group="vegetation",
        description="Highest NDVI value observed in the scene.",
    ),
    "ndvi_std": FeatureLabel(
        name="ndvi_std",
        label="Vegetation spread",
        group="vegetation",
        description="How much the vegetation signal varies across the field.",
    ),
    "ndvi_cv": FeatureLabel(
        name="ndvi_cv",
        label="Vegetation variability",
        group="vegetation",
        description="Relative variation in vegetation signal.",
    ),
    "ndvi_p25": FeatureLabel(
        name="ndvi_p25",
        label="Lower vegetation range",
        group="vegetation",
        description="25th percentile of vegetation vigor.",
    ),
    "ndvi_p75": FeatureLabel(
        name="ndvi_p75",
        label="Upper vegetation range",
        group="vegetation",
        description="75th percentile of vegetation vigor.",
    ),
    "ndvi_above_0_3_ratio": FeatureLabel(
        name="ndvi_above_0_3_ratio",
        label="Healthy vegetation share",
        group="vegetation",
        description="Share of pixels above a moderate vegetation threshold.",
        unit="percent",
        display="percent",
    ),
    "ndvi_above_0_5_ratio": FeatureLabel(
        name="ndvi_above_0_5_ratio",
        label="Strong vegetation share",
        group="vegetation",
        description="Share of pixels above a stronger vegetation threshold.",
        unit="percent",
        display="percent",
    ),
    "ndvi_valid_coverage_ratio": FeatureLabel(
        name="ndvi_valid_coverage_ratio",
        label="Useful NDVI coverage",
        group="vegetation",
        description="How much of the image produced a valid NDVI reading.",
        unit="percent",
        display="percent",
    ),
    "weather_temp_mean": FeatureLabel(
        name="weather_temp_mean",
        label="Average temperature",
        group="weather",
        description="Mean weather temperature for the month.",
    ),
    "weather_temp_max": FeatureLabel(
        name="weather_temp_max",
        label="Warmest temperature",
        group="weather",
        description="Highest weather temperature for the month.",
    ),
    "weather_temp_min": FeatureLabel(
        name="weather_temp_min",
        label="Coolest temperature",
        group="weather",
        description="Lowest weather temperature for the month.",
    ),
    "weather_gdd": FeatureLabel(
        name="weather_gdd",
        label="Growing degree days",
        group="weather",
        description="Heat accumulation that supports crop growth.",
    ),
    "weather_total_precipitation": FeatureLabel(
        name="weather_total_precipitation",
        label="Total rainfall",
        group="weather",
        description="Total precipitation collected for the month.",
    ),
    "weather_drought_index": FeatureLabel(
        name="weather_drought_index",
        label="Drought pressure",
        group="weather",
        description="Higher values indicate stronger drought stress.",
    ),
    "weather_humidity_mean": FeatureLabel(
        name="weather_humidity_mean",
        label="Average humidity",
        group="weather",
        description="Average humidity for the month.",
    ),
    "weather_solar_radiation_mean": FeatureLabel(
        name="weather_solar_radiation_mean",
        label="Average sunlight",
        group="weather",
        description="Average solar radiation during the month.",
    ),
    "month": FeatureLabel(
        name="month",
        label="Season month",
        group="season",
        description="Which month of the growing season the row represents.",
    ),
    "month_sin": FeatureLabel(
        name="month_sin",
        label="Season timing sine",
        group="season",
        description="Cyclical season encoding used by the model.",
        hidden=True,
    ),
    "month_cos": FeatureLabel(
        name="month_cos",
        label="Season timing cosine",
        group="season",
        description="Cyclical season encoding used by the model.",
        hidden=True,
    ),
}

GROUP_LABELS: dict[str, str] = {
    "canopy": "Crop canopy",
    "vegetation": "Vegetation health",
    "weather": "Weather conditions",
    "season": "Season timing",
}

GROUP_ORDER: dict[str, int] = {group: index for index, group in enumerate(FEATURE_GROUPS)}


def feature_label(name: str) -> FeatureLabel:
    label = FEATURE_LABELS.get(name)
    if label is not None:
        return label
    return FeatureLabel(
        name=name,
        label=name.replace("_", " ").title(),
        group="season",
        description="Model feature",
    )


def group_for_feature(name: str) -> str:
    return feature_label(name).group


def visible_feature_names(feature_names: list[str]) -> list[str]:
    return [name for name in feature_names if not feature_label(name).hidden]


def label_payload() -> dict[str, dict[str, str]]:
    return {
        name: {
            "label": spec.label,
            "group": spec.group,
            "group_label": GROUP_LABELS.get(spec.group, spec.group.title()),
            "description": spec.description,
            "unit": spec.unit,
            "display": spec.display,
            "hidden": str(spec.hidden).lower(),
        }
        for name, spec in FEATURE_LABELS.items()
    }

