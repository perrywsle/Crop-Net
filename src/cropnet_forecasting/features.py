from __future__ import annotations

from collections.abc import Sequence

AG_CORE = [
    "ag_green_pixel_ratio",
    "ag_vegetation_area_percent",
    "ag_brown_yellow_pixel_ratio",
    "ag_soil_exposure_ratio",
    "ag_shadow_cloud_ratio",
    "ag_mean_brightness",
    "ag_texture_entropy",
    "ag_field_uniformity_score",
]
NDVI_CORE = [
    "ndvi_mean",
    "ndvi_median",
    "ndvi_max",
    "ndvi_std",
    "ndvi_cv",
    "ndvi_p25",
    "ndvi_p75",
    "ndvi_above_0_3_ratio",
    "ndvi_above_0_5_ratio",
    "ndvi_above_0_7_ratio",
    "ndvi_low_ratio",
    "ndvi_valid_coverage_ratio",
]
WEATHER_CORE = [
    "weather_temp_mean",
    "weather_temp_max",
    "weather_temp_min",
    "weather_gdd",
    "weather_heat_stress_days",
    "weather_cold_stress_days",
    "weather_total_precipitation",
    "weather_precipitation_days",
    "weather_heavy_rain_days",
    "weather_drought_index",
    "weather_humidity_mean",
    "weather_wind_mean",
    "weather_solar_radiation_mean",
    "weather_vpd_mean",
    "weather_temp_range_mean",
]
FEATURE_COLS = AG_CORE + NDVI_CORE + WEATHER_CORE
FEATURE_GROUP_SELECTIONS = {
    "all": FEATURE_COLS,
    "ag": AG_CORE,
    "ndvi": NDVI_CORE,
    "weather": WEATHER_CORE,
    "ag_ndvi": AG_CORE + NDVI_CORE,
    "ag_weather": AG_CORE + WEATHER_CORE,
    "ndvi_weather": NDVI_CORE + WEATHER_CORE,
}
META_COLS = ["county_id", "crop_type", "year", "month"]

def selected_feature_columns(feature_group: str = "all") -> list[str]:
    try:
        return list(FEATURE_GROUP_SELECTIONS[feature_group])
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_GROUP_SELECTIONS))
        raise ValueError(f"Unknown feature group '{feature_group}'. Valid groups: {valid}") from exc

def modality_for_feature(feature_name: str) -> str:
    if feature_name in AG_CORE:
        return "ag"
    if feature_name in NDVI_CORE:
        return "ndvi"
    if feature_name in WEATHER_CORE:
        return "weather"
    raise KeyError(f"Unknown feature: {feature_name}")

def features_by_modality(feature_names: Sequence[str] | None = None) -> dict[str, list[str]]:
    names = list(feature_names or FEATURE_COLS)
    return {
        "ag": [name for name in names if name in AG_CORE],
        "ndvi": [name for name in names if name in NDVI_CORE],
        "weather": [name for name in names if name in WEATHER_CORE],
    }
