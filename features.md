# CropNet Feature Reference

This document describes the feature sets extracted by the preprocessing package for the three CropNet modalities:

- Agriculture imagery (`AG`)
- NDVI imagery (`NDVI`)
- Weather time-series (`weather`)

The intended output shape is:

- `county_id × crop_type × year × month → feature vector`

The monthly records can be used directly as a time series or aggregated into sequence-level summary features.

## Shared Metadata

Each modality output includes the same metadata fields when available:

- `county_id`
- `crop_type`
- `year`
- `month`

These fields are used for alignment and merging across modalities.

## Agriculture Imagery Features

AG preprocessing focuses on visible vegetation coverage, patch structure, texture, and quality control.

### Core vegetation and color features

- `ag_green_pixel_ratio` - Fraction of pixels classified as green vegetation.
- `ag_vegetation_area_percent` - Same as green coverage, expressed as a percentage.
- `ag_brown_yellow_pixel_ratio` - Fraction of pixels classified as brown/yellow stressed vegetation.
- `ag_green_to_brown_ratio` - Ratio of green coverage to brown/yellow coverage.
- `ag_soil_exposure_ratio` - Fraction of pixels that look like exposed soil.
- `ag_shadow_cloud_ratio` - Fraction of pixels likely affected by shadow or cloud contamination.

### Vegetation patch and morphology features

- `ag_number_of_vegetation_chunks` - Number of connected green vegetation regions.
- `ag_largest_vegetation_chunk_ratio` - Largest vegetation patch area divided by total image area.
- `ag_mean_chunk_area` - Mean connected-component area of vegetation patches.
- `ag_chunk_area_std` - Standard deviation of vegetation patch areas.
- `ag_vegetation_perimeter_to_area_ratio` - Approximate boundary length relative to vegetation area.
- `ag_morphological_compactness` - Compactness score computed from vegetation area and boundary size.

### Texture and visual quality features

- `ag_texture_entropy` - Shannon entropy of grayscale texture.
- `ag_edge_density` - Fraction of edge pixels detected in grayscale space.
- `ag_local_contrast` - Normalized grayscale standard deviation.
- `ag_mean_brightness` - Average brightness from the HSV value channel.
- `ag_color_saturation_mean` - Average color saturation from the HSV saturation channel.
- `ag_field_uniformity_score` - Stability proxy combining texture, edge density, and patch variation.

### Sequence-level AG features

When monthly AG features are available, the following seasonal summaries are derived:

- `ag_growth_slope` - Linear trend of vegetation coverage across months.
- `ag_peak_month` - Month with maximum vegetation coverage.
- `ag_peak_value` - Maximum vegetation coverage value.
- `ag_amplitude` - Difference between maximum and minimum vegetation coverage.
- `ag_auc` - Area under the vegetation coverage curve across months.
- `ag_stability_score` - Smoothness score based on month-to-month variation.
- `ag_month_to_month_vegetation_change_mean` - Mean change in vegetation coverage between months.
- `ag_month_to_month_vegetation_change_std` - Standard deviation of month-to-month vegetation change.

## NDVI Features

NDVI preprocessing extracts physically meaningful vegetation-health statistics from each scene.

### Pixel statistics

- `ndvi_mean` - Mean NDVI over valid pixels.
- `ndvi_median` - Median NDVI over valid pixels.
- `ndvi_max` - Maximum valid NDVI value.
- `ndvi_std` - Standard deviation of valid NDVI pixels.
- `ndvi_p25` - 25th percentile of valid NDVI.
- `ndvi_p75` - 75th percentile of valid NDVI.
- `ndvi_iqr` - Interquartile range (`ndvi_p75 - ndvi_p25`).

### Threshold-based coverage features

- `ndvi_above_0_3_ratio` - Fraction of valid pixels above 0.3.
- `ndvi_above_0_5_ratio` - Fraction of valid pixels above 0.5.
- `ndvi_above_0_7_ratio` - Fraction of valid pixels above 0.7.
- `ndvi_low_ratio` - Fraction of valid pixels below 0.2.
- `ndvi_valid_coverage_ratio` - Fraction of pixels considered valid after filtering.
- `ndvi_vegetation_coverage_ratio` - Fraction of the full scene above the vegetation threshold.

### Spatial patch features

- `ndvi_healthy_patch_count` - Number of connected regions above the healthy threshold.
- `ndvi_largest_high_ndvi_patch_ratio` - Largest high-NDVI patch area divided by total scene area.

### Sequence-level NDVI features

When monthly NDVI records are available, the following seasonal summaries are derived:

- `ndvi_growth_slope` - Linear trend of mean NDVI across months.
- `ndvi_peak_value` - Maximum mean NDVI value across months.
- `ndvi_peak_month` - Month where NDVI reaches its maximum.
- `ndvi_amplitude` - Difference between max and min monthly NDVI.
- `ndvi_auc` - Area under the monthly mean NDVI curve.
- `ndvi_stability_score` - Smoothness score based on month-to-month changes.
- `ndvi_month_to_month_change_mean` - Mean change in NDVI between months.
- `ndvi_month_to_month_change_std` - Standard deviation of month-to-month NDVI change.

## Weather Features

Weather preprocessing converts daily weather observations into monthly agronomic features.

### Monthly temperature features

- `weather_temp_mean` - Monthly mean temperature.
- `weather_temp_max` - Monthly maximum temperature.
- `weather_temp_min` - Monthly minimum temperature.
- `weather_temp_range` - Difference between monthly max and min temperature.
- `weather_temp_lag_1` - Previous month’s mean temperature.
- `weather_temperature_anomaly` - Difference from the month-specific climatology baseline.

### Heat and cold stress features

- `weather_gdd` - Growing degree days accumulated in the month.
- `weather_extreme_heat_days` - Number of days at or above the heat threshold.
- `weather_cold_stress_degree_days` - Accumulated cold stress relative to the frost threshold.
- `weather_heat_stress_degree_days` - Accumulated heat stress above the heat threshold.
- `weather_heat_stress_total` - Seasonal sum of monthly heat stress degree days.

### Precipitation and drought features

- `weather_total_precipitation` - Total monthly rainfall.
- `weather_rainy_days` - Number of rainy days in the month.
- `weather_dry_days` - Number of dry days in the month.
- `weather_max_dry_streak` - Longest consecutive dry spell in the month.
- `weather_precipitation_intensity` - Average rainfall per rainy day.
- `weather_drought_index` - Monthly drought severity proxy.
- `weather_rainfall_anomaly` - Difference from the month-specific rainfall baseline.
- `weather_rainfall_lag_1` - Previous month’s precipitation total.
- `weather_rainfall_lag_2` - Preceding two-month precipitation lag.
- `weather_drought_severity_max` - Maximum drought severity across the sequence.

### Humidity, VPD, wind, and solar features

- `weather_humidity_mean` - Monthly mean relative humidity.
- `weather_vpd` - Monthly mean vapor pressure deficit.
- `weather_vpd_lag_1` - Previous month’s VPD.
- `weather_wind_speed_mean` - Monthly mean wind speed.
- `weather_wind_speed_max` - Monthly maximum wind speed.
- `weather_solar_radiation_mean` - Monthly mean solar radiation.
- `weather_cumulative_solar_radiation` - Monthly sum of solar radiation.
- `weather_weather_volatility_score` - Stability proxy from temperature, precipitation, and solar variability.

### Sequence-level weather features

When monthly weather records are available, the following seasonal summaries are derived:

- `weather_temperature_trend_slope` - Linear trend of monthly mean temperature.
- `weather_rainfall_trend_slope` - Linear trend of monthly rainfall.
- `weather_vpd_trend_slope` - Linear trend of monthly VPD.
- `weather_total_precipitation_auc` - Area under the monthly rainfall curve.
- `weather_peak_rainfall_month` - Month with maximum rainfall.
- `weather_rainfall_seasonality_index` - Rainfall variability relative to the mean.

## Recommended Core Feature Set

If you want a compact modeling set first, start with:

### AG

- `ag_green_pixel_ratio`
- `ag_vegetation_area_percent`
- `ag_largest_vegetation_chunk_ratio`
- `ag_brown_yellow_pixel_ratio`
- `ag_soil_exposure_ratio`
- `ag_texture_entropy`
- `ag_edge_density`
- `ag_field_uniformity_score`

### NDVI

- `ndvi_mean`
- `ndvi_median`
- `ndvi_std`
- `ndvi_p25`
- `ndvi_p75`
- `ndvi_above_0_3_ratio`
- `ndvi_above_0_5_ratio`
- `ndvi_low_ratio`
- `ndvi_peak_value`
- `ndvi_peak_month`
- `ndvi_growth_slope`
- `ndvi_auc`

### Weather

- `weather_temp_mean`
- `weather_temp_max`
- `weather_temp_min`
- `weather_temp_range`
- `weather_gdd`
- `weather_extreme_heat_days`
- `weather_total_precipitation`
- `weather_rainy_days`
- `weather_dry_days`
- `weather_max_dry_streak`
- `weather_humidity_mean`
- `weather_vpd`
- `weather_solar_radiation_mean`
- `weather_temperature_anomaly`
- `weather_rainfall_anomaly`

## Notes

- NDVI values are filtered to ignore invalid pixels before statistics are computed.
- Weather data is aggregated from daily records into monthly features.
- Sequence-level features are derived from monthly records and are useful for forecasting models such as LSTM, Transformer, SSM, ARIMA, or linear trend models.
- The preprocessing code is intentionally designed to stay stable over time, so the features are more suitable for extrapolation than raw pixels or raw daily sequences.
