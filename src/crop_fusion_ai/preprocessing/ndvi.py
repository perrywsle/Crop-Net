"""Feature extraction for CropNet NDVI imagery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crop_fusion_ai.preprocessing.common import ImageLike, load_ndvi_array


@dataclass(frozen=True, slots=True)
class NdviFeatureConfig:
    """Thresholds used by NDVI feature extraction."""

    valid_min: float = -0.2
    valid_max: float = 1.0
    vegetation_threshold: float = 0.3
    healthy_threshold: float = 0.5
    dense_threshold: float = 0.7
    low_threshold: float = 0.2


def _valid_ndvi_values(ndvi: np.ndarray, config: NdviFeatureConfig) -> np.ndarray:
    valid_mask = np.isfinite(ndvi)
    valid_mask &= ndvi >= config.valid_min
    valid_mask &= ndvi <= config.valid_max
    return ndvi[valid_mask]


def _connected_patch_ratio(mask: np.ndarray) -> tuple[int, float]:
    from crop_fusion_ai.preprocessing.common import connected_component_sizes

    sizes = connected_component_sizes(mask)
    if not sizes:
        return 0, 0.0
    areas = np.asarray(sizes, dtype=np.float64)
    return int(areas.size), float(areas.max() / mask.size)

def _compute_coefficient_of_variation(values: np.ndarray) -> float:
    """Compute CV = std/mean, robust to near-zero means."""
    mean_val = np.mean(values)
    std_val = np.std(values)
    if mean_val == 0 or np.abs(mean_val) < 1e-6:
        return 0.0
    return float(std_val / np.abs(mean_val))


def extract_ndvi_features(
    image_input: ImageLike,
    *,
    county_id: str | None = None,
    crop_type: str | None = None,
    year: int | None = None,
    month: int | None = None,
    config: NdviFeatureConfig | None = None,
) -> pd.DataFrame:
    """Extract stable statistics from a single NDVI scene."""
    config = config or NdviFeatureConfig()
    ndvi = load_ndvi_array(image_input)
    valid_values = _valid_ndvi_values(ndvi, config)

    if valid_values.size == 0:
        msg = "No valid NDVI pixels were found"
        raise ValueError(msg)

    mean_ndvi = float(np.mean(valid_values))
    median_ndvi = float(np.median(valid_values))
    max_ndvi = float(np.max(valid_values))
    std_ndvi = float(np.std(valid_values))
    p25 = float(np.percentile(valid_values, 25))
    p75 = float(np.percentile(valid_values, 75))
    iqr = p75 - p25

    cv_ndvi = _compute_coefficient_of_variation(valid_values)

    ratio_above_03 = float(np.mean(valid_values > config.vegetation_threshold))
    ratio_above_05 = float(np.mean(valid_values > config.healthy_threshold))
    ratio_above_07 = float(np.mean(valid_values > config.dense_threshold))
    low_ratio = float(np.mean(valid_values < config.low_threshold))

    high_mask = ndvi > config.healthy_threshold
    num_patches, largest_patch_ratio = _connected_patch_ratio(high_mask.astype(np.uint8))

    #  Entropy (vegetation diversity)
    hist, _ = np.histogram(valid_values, bins=10)
    prob = hist / (hist.sum() + 1e-8)
    ndvi_entropy = float(-np.sum(prob * np.log(prob + 1e-8)))

    valid_monthly_mean = mean_ndvi
    vegetation_mask = ndvi > config.vegetation_threshold
    monthly_coverage = float(vegetation_mask.mean())

     # Derived interaction features
    health_index = float(mean_ndvi * ratio_above_05)
    stress_index = float(std_ndvi * low_ratio)

    # Greenup potential (relationship between healthy and any vegetation)
    greenup_potential = float(ratio_above_05 / (ratio_above_03 + 1e-8))

    features: dict[str, float | int | str | None] = {
        "county_id": county_id,
        "crop_type": crop_type,
        "year": year,
        "month": month,
        "ndvi_mean": mean_ndvi,
        "ndvi_median": median_ndvi,
        "ndvi_max": max_ndvi,
        "ndvi_std": std_ndvi,
        "ndvi_cv": cv_ndvi,
        "ndvi_p25": p25,
        "ndvi_p75": p75,
        "ndvi_iqr": iqr,
        "ndvi_above_0_3_ratio": ratio_above_03,
        "ndvi_above_0_5_ratio": ratio_above_05,
        "ndvi_above_0_7_ratio": ratio_above_07,
        "ndvi_low_ratio": low_ratio,
        "ndvi_valid_coverage_ratio": float(valid_values.size / ndvi.size),
        "ndvi_healthy_patch_count": num_patches,
        "ndvi_largest_high_ndvi_patch_ratio": largest_patch_ratio,
        "ndvi_vegetation_coverage_ratio": monthly_coverage,
        "ndvi_valid_mean_proxy": valid_monthly_mean,
        "ndvi_entropy": ndvi_entropy,
         "ndvi_health_index": health_index,  
        "ndvi_stress_index": stress_index,
        "ndvi_greenup_potential": greenup_potential,
    }

    return pd.DataFrame([features])


def derive_ndvi_time_series_features(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """Derive sequence-level NDVI features from monthly NDVI records."""
    if monthly_features.empty:
        return pd.DataFrame()

    frame = monthly_features.sort_values("month").reset_index(drop=True)
    mean_series = frame["ndvi_mean"].astype(np.float64)
    months = frame["month"].astype(np.float64)
    monthly_change = mean_series.diff().fillna(0.0)
    slope = float(np.polyfit(months.to_numpy(), mean_series.to_numpy(), 1)[0]) if len(frame) > 1 else 0.0
    peak_idx = int(mean_series.idxmax())
    
    # Greenup rate (rate of increase to peak)
    if peak_idx > 0 and months[peak_idx] - months[0] > 0:
        greenup_rate = float((mean_series[peak_idx] - mean_series[0]) / (months[peak_idx] - months[0]))
    else:
        greenup_rate = 0.0
    
    # Coefficient of variation for the time series (stability metric)
    ts_cv = float(mean_series.std() / (mean_series.mean() + 1e-8))
    
    summary = {
        "ndvi_growth_slope": slope,
        "ndvi_peak_value": float(mean_series.max()),
        "ndvi_peak_month": int(frame.loc[peak_idx, "month"]),
        "ndvi_amplitude": float(mean_series.max() - mean_series.min()),
        "ndvi_auc": float(np.trapezoid(mean_series.to_numpy(), months.to_numpy())) if len(frame) > 1 else float(mean_series.iloc[0]),
        "ndvi_stability_score": float(1.0 / (1.0 + monthly_change.std())),
        "ndvi_month_to_month_change_mean": float(monthly_change.mean()),
        "ndvi_month_to_month_change_std": float(monthly_change.std()),
        "ndvi_greenup_rate": greenup_rate,
        "ndvi_timeseries_cv": ts_cv,  
    }
    return pd.DataFrame([summary])