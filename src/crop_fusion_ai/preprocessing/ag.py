"""Feature extraction for CropNet agriculture imagery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from crop_fusion_ai.preprocessing.common import (
    ImageLike,
    boundary_pixel_count,
    connected_component_sizes,
    edge_density,
    grayscale_contrast,
    load_rgb_image,
    safe_divide,
    shannon_entropy,
)


@dataclass(frozen=True, slots=True)
class AgFeatureConfig:
    """Thresholds and constants used by AG preprocessing."""

    green_hue_min: int = 35
    green_hue_max: int = 110
    green_sat_min: int = 45
    green_val_min: int = 35
    brown_hue_min: int = 10
    brown_hue_max: int = 70
    brown_sat_min: int = 25
    brown_val_min: int = 20
    soil_sat_max: int = 95
    shadow_val_max: int = 45
    cloud_val_min: int = 220
    cloud_sat_max: int = 40


def _extract_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = np.asarray(Image.fromarray(rgb, mode="RGB").convert("HSV"), dtype=np.uint8)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def _green_mask(h: np.ndarray, s: np.ndarray, v: np.ndarray, config: AgFeatureConfig) -> np.ndarray:
    rgb_green = (s >= config.green_sat_min) & (v >= config.green_val_min)
    return rgb_green & (h >= config.green_hue_min) & (h <= config.green_hue_max)


def _brown_yellow_mask(
    rgb: np.ndarray,
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    config: AgFeatureConfig,
) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    rgb_brown = (
        (red > green)
        & (green > blue)
        & ((red - green) >= 5)
        & ((red - green) <= 30)
        & ((green - blue) >= 50)
    )
    chroma = (s >= config.brown_sat_min) & (v >= config.brown_val_min)
    return rgb_brown & chroma & (h >= config.brown_hue_min) & (h <= config.brown_hue_max)


def _soil_mask(rgb: np.ndarray, s: np.ndarray, v: np.ndarray, config: AgFeatureConfig) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    rgb_soil = (red >= green) & (green >= blue) & ((red - blue) <= 140)
    low_saturation = s <= config.soil_sat_max
    moderate_brightness = (v >= 25) & (v <= 220)
    return (rgb_soil & moderate_brightness) | (low_saturation & moderate_brightness)


def _shadow_cloud_mask(s: np.ndarray, v: np.ndarray, config: AgFeatureConfig) -> np.ndarray:
    return (v <= config.shadow_val_max) | ((v >= config.cloud_val_min) & (s <= config.cloud_sat_max))


def _texture_entropy_and_contrast(rgb: np.ndarray) -> tuple[float, float, float]:
    gray = np.asarray(Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.uint8)
    return shannon_entropy(gray), grayscale_contrast(gray), edge_density(gray)


def _component_features(mask: np.ndarray) -> tuple[int, float, float, float, float, float]:
    sizes = connected_component_sizes(mask)
    if not sizes:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    areas = np.asarray(sizes, dtype=np.float64)
    num_chunks = int(areas.size)
    mean_area = float(areas.mean())
    std_area = float(areas.std())
    largest_ratio = float(areas.max() / mask.size)
    boundary_count = boundary_pixel_count(mask)
    perimeter_to_area = safe_divide(float(boundary_count), float(mask.sum()))
    compactness = safe_divide(4.0 * np.pi * float(mask.sum()), float(boundary_count**2))
    return num_chunks, mean_area, std_area, largest_ratio, perimeter_to_area, compactness


def extract_ag_features(
    image_input: ImageLike,
    *,
    county_id: str | None = None,
    crop_type: str | None = None,
    year: int | None = None,
    month: int | None = None,
    config: AgFeatureConfig | None = None,
) -> pd.DataFrame:
    """Extract stable agriculture imagery features from a single AG scene."""
    config = config or AgFeatureConfig()
    rgb = load_rgb_image(image_input)
    h, s, v = _extract_hsv(rgb)

    green_mask = _green_mask(h, s, v, config)
    brown_mask = _brown_yellow_mask(rgb, h, s, v, config)
    soil_mask = _soil_mask(rgb, s, v, config)
    qc_mask = _shadow_cloud_mask(s, v, config)

    green_ratio = float(green_mask.mean())
    vegetation_area_percent = green_ratio * 100.0
    brown_ratio = float(brown_mask.mean())
    soil_ratio = float(soil_mask.mean())
    shadow_cloud_ratio = float(qc_mask.mean())

    num_chunks, mean_area, std_area, largest_ratio, perimeter_to_area, compactness = (
        _component_features(green_mask)
    )
    entropy, contrast, edges = _texture_entropy_and_contrast(rgb)

    mean_brightness = float(v.astype(np.float64).mean() / 255.0)
    color_saturation = float(s.astype(np.float64).mean() / 255.0)
    field_uniformity_score = float(1.0 / (1.0 + edges * 6.0 + contrast * 4.0 + std_area / (mean_area + 1.0)))
    green_to_brown_ratio = float(min(safe_divide(green_ratio, brown_ratio, default=0.0), 100.0))

    features: dict[str, float | int | str | None] = {
        "county_id": county_id,
        "crop_type": crop_type,
        "year": year,
        "month": month,
        "ag_green_pixel_ratio": green_ratio,
        "ag_vegetation_area_percent": vegetation_area_percent,
        "ag_number_of_vegetation_chunks": num_chunks,
        "ag_largest_vegetation_chunk_ratio": largest_ratio,
        "ag_mean_chunk_area": mean_area,
        "ag_chunk_area_std": std_area,
        "ag_brown_yellow_pixel_ratio": brown_ratio,
        "ag_green_to_brown_ratio": green_to_brown_ratio,
        "ag_soil_exposure_ratio": soil_ratio,
        "ag_texture_entropy": entropy,
        "ag_edge_density": edges,
        "ag_local_contrast": contrast,
        "ag_mean_brightness": mean_brightness,
        "ag_color_saturation_mean": color_saturation,
        "ag_shadow_cloud_ratio": shadow_cloud_ratio,
        "ag_field_uniformity_score": field_uniformity_score,
        "ag_vegetation_perimeter_to_area_ratio": perimeter_to_area,
        "ag_morphological_compactness": compactness,
    }

    return pd.DataFrame([features])


def derive_ag_time_series_features(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """Derive sequence-level AG features from monthly AG scene features."""
    if monthly_features.empty:
        return pd.DataFrame()

    frame = monthly_features.sort_values("month").reset_index(drop=True)
    vegetation = frame["ag_vegetation_area_percent"].astype(np.float64)
    months = frame["month"].astype(np.float64)

    month_to_month_change = vegetation.diff().fillna(0.0)
    growth_slope = float(np.polyfit(months.to_numpy(), vegetation.to_numpy(), 1)[0]) if len(frame) > 1 else 0.0
    peak_idx = int(vegetation.idxmax())
    peak_month = int(frame.loc[peak_idx, "month"])
    peak_value = float(vegetation.max())
    amplitude = float(vegetation.max() - vegetation.min())
    auc = float(np.trapezoid(vegetation.to_numpy(), months.to_numpy())) if len(frame) > 1 else float(vegetation.iloc[0])
    stability_score = float(1.0 / (1.0 + month_to_month_change.std()))
    summary = {
        "ag_growth_slope": growth_slope,
        "ag_peak_month": peak_month,
        "ag_peak_value": peak_value,
        "ag_amplitude": amplitude,
        "ag_auc": auc,
        "ag_stability_score": stability_score,
        "ag_month_to_month_vegetation_change_mean": float(month_to_month_change.mean()),
        "ag_month_to_month_vegetation_change_std": float(month_to_month_change.std()),
    }
    return pd.DataFrame([summary])
