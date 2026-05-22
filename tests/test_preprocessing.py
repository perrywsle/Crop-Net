"""Tests for CropNet preprocessing across AG, NDVI, weather, and GUI controller."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from crop_fusion_ai.gui.controller import PreprocessingController, UploadMetadata
from crop_fusion_ai.preprocessing.ag import (
    derive_ag_time_series_features,
    extract_ag_features,
)
from crop_fusion_ai.preprocessing.common import combine_modality_feature_frames
from crop_fusion_ai.preprocessing.ndvi import (
    derive_ndvi_time_series_features,
    extract_ndvi_features,
)
from crop_fusion_ai.preprocessing.weather import (
    derive_weather_time_series_features,
    extract_weather_features,
)


def _write_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _write_ndvi_image(path: Path, array: np.ndarray) -> None:
    scaled = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L").save(path)


def _ag_fixture() -> np.ndarray:
    image = np.full((96, 96, 3), (165, 135, 90), dtype=np.uint8)
    image[12:44, 12:44] = (40, 170, 55)
    image[56:80, 58:82] = (48, 180, 62)
    image[20:36, 62:86] = (185, 165, 72)
    image[70:90, 0:18] = (55, 55, 55)
    image[0:10, 0:96] = (240, 240, 240)
    return image


def _ndvi_fixture() -> np.ndarray:
    array = np.full((40, 40), -0.05, dtype=np.float64)
    array[0:20, 0:20] = 0.45
    array[0:20, 20:40] = 0.65
    array[20:40, 0:20] = 0.10
    array[20:40, 20:40] = -0.05
    return array


def _weather_fixture() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for year, temp_offset, rain_offset in ((2022, 0.0, 0.0), (2023, 2.0, 4.0)):
        for month, days in ((1, 31), (2, 28)):
            for day in range(1, days + 1):
                temp_mean = 18.0 + month + temp_offset + day * 0.02
                precip = 0.0 if day % 5 else 4.0 + rain_offset
                rows.append(
                    {
                        "date": f"{year}-{month:02d}-{day:02d}",
                        "temp_mean": temp_mean,
                        "temp_max": temp_mean + 6.0,
                        "temp_min": temp_mean - 6.0,
                        "precipitation": precip,
                        "humidity": 68.0 + month,
                        "solar_radiation": 120.0 + month * 5.0 + day * 0.1,
                        "wind_speed": 2.5 + month * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_extract_ag_features_detects_stable_vegetation_structure() -> None:
    """AG preprocessing should produce interpretable vegetation and texture features."""
    features = extract_ag_features(
        _ag_fixture(),
        county_id="1001",
        crop_type="corn",
        year=2022,
        month=6,
    )

    row = features.iloc[0]
    assert row["county_id"] == "1001"
    assert row["crop_type"] == "corn"
    assert row["ag_green_pixel_ratio"] > 0.10
    assert row["ag_brown_yellow_pixel_ratio"] > 0.02
    assert row["ag_soil_exposure_ratio"] > 0.40
    assert row["ag_number_of_vegetation_chunks"] >= 2
    assert 0.0 <= row["ag_field_uniformity_score"] <= 1.0
    assert row["ag_green_to_brown_ratio"] > 1.0


def test_derive_ag_time_series_features_summarizes_monthly_growth() -> None:
    """AG time-series summarization should expose a growth trend and peak month."""
    monthly = pd.DataFrame(
        {
            "month": [1, 2, 3, 4],
            "ag_vegetation_area_percent": [12.0, 19.0, 28.0, 22.0],
        }
    )

    summary = derive_ag_time_series_features(monthly).iloc[0]

    assert summary["ag_peak_month"] == 3
    assert summary["ag_growth_slope"] > 0.0
    assert summary["ag_amplitude"] == pytest.approx(16.0)
    assert summary["ag_auc"] > 0.0


def test_extract_ndvi_features_filters_values_and_thresholds() -> None:
    """NDVI preprocessing should keep valid pixels and compute threshold ratios."""
    features = extract_ndvi_features(
        _ndvi_fixture(),
        county_id="1001",
        crop_type="corn",
        year=2022,
        month=6,
    )

    row = features.iloc[0]
    assert row["ndvi_mean"] == pytest.approx(0.2875, rel=1e-3)
    assert row["ndvi_above_0_3_ratio"] > 0.30
    assert row["ndvi_above_0_5_ratio"] > 0.20
    assert row["ndvi_low_ratio"] > 0.20
    assert row["ndvi_valid_coverage_ratio"] == pytest.approx(1.0)
    assert row["ndvi_healthy_patch_count"] >= 1


def test_derive_ndvi_time_series_features_detects_peak_and_trend() -> None:
    """NDVI time-series summarization should identify the growth peak and slope."""
    monthly = pd.DataFrame(
        {
            "month": [1, 2, 3, 4],
            "ndvi_mean": [0.18, 0.29, 0.52, 0.41],
        }
    )

    summary = derive_ndvi_time_series_features(monthly).iloc[0]

    assert summary["ndvi_peak_month"] == 3
    assert summary["ndvi_growth_slope"] > 0.0
    assert summary["ndvi_amplitude"] == pytest.approx(0.34, rel=1e-2)
    assert summary["ndvi_auc"] > 0.0


def test_extract_weather_features_aggregates_monthly_series() -> None:
    """Weather preprocessing should aggregate daily records into monthly features."""
    features = extract_weather_features(
        _weather_fixture(),
        county_id="1001",
        crop_type="corn",
    )

    assert len(features) == 4
    january_2022 = features[(features["year"] == 2022) & (features["month"] == 1)].iloc[0]
    february_2023 = features[(features["year"] == 2023) & (features["month"] == 2)].iloc[0]

    assert january_2022["weather_total_precipitation"] > 0.0
    assert january_2022["weather_gdd"] > 0.0
    assert january_2022["weather_max_dry_streak"] >= 1
    assert february_2023["weather_rainfall_anomaly"] > 0.0
    assert not pd.isna(february_2023["weather_rainfall_lag_1"])
    assert february_2023["weather_rainfall_lag_1"] == pytest.approx(
        features[(features["year"] == 2023) & (features["month"] == 1)]["weather_total_precipitation"].iloc[0]
    )


def test_derive_weather_time_series_features_summarizes_seasonality() -> None:
    """Weather time-series summarization should expose season-scale metrics."""
    monthly = extract_weather_features(_weather_fixture(), county_id="1001", crop_type="corn")

    summary = derive_weather_time_series_features(monthly).iloc[0]

    assert summary["weather_temperature_trend_slope"] != 0.0
    assert summary["weather_peak_rainfall_month"] in {1, 2}
    assert summary["weather_total_precipitation_auc"] > 0.0
    assert summary["weather_heat_stress_total"] >= 0.0


def test_combine_modality_feature_frames_joins_on_metadata() -> None:
    """Modality feature frames should merge cleanly on shared metadata columns."""
    ag = extract_ag_features(_ag_fixture(), county_id="1001", crop_type="corn", year=2022, month=6)
    ndvi = extract_ndvi_features(_ndvi_fixture(), county_id="1001", crop_type="corn", year=2022, month=6)
    weather = extract_weather_features(_weather_fixture(), county_id="1001", crop_type="corn")
    weather_row = weather[(weather["year"] == 2022) & (weather["month"] == 6)].copy()
    if weather_row.empty:
        weather_row = weather.iloc[[0]].copy()

    combined = combine_modality_feature_frames(ag, ndvi, weather_row)

    assert not combined.empty
    assert combined.loc[0, "county_id"] == "1001"
    assert "ag_green_pixel_ratio" in combined.columns
    assert "ndvi_mean" in combined.columns
    assert "weather_temp_mean" in combined.columns


def test_preprocessing_controller_processes_all_modalities(tmp_path: Path) -> None:
    """The GUI controller should call the preprocessing modules without Tk state."""
    ag_path = tmp_path / "ag.png"
    ndvi_path = tmp_path / "ndvi.png"
    weather_path = tmp_path / "weather.csv"

    _write_rgb_image(ag_path, _ag_fixture())
    _write_ndvi_image(ndvi_path, _ndvi_fixture())
    _weather_fixture().to_csv(weather_path, index=False)

    controller = PreprocessingController()
    metadata = UploadMetadata(county_id="1001", crop_type="corn", year=2022, month=6)

    ag = controller.process_ag(ag_path, metadata)
    ndvi = controller.process_ndvi(ndvi_path, metadata)
    weather = controller.process_weather(weather_path, metadata)

    assert not ag.empty
    assert not ndvi.empty
    assert not weather.empty
    assert ag.loc[0, "county_id"] == "1001"
    assert ndvi.loc[0, "crop_type"] == "corn"
    assert "weather_total_precipitation" in weather.columns
