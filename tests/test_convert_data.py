"""Tests for GUI sample data conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_data import execute_plan, plan_conversion
from crop_fusion_ai.gui.forecasting import scan_directory


def _write_usda_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "county_id": "55141",
                "state_name": "Wisconsin",
                "county_name": "Oconto",
                "commodity_desc": "CORN",
                "year": 2021,
                "target_value": 180.0,
            }
        ]
    ).to_csv(path, index=False)


def _write_ag_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        county = handle.create_group("01003")
        date_group = county.create_group("2021-01-01")
        date_group.create_dataset(
            "data",
            data=np.array(
                [
                    np.full((8, 8, 3), 64, dtype=np.uint8),
                    np.full((8, 8, 3), 192, dtype=np.uint8),
                ]
            ),
        )


def _write_ndvi_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        county = handle.create_group("01003")
        date_group = county.create_group("2021-01-01")
        date_group.create_dataset(
            "data",
            data=np.array(
                [
                    np.full((8, 8), -0.2, dtype=np.float32),
                    np.full((8, 8), 0.7, dtype=np.float32),
                ]
            ),
        )


def _write_weather_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2021-01-01",
                "FIPS Code": "01003",
                "Daily/Monthly": "Monthly",
                "Avg Temperature (K)": 290.0,
                "Max Temperature (K)": 295.0,
                "Min Temperature (K)": 285.0,
                "Precipitation (kg m**-2)": 4.0,
                "Relative Humidity (%)": 65.0,
                "Wind Speed (m s**-1)": 3.0,
                "Downward Shortwave Radiation Flux (W m**-2)": 150.0,
            }
        ]
    ).to_csv(path, index=False)


def _write_cropnet_weather_csv(path: Path) -> None:
    _write_weather_csv(path)


def test_convert_data_creates_gui_ready_tree(tmp_path: Path) -> None:
    source = tmp_path / "sample_data"
    output = tmp_path / "test_data"
    _write_usda_csv(
        source / "USDA Crop Dataset" / "Corn" / "2021" / "USDA_Corn_County_2021.csv"
    )
    _write_ag_h5(
        source
        / "Sentinel-2 Imagery"
        / "data"
        / "AG"
        / "2021"
        / "WI"
        / "Agriculture_55_WI_2021-01-01_2021-03-31.h5"
    )
    _write_ndvi_h5(
        source
        / "Sentinel-2 Imagery"
        / "data"
        / "NDVI"
        / "2021"
        / "WI"
        / "Vegetation_55_WI_2021-01-01_2021-03-31.h5"
    )
    _write_weather_csv(
        source
        / "WRF-HRRR Computed Dataset"
        / "data"
        / "2021"
        / "WI"
        / "HRRR_55_WI_2021-01.csv"
    )

    planned = plan_conversion(source, output, allow_demo_fallback=False)
    manifest = execute_plan(planned, output)

    assert (output / "ag").is_dir()
    assert (output / "ndvi").is_dir()
    assert (output / "weather").is_dir()
    assert (output / "usda").is_dir()
    assert (output / "manifest.json").is_file()
    assert manifest["counts"]["usda"] == 1
    assert all(item.origin == "source" for item in planned)
    assert any(item.transform == "h5_preview" for item in planned)

    samples = scan_directory(output)
    assert {sample.modality for sample in samples} == {"ag", "ndvi", "weather"}


def test_convert_data_dry_run_only_builds_plan(tmp_path: Path) -> None:
    source = tmp_path / "sample_data"
    output = tmp_path / "test_data"
    _write_usda_csv(
        source / "USDA Crop Dataset" / "Corn" / "2021" / "USDA_Corn_County_2021.csv"
    )
    _write_ag_h5(
        source
        / "Sentinel-2 Imagery"
        / "data"
        / "AG"
        / "2021"
        / "WI"
        / "Agriculture_55_WI_2021-01-01_2021-03-31.h5"
    )
    _write_ndvi_h5(
        source
        / "Sentinel-2 Imagery"
        / "data"
        / "NDVI"
        / "2021"
        / "WI"
        / "Vegetation_55_WI_2021-01-01_2021-03-31.h5"
    )
    _write_weather_csv(
        source
        / "WRF-HRRR Computed Dataset"
        / "data"
        / "2021"
        / "WI"
        / "HRRR_55_WI_2021-01.csv"
    )

    planned = plan_conversion(source, output, allow_demo_fallback=False)

    assert planned
    assert not output.exists()


def test_plan_conversion_detects_cropnet_weather_directory(tmp_path: Path) -> None:
    source = tmp_path / "sample_data"
    output = tmp_path / "test_data"
    _write_ag_h5(
        source
        / "Sentinel-2 Imagery"
        / "data"
        / "AG"
        / "2021"
        / "WI"
        / "Agriculture_55_WI_2021-01-01_2021-03-31.h5"
    )
    _write_ndvi_h5(
        source
        / "Sentinel-2 Imagery"
        / "data"
        / "NDVI"
        / "2021"
        / "WI"
        / "Vegetation_55_WI_2021-01-01_2021-03-31.h5"
    )
    _write_cropnet_weather_csv(
        source
        / "WRF-HRRR Computed Dataset"
        / "data"
        / "2021"
        / "WI"
        / "HRRR_55_WI_2021-01.csv"
    )

    planned = plan_conversion(source, output, allow_demo_fallback=False)

    assert any(item.modality == "weather" for item in planned)
