"""Tests for CropNet modality export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from crop_fusion_ai.preprocessing import (
    build_modality_allow_patterns,
    build_modality_records_by_split,
    get_state_abbr,
    write_modality_jsonl_splits,
)


def _write_ag_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        county = handle.create_group("01003")
        date_group = county.create_group("2017-02-01")
        date_group.create_dataset(
            "data",
            data=np.array(
                [
                    np.full((2, 2, 3), 64, dtype=np.uint8),
                    np.full((2, 2, 3), 192, dtype=np.uint8),
                ]
            ),
        )


def _write_ndvi_h5(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        county = handle.create_group("01003")
        date_group = county.create_group("2017-02-01")
        date_group.create_dataset(
            "data",
            data=np.array(
                [
                    np.full((2, 2), 0.25, dtype=np.float32),
                    np.full((2, 2), 0.75, dtype=np.float32),
                ]
            ),
        )


def _write_weather_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "date": "2017-01-01",
                "FIPS Code": "01003",
                "Daily/Monthly": "Monthly",
                "Avg Temperature (K)": 290.0,
                "Max Temperature (K)": 295.0,
                "Min Temperature (K)": 285.0,
                "Precipitation (kg m**-2)": 4.0,
                "Relative Humidity (%)": 65.0,
                "Wind Speed (m s**-1)": 3.0,
                "Downward Shortwave Radiation Flux (W m**-2)": 150.0,
            },
            {
                "date": "2017-02-01",
                "FIPS Code": "01003",
                "Daily/Monthly": "Monthly",
                "Avg Temperature (K)": 294.0,
                "Max Temperature (K)": 300.0,
                "Min Temperature (K)": 286.0,
                "Precipitation (kg m**-2)": 6.0,
                "Relative Humidity (%)": 70.0,
                "Wind Speed (m s**-1)": 4.0,
                "Downward Shortwave Radiation Flux (W m**-2)": 175.0,
            },
        ]
    )
    frame.to_csv(path, index=False)


def _usda_record() -> dict[str, object]:
    return {
        "split": "train",
        "county_id": "01003",
        "crop_type": "corn",
        "year": 2017,
        "target_kind": "yield",
        "target_value": 162.5,
        "target_unit": "BU / ACRE",
        "source_path": "USDA/data/Corn/2017/USDA_Corn_County_2017.csv",
        "state_ansi": "01",
        "county_ansi": "003",
        "state_name": "ALABAMA",
        "county_name": "BALDWIN",
    }


def test_build_modality_allow_patterns_tracks_states_and_years() -> None:
    """State-specific allow patterns should be built from the selected FIPS list."""
    patterns = build_modality_allow_patterns(
        "ag",
        years=[2017, 2022],
        fips_codes=["01003", "22007"],
    )

    assert "Sentinel-2 Imagery/data/AG/2017/AL/*.h5" in patterns
    assert get_state_abbr("22") == "LA"


def test_build_modality_records_by_split_extracts_ag_ndvi_and_weather(
    tmp_path: Path,
) -> None:
    """Each modality should flatten into one JSONL-ready record per USDA anchor."""
    _write_ag_h5(
        tmp_path
        / "Sentinel-2 Imagery"
        / "data"
        / "AG"
        / "2017"
        / "AL"
        / "Agriculture_01_AL_2017-01-01_2017-03-31.h5"
    )
    _write_ndvi_h5(
        tmp_path
        / "Sentinel-2 Imagery"
        / "data"
        / "NDVI"
        / "2017"
        / "AL"
        / "NDVI_01_AL_2017-01-01_2017-03-31.h5"
    )
    _write_weather_csv(
        tmp_path
        / "WRF-HRRR Computed Dataset"
        / "data"
        / "2017"
        / "AL"
        / "HRRR_01_AL_2017-01.csv"
    )

    record = _usda_record()

    ag_records = build_modality_records_by_split(
        tmp_path,
        [record],
        modality="ag",
    )
    ndvi_records = build_modality_records_by_split(
        tmp_path,
        [record],
        modality="ndvi",
    )
    weather_records = build_modality_records_by_split(
        tmp_path,
        [record],
        modality="weather",
    )

    assert len(ag_records["train"]) == 1
    assert len(ndvi_records["train"]) == 1
    assert len(weather_records["train"]) == 1

    ag_payload = ag_records["train"][0]
    ndvi_payload = ndvi_records["train"][0]
    weather_payload = weather_records["train"][0]

    assert ag_payload["county_id"] == "01003"
    assert ag_payload["features"]["ag_available_months"] == 1
    assert any(key.startswith("ag_m02_") for key in ag_payload["features"])
    assert "ag_growth_slope" in ag_payload["features"]

    assert ndvi_payload["features"]["ndvi_available_months"] == 1
    assert any(key.startswith("ndvi_m02_") for key in ndvi_payload["features"])
    assert "ndvi_growth_slope" in ndvi_payload["features"]

    assert weather_payload["features"]["weather_available_months"] == 2
    assert any(key.startswith("weather_m01_") for key in weather_payload["features"])
    assert "weather_temperature_trend_slope" in weather_payload["features"]


def test_write_modality_jsonl_splits_writes_three_split_files(
    tmp_path: Path,
) -> None:
    """The writer should emit train/validation/test JSONL files for one modality."""
    records_by_split = {
        "train": [_usda_record()],
        "validation": [],
        "test": [],
    }

    counts = write_modality_jsonl_splits(records_by_split, tmp_path / "ag")
    assert counts["train"] == 1

    train_path = tmp_path / "ag" / "train.jsonl"
    validation_path = tmp_path / "ag" / "validation.jsonl"
    test_path = tmp_path / "ag" / "test.jsonl"

    assert train_path.exists()
    assert validation_path.exists()
    assert test_path.exists()

    payload = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["county_id"] == "01003"
