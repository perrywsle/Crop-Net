"""Tests for direct ground-truth CropNet yield regression."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cropnet_forecasting import yield_regression
from cropnet_forecasting.yield_regression import (
    aggregate_growing_season_features,
    build_model_pipelines,
    read_monthly_features,
    run_yield_regression,
)


def _monthly_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for county_id, offset in [("19001", 0.0), ("19003", 5.0)]:
        for year in [2021, 2022]:
            for month in range(4, 10):
                value = float(month + offset + (year - 2021))
                rows.append(
                    {
                        "county_id": county_id,
                        "crop_type": "corn",
                        "year": year,
                        "month": month,
                        "ag_green_pixel_ratio": value,
                    }
                )
    return rows


def _write_usda(path: Path) -> None:
    rows = [
        {
            "state_ansi": 19,
            "county_ansi": 1,
            "year": 2021,
            "YIELD, MEASURED IN BU / ACRE": 180.0,
        },
        {
            "state_ansi": 19,
            "county_ansi": 3,
            "year": 2021,
            "YIELD, MEASURED IN BU / ACRE": 190.0,
        },
        {
            "state_ansi": 19,
            "county_ansi": 1,
            "year": 2022,
            "YIELD, MEASURED IN BU / ACRE": 184.0,
        },
        {
            "state_ansi": 19,
            "county_ansi": 3,
            "year": 2022,
            "YIELD, MEASURED IN BU / ACRE": 195.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _small_model_pipelines(random_state: int = 42, *, include_optional_models: bool = False):
    del random_state, include_optional_models
    return {
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    }


def test_default_yield_models_are_the_baseline_three() -> None:
    """Direct yield benchmarking should default to the planned baseline models."""
    assert list(build_model_pipelines()) == ["Ridge", "RandomForest", "ExtraTrees"]


def test_annualization_adds_mean_slope_delta_and_amplitude() -> None:
    """Growing-season aggregation should keep means and simple trend descriptors."""
    monthly = pd.DataFrame(
        [
            {
                "county_id": "19001",
                "crop_type": "corn",
                "year": 2022,
                "month": month,
                "ag_green_pixel_ratio": float(month - 3),
            }
            for month in range(4, 10)
        ]
    )

    annual = aggregate_growing_season_features(monthly, ["ag_green_pixel_ratio"])
    row = annual.iloc[0]

    assert row["ag_green_pixel_ratio"] == pytest.approx(3.5)
    assert row["ag_green_pixel_ratio_slope"] == pytest.approx(1.0)
    assert row["ag_green_pixel_ratio_delta"] == pytest.approx(5.0)
    assert row["ag_green_pixel_ratio_amplitude"] == pytest.approx(5.0)


def test_read_monthly_features_rejects_forecast_generated_columns(tmp_path: Path) -> None:
    """Direct yield training must not accept blank-fill or forecasting outputs."""
    path = tmp_path / "blank_fill_predictions.csv"
    pd.DataFrame(
        [
            {
                "county_id": "19001",
                "crop_type": "corn",
                "year": 2022,
                "month": 4,
                "forecast_step": 1,
                "ag_green_pixel_ratio": 0.5,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="generated forecast columns"):
        read_monthly_features(path)


def test_run_yield_regression_saves_training_frame_model_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline run should persist reproducible direct-yield artifacts."""
    monthly_path = tmp_path / "official_monthly_feature_table.csv"
    usda_path = tmp_path / "USDA_Corn_County_2022.csv"
    output_dir = tmp_path / "yield_baseline"
    pd.DataFrame(_monthly_rows()).to_csv(monthly_path, index=False)
    _write_usda(usda_path)
    monkeypatch.setattr(yield_regression, "build_model_pipelines", _small_model_pipelines)

    artifacts = run_yield_regression(
        monthly_path=monthly_path,
        usda_paths=[usda_path],
        output_dir=output_dir,
        crop_type="corn",
        feature_group="ag",
        min_overlap_rows=1,
    )

    assert artifacts.training_frame_path.exists()
    assert artifacts.results_path.exists()
    assert artifacts.model_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.best_model_name == "Ridge"

    model = joblib.load(artifacts.model_path)
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    training_frame = pd.read_csv(artifacts.training_frame_path)

    assert hasattr(model, "predict")
    assert metadata["uses_forecast_generated_features"] is False
    assert metadata["target_units"] == ["BU / ACRE"]
    assert metadata["split_mode"] == "year_split (test_year=2022)"
    assert "forecast_step" not in training_frame.columns
    assert {"yield_bu_acre", "ag_green_pixel_ratio_slope"}.issubset(training_frame.columns)
