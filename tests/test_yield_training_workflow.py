from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cropnet_forecasting.features import FEATURE_COLS
from cropnet_forecasting.training_dataset import prepare_training_dataset
from cropnet_forecasting.yield_training import (
    BASELINE_MODELS,
    load_prepared_yield_dataset,
    prepare_yield_dataset,
    train_yield_models,
)


def _write_source_monthly(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for county_id, offset in [("19001", 0.0), ("19003", 5.0)]:
        for year in [2020, 2021, 2022]:
            for month in range(4, 10):
                row: dict[str, object] = {
                    "county_id": county_id,
                    "crop_type": "corn",
                    "year": year,
                    "month": month,
                }
                for idx, feature in enumerate(FEATURE_COLS):
                    row[feature] = float(month + offset + idx / 10.0)
                rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_usda(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for year in [2020, 2021, 2022]:
        rows.extend(
            [
                {
                    "state_ansi": 19,
                    "county_ansi": 1,
                    "year": year,
                    "YIELD, MEASURED IN BU / ACRE": 180.0 + float(year - 2020),
                },
                {
                    "state_ansi": 19,
                    "county_ansi": 3,
                    "year": year,
                    "YIELD, MEASURED IN BU / ACRE": 190.0 + float(year - 2020),
                },
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_prepare_yield_dataset_writes_split_artifacts(tmp_path: Path) -> None:
    monthly_path = tmp_path / "monthly.csv"
    usda_path = tmp_path / "USDA_Corn_County_2022.csv"
    source_dir = tmp_path / "source_training"
    output_dir = tmp_path / "prepared"
    _write_source_monthly(monthly_path)
    _write_usda(usda_path)
    prepare_training_dataset(
        monthly_path,
        source_dir,
        feature_group="all",
        crop_type="corn",
        train_years=[2020],
        val_years=[2021],
        test_years=[2022],
        overwrite=True,
    )

    prepared = prepare_yield_dataset(
        source_dataset_dir=source_dir,
        usda_paths=[usda_path],
        output_dir=output_dir,
        crop_type="corn",
        feature_group="all",
    )

    assert prepared.dataset_dir == output_dir
    assert (output_dir / "all.parquet").exists()
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert (output_dir / "test.parquet").exists()
    assert (output_dir / "metadata.json").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_years"] == [2020]
    assert metadata["val_years"] == [2021]
    assert metadata["test_years"] == [2022]
    assert "ag_green_pixel_ratio" in metadata["feature_columns"]
    assert "month_sin" in metadata["feature_columns"]


def test_train_yield_models_writes_reports_and_best_model(tmp_path: Path) -> None:
    monthly_path = tmp_path / "monthly.csv"
    usda_path = tmp_path / "USDA_Corn_County_2022.csv"
    source_dir = tmp_path / "source_training"
    dataset_dir = tmp_path / "prepared"
    run_dir = tmp_path / "runs" / "yield_smoke"
    _write_source_monthly(monthly_path)
    _write_usda(usda_path)
    prepare_training_dataset(
        monthly_path,
        source_dir,
        feature_group="all",
        crop_type="corn",
        train_years=[2020],
        val_years=[2021],
        test_years=[2022],
        overwrite=True,
    )

    prepare_yield_dataset(
        source_dataset_dir=source_dir,
        usda_paths=[usda_path],
        output_dir=dataset_dir,
        crop_type="corn",
        feature_group="all",
    )
    prepared = load_prepared_yield_dataset(dataset_dir)

    results, metadata = train_yield_models(
        prepared,
        run_dir=run_dir,
        models=["Ridge", "RandomForest", "ExtraTrees", *BASELINE_MODELS],
        random_state=42,
    )

    assert not results.empty
    assert {"Ridge", "RandomForest", "ExtraTrees", *BASELINE_MODELS}.issubset(set(results["model"]))
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "model_specs.csv").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "best_yield_model.joblib").exists()
    assert (run_dir / "val_predictions.csv").exists()
    assert (run_dir / "test_predictions.csv").exists()
    assert (run_dir / "prediction_residuals.csv").exists()
    assert (run_dir / "yield_feature_importance.csv").exists()
    assert (run_dir / "yield_feature_importance.png").exists()
    assert (run_dir / "month_benchmark.csv").exists()
    assert (run_dir / "window_benchmark.csv").exists()
    assert metadata["best_trainable_model"] in {"Ridge", "RandomForest", "ExtraTrees"}
