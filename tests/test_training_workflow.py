from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import torch

from cropnet_forecasting.features import FEATURE_COLS, META_COLS
from cropnet_forecasting.scaling import FeatureScaler
from cropnet_forecasting.training_dataset import (
    build_raw_dataset_dir,
    prepare_training_dataset,
    prepare_training_dataset_from_download,
)
from cropnet_forecasting.training_engine import build_model, build_pinn_model, build_sequence_splits, train_torch_model


def _synthetic_monthly_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    feature_count = len(FEATURE_COLS)
    for county_id in ("19001", "19003"):
        for year in range(2017, 2023):
            for month in range(1, 13):
                base = (year - 2016) * 10 + month
                row: dict[str, float | int | str] = {
                    "county_id": county_id,
                    "crop_type": "corn",
                    "year": year,
                    "month": month,
                }
                for idx, feature in enumerate(FEATURE_COLS):
                    row[feature] = float(base + idx)
                rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame.columns) == len(META_COLS) + feature_count
    return frame


def _write_synthetic_raw_snapshot(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in (2017, 2018, 2019):
        usda_dir = raw_dir / "USDA Crop Dataset" / "Corn" / str(year)
        usda_dir.mkdir(parents=True, exist_ok=True)
        usda_frame = pd.DataFrame(
            [
                {
                    "state_ansi": "19",
                    "county_ansi": "003",
                    "state_name": "IOWA",
                    "county_name": "ADAMS",
                    "agg_level_desc": "COUNTY",
                    "domain_desc": "TOTAL",
                    "YIELD, MEASURED IN BU / ACRE": 180.0 + year - 2017,
                }
            ]
        )
        usda_frame.to_csv(usda_dir / f"USDA_Corn_County_{year}.csv", index=False)

        ag_dir = raw_dir / "Sentinel-2 Imagery" / "data" / "AG" / str(year) / "IA"
        ndvi_dir = raw_dir / "Sentinel-2 Imagery" / "data" / "NDVI" / str(year) / "IA"
        weather_dir = raw_dir / "WRF-HRRR Computed Dataset" / "data" / str(year) / "IA"
        ag_dir.mkdir(parents=True, exist_ok=True)
        ndvi_dir.mkdir(parents=True, exist_ok=True)
        weather_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(ag_dir / f"ag_{year}.h5", "w") as handle:
            county = handle.create_group("19003")
            date_group = county.create_group(f"{year}-02-01")
            date_group.create_dataset(
                "data",
                data=np.array(
                    [
                        np.full((2, 2, 3), 64 + (year - 2017) * 10, dtype=np.uint8),
                        np.full((2, 2, 3), 128 + (year - 2017) * 10, dtype=np.uint8),
                    ]
                ),
            )

        with h5py.File(ndvi_dir / f"ndvi_{year}.h5", "w") as handle:
            county = handle.create_group("19003")
            date_group = county.create_group(f"{year}-02-01")
            date_group.create_dataset(
                "data",
                data=np.array(
                    [
                        np.full((2, 2), 0.25 + (year - 2017) * 0.05, dtype=np.float32),
                        np.full((2, 2), 0.75 - (year - 2017) * 0.05, dtype=np.float32),
                    ]
                ),
            )

        weather_frame = pd.DataFrame(
            [
                {
                    "date": f"{year}-01-01",
                    "FIPS Code": "19003",
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
                    "date": f"{year}-02-01",
                    "FIPS Code": "19003",
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
        weather_frame.to_csv(weather_dir / f"weather_{year}.csv", index=False)


def test_prepare_training_dataset_writes_year_splits(tmp_path: Path) -> None:
    source = tmp_path / "monthly.parquet"
    frame = _synthetic_monthly_frame()
    frame.to_parquet(source, index=False)

    prepared = prepare_training_dataset(
        source,
        tmp_path / "prepared",
        train_years=[2017, 2018, 2019, 2020],
        val_years=[2021],
        test_years=[2022],
        overwrite=True,
    )

    assert prepared.train_path.exists()
    assert prepared.val_path.exists()
    assert prepared.test_path.exists()
    metadata = json.loads(prepared.metadata_path.read_text(encoding="utf-8"))
    assert metadata["train_years"] == [2017, 2018, 2019, 2020]
    assert metadata["val_years"] == [2021]
    assert metadata["test_years"] == [2022]


def test_sequence_builder_and_lstm_training(tmp_path: Path) -> None:
    source = tmp_path / "monthly.parquet"
    frame = _synthetic_monthly_frame()
    frame.to_parquet(source, index=False)

    prepared = prepare_training_dataset(
        source,
        tmp_path / "prepared",
        train_years=[2017, 2018, 2019, 2020],
        val_years=[2021],
        test_years=[2022],
        overwrite=True,
    )
    scaler = FeatureScaler.from_csv(prepared.scaler_path).subset(FEATURE_COLS)
    all_frame = pd.read_parquet(prepared.all_path)
    splits = build_sequence_splits(all_frame, FEATURE_COLS, scaler, seq_len=6, target_mode="seasonal_residual")

    assert not splits["train"].empty()
    assert not splits["val"].empty()
    assert not splits["test"].empty()

    model = build_model("lstm", len(FEATURE_COLS), hidden_size=16, num_layers=1, dropout=0.0)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(splits["train"].X_scaled, dtype=torch.float32),
            torch.tensor(splits["train"].y_model_scaled, dtype=torch.float32),
        ),
        batch_size=16,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(splits["val"].X_scaled, dtype=torch.float32),
            torch.tensor(splits["val"].y_model_scaled, dtype=torch.float32),
        ),
        batch_size=16,
        shuffle=False,
    )

    trained, history = train_torch_model(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=1,
        patience=1,
    )

    assert history
    assert isinstance(trained, torch.nn.Module)


def test_pinn_training_records_physics_loss(tmp_path: Path) -> None:
    source = tmp_path / "monthly.parquet"
    frame = _synthetic_monthly_frame()
    frame.to_parquet(source, index=False)

    prepared = prepare_training_dataset(
        source,
        tmp_path / "prepared_pinn",
        train_years=[2017, 2018, 2019, 2020],
        val_years=[2021],
        test_years=[2022],
        overwrite=True,
    )
    scaler = FeatureScaler.from_csv(prepared.scaler_path).subset(FEATURE_COLS)
    all_frame = pd.read_parquet(prepared.all_path)
    splits = build_sequence_splits(all_frame, FEATURE_COLS, scaler, seq_len=6, target_mode="seasonal_residual")

    model = build_pinn_model(
        "lstm",
        len(FEATURE_COLS),
        feature_names=FEATURE_COLS,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        latent_state_dim=3,
        physics_loss="combined",
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(splits["train"].X_scaled, dtype=torch.float32),
            torch.tensor(splits["train"].y_model_scaled, dtype=torch.float32),
        ),
        batch_size=16,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(splits["val"].X_scaled, dtype=torch.float32),
            torch.tensor(splits["val"].y_model_scaled, dtype=torch.float32),
        ),
        batch_size=16,
        shuffle=False,
    )

    trained, history = train_torch_model(
        model,
        train_loader,
        val_loader,
        torch.device("cpu"),
        learning_rate=1e-3,
        weight_decay=0.0,
        max_epochs=1,
        patience=1,
        physics_weight=0.1,
        physics_warmup_epochs=0,
    )

    assert history
    assert "train_physics_loss" in history[0]
    assert "val_physics_loss" in history[0]
    assert history[0]["train_physics_loss"] >= 0.0
    assert isinstance(trained, torch.nn.Module)


def test_prepare_training_dataset_from_download_builds_raw_snapshot(tmp_path: Path, monkeypatch) -> None:
    raw_root = tmp_path / "raw"
    raw_dir = build_raw_dataset_dir(raw_root, crop_type="corn", state_codes=["IA"], years=[2017, 2018, 2019])
    _write_synthetic_raw_snapshot(raw_dir)

    from cropnet_forecasting import training_dataset as td

    monkeypatch.setattr(td, "snapshot_download", lambda *args, **kwargs: str(raw_dir))

    prepared = prepare_training_dataset_from_download(
        raw_root=raw_root,
        output_dir=tmp_path / "prepared",
        crop_type="corn",
        state_codes=["IA"],
        years=[2017, 2018, 2019],
        train_years=[2017],
        val_years=[2018],
        test_years=[2019],
        overwrite=True,
    )

    assert prepared.all_path.exists()
    assert prepared.train_path.exists()
    assert prepared.val_path.exists()
    assert prepared.test_path.exists()
    all_frame = pd.read_parquet(prepared.all_path)
    assert set(META_COLS).issubset(all_frame.columns)
    assert set(FEATURE_COLS).issubset(all_frame.columns)
