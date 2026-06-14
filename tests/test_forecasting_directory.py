"""Tests for directory-based GUI forecasting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from crop_fusion_ai.gui.forecasting import build_monthly_features_from_directory, scan_directory
from crop_fusion_ai.gui.forecasting import build_forecast_from_directory
from crop_fusion_ai.gui.controller import PreprocessingController, UploadMetadata
from crop_fusion_ai.preprocessing.ndvi import extract_ndvi_features
from cropnet_forecasting.blank_fill import rollout_autoregressive


def _write_rgb_image(path: Path) -> None:
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[:, :16] = (40, 170, 55)
    array[:, 16:] = (170, 140, 90)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _write_ndvi_image(path: Path) -> None:
    array = np.full((32, 32), 0.35, dtype=np.float32)
    array[:16, :16] = 0.7
    array[16:, 16:] = 0.1
    scaled = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L").save(path)


def _write_weather_csv(path: Path) -> None:
    rows = []
    for day in range(1, 4):
        rows.append(
            {
                "date": f"2017-12-{day:02d}",
                "temp_mean": 21.0 + day,
                "temp_max": 27.0 + day,
                "temp_min": 15.0 + day,
                "precipitation": 1.5 if day == 2 else 0.0,
                "humidity": 74.0,
                "solar_radiation": 135.0 + day,
                "wind_speed": 2.1,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_scan_directory_discovers_modality_files(tmp_path: Path) -> None:
    """The directory scanner should identify supported modality files."""
    _write_rgb_image(tmp_path / "ag" / "2017_12_21.png")
    _write_ndvi_image(tmp_path / "ndvi" / "2017_12_21.png")
    _write_weather_csv(tmp_path / "weather" / "2017_12.csv")

    samples = scan_directory(tmp_path)

    assert {sample.modality for sample in samples} == {"ag", "ndvi", "weather"}
    assert any(sample.year == 2017 and sample.month == 12 for sample in samples)


def test_build_monthly_features_from_directory_matches_model_contract(tmp_path: Path) -> None:
    """The built monthly frame should expose the model input feature columns."""
    _write_rgb_image(tmp_path / "ag" / "2017_12_21.png")
    _write_ndvi_image(tmp_path / "ndvi" / "2017_12_21.png")
    _write_weather_csv(tmp_path / "weather" / "2017_12.csv")

    monthly, samples = build_monthly_features_from_directory(
        tmp_path,
        county_id="01003",
        crop_type="corn",
    )

    assert len(samples) == 3
    assert list(monthly[["county_id", "crop_type", "year", "month"]].iloc[0]) == [
        "01003",
        "corn",
        2017,
        12,
    ]
    for column in [
        "ag_green_pixel_ratio",
        "ndvi_mean",
        "weather_temp_mean",
        "weather_heat_stress_days",
        "weather_precipitation_days",
        "weather_wind_mean",
        "weather_vpd_mean",
        "weather_temp_range_mean",
    ]:
        assert column in monthly.columns


@dataclass
class _DummyPredictor:
    feature_names: list[str]
    seq_len: int = 2
    supports_latent_state: bool = False

    def predict_next(self, window: np.ndarray, seasonal_base: np.ndarray | None = None) -> np.ndarray:
        last_row = window[-1]
        return last_row + 1.0

    def predict_next_with_latents(self, window: np.ndarray, seasonal_base: np.ndarray | None = None) -> tuple[np.ndarray, None]:
        return self.predict_next(window, seasonal_base=seasonal_base), None


def test_rollout_autoregressive_crosses_year_boundary() -> None:
    """The autoregressive horizon should roll from December into the next year."""
    monthly = pd.DataFrame(
        [
            {"county_id": "01003", "crop_type": "corn", "year": 2021, "month": 11, "ag_green_pixel_ratio": 1.0},
            {"county_id": "01003", "crop_type": "corn", "year": 2021, "month": 12, "ag_green_pixel_ratio": 2.0},
        ]
    )

    result = rollout_autoregressive(_DummyPredictor(["ag_green_pixel_ratio"]), monthly, horizon=3)

    assert list(result.predictions[["year", "month"]].itertuples(index=False, name=None)) == [
        (2022, 1),
        (2022, 2),
        (2022, 3),
    ]
    assert list(result.predictions["forecast_step"]) == [1, 2, 3]


def test_preprocessing_controller_reuses_cached_features(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Repeated preprocessing of the same file should come from cache."""
    image_path = tmp_path / "ag" / "2017_12_21.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not-an-image-but-cacheable")

    calls = {"count": 0}

    def fake_extract_ag_features(*args, **kwargs):  # noqa: ANN001,ANN003
        calls["count"] += 1
        return pd.DataFrame(
            [
                {
                    "county_id": "01003",
                    "crop_type": "corn",
                    "year": 2017,
                    "month": 12,
                    "ag_green_pixel_ratio": 0.1,
                }
            ]
        )

    monkeypatch.setattr(
        "crop_fusion_ai.gui.controller.extract_ag_features",
        fake_extract_ag_features,
    )

    controller = PreprocessingController(cache_dir=tmp_path / "cache")
    metadata = UploadMetadata(county_id="01003", crop_type="corn", year=2017, month=12)

    first = controller.process_ag(image_path, metadata)
    second = controller.process_ag(image_path, metadata)

    assert calls["count"] == 1
    assert not first.empty
    assert not second.empty
    assert controller.last_cache_hit is True


def test_build_forecast_from_directory_runs_all_four_models(tmp_path: Path, monkeypatch) -> None:
    monthly = pd.DataFrame(
        [
            {
                "county_id": "01003",
                "crop_type": "corn",
                "year": 2021,
                "month": 11,
                "date": pd.Timestamp("2021-11-01"),
                "feature_a": 1.0,
            },
            {
                "county_id": "01003",
                "crop_type": "corn",
                "year": 2021,
                "month": 12,
                "date": pd.Timestamp("2021-12-01"),
                "feature_a": 2.0,
            },
        ]
    )

    class _DummyPredictor:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.feature_names = ["feature_a"]
            self.seq_len = 2
            self.supports_latent_state = False

        def predict_future_months(self, monthly_features: pd.DataFrame, horizon: int = 12, progress=None):  # noqa: ANN001
            del progress
            dates = pd.date_range("2022-01-01", periods=horizon, freq="MS")
            forecast = pd.DataFrame(
                {
                    "county_id": "01003",
                    "crop_type": "corn",
                    "year": dates.year,
                    "month": dates.month,
                    "date": dates,
                    "forecast_step": range(1, horizon + 1),
                    "source_note": "dummy",
                    "feature_a": [float(len(self.model_name) + step) for step in range(horizon)],
                }
            )
            return forecast

        def predict_future_months_with_latents(self, monthly_features: pd.DataFrame, horizon: int = 12, progress=None):  # noqa: ANN001
            return self.predict_future_months(monthly_features, horizon=horizon, progress=progress), None

    monkeypatch.setattr(
        "crop_fusion_ai.gui.forecasting.build_monthly_features_from_directory",
        lambda *args, **kwargs: (monthly, ["ag", "ndvi", "weather"]),
    )
    monkeypatch.setattr("crop_fusion_ai.gui.forecasting.prepare_monthly_features", lambda frame, feature_names: frame)
    monkeypatch.setattr(
        "cropnet_forecasting.predictor.BlankFillPredictor.from_artifacts",
        lambda checkpoint_path, scaler_path, config_path, device=None, model_name=None: _DummyPredictor(model_name or Path(checkpoint_path).stem.replace("_best", "")),
    )

    result = build_forecast_from_directory(
        tmp_path,
        county_id="01003",
        crop_type="corn",
        checkpoint_path="weights/lstm_best.pt",
        scaler_path="weights/scaler.csv",
        config_path="configs/residual_lstm_all.yaml",
        horizon=3,
    )

    assert set(result.forecast_by_model) == {"lstm", "transformer_encoder", "gru", "tiny_mamba_ssm"}
    assert result.predictor.model_name == "lstm"
    assert len(result.forecast) == 3
    assert result.derived_drivers_by_model == {}


def test_extract_ndvi_features_falls_back_for_low_signal_images(tmp_path: Path) -> None:
    """Low-signal NDVI previews should still produce a feature row."""
    path = tmp_path / "ndvi" / "2017_12_21.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(path)

    frame = extract_ndvi_features(path, county_id="01003", crop_type="corn", year=2017, month=12)

    assert not frame.empty
    assert frame.loc[0, "county_id"] == "01003"
