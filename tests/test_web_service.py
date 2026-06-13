"""Tests for the browser-first yield web service."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd

from crop_fusion_ai.web.feature_labels import label_payload
import crop_fusion_ai.web.service as web_service_module
from crop_fusion_ai.web.service import YieldModelService
from cropnet_forecasting.data import prepare_monthly_features
from cropnet_forecasting.models import CropNetModelFactory, infer_architecture_from_state_dict


def _synthetic_monthly_frame(service: YieldModelService) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in [4, 5, 6]:
        row: dict[str, object] = {
            "county_id": "19001",
            "crop_type": "corn",
            "year": 2022,
            "month": month,
        }
        for index, feature_name in enumerate(service.feature_names):
            if feature_name == "month":
                row[feature_name] = month
            elif feature_name == "month_sin":
                row[feature_name] = math.sin(2.0 * math.pi * month / 12.0)
            elif feature_name == "month_cos":
                row[feature_name] = math.cos(2.0 * math.pi * month / 12.0)
            else:
                row[feature_name] = (index + month) / 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_labels_are_farmer_friendly() -> None:
    payload = label_payload()

    assert payload["ag_green_pixel_ratio"]["label"] == "Green canopy cover"
    assert payload["ag_green_pixel_ratio"]["group"] == "canopy"
    assert payload["month_sin"]["hidden"] == "true"


def test_prepare_monthly_features_deduplicates_month_column() -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service)

    prepared = prepare_monthly_features(frame, service.feature_names)

    assert list(prepared.columns).count("month") == 1
    assert prepared.shape[0] == 3


def test_prepare_monthly_features_derives_month_sin_cos() -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service).drop(columns=["month_sin", "month_cos"])

    prepared = prepare_monthly_features(frame, service.feature_names)

    assert "month_sin" in prepared.columns
    assert "month_cos" in prepared.columns
    assert prepared[["month_sin", "month_cos"]].notna().all().all()


def test_predict_from_monthly_frame_returns_friendly_payload() -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service)

    result = service.predict_from_monthly_frame(frame)

    assert result["headline"]["model_name"] == service.model_name
    assert result["headline"]["predicted_yield"] > 0
    assert len(result["prediction_rows"]) == 3
    assert len(result["feature_groups"]) >= 1
    assert result["monthly_features"][-1]["predicted_yield"] > 0
    assert result["drivers"][0]["label"]
    assert "current" in result["yield_series_by_model"]
    assert "lstm" in result["yield_series_by_model"]


def test_predict_from_directory_includes_multi_model_feature_forecasts(monkeypatch, tmp_path) -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service)
    source_files = [SimpleNamespace(path=tmp_path / "ag" / "2022_04.png", modality="ag", year=2022, month=4, day=None)]

    monkeypatch.setattr(service, "build_monthly_frame", lambda *args, **kwargs: (frame, source_files))
    monkeypatch.setattr(
        web_service_module,
        "build_forecast_from_monthly_features",
        lambda *args, **kwargs: SimpleNamespace(
            forecast_by_model={
                "lstm": pd.DataFrame([{"year": 2022, "month": 7, "month_label": "2022-07", service.feature_names[0]: 1.0}]),
                "transformer_encoder": pd.DataFrame([{"year": 2022, "month": 7, "month_label": "2022-07", service.feature_names[0]: 2.0}]),
                "gru": pd.DataFrame([{"year": 2022, "month": 7, "month_label": "2022-07", service.feature_names[0]: 3.0}]),
                "tiny_mamba_ssm": pd.DataFrame([{"year": 2022, "month": 7, "month_label": "2022-07", service.feature_names[0]: 4.0}]),
            },
            predictor_by_model={key: SimpleNamespace(model_name=key) for key in ("lstm", "transformer_encoder", "gru", "tiny_mamba_ssm")},
        ),
    )

    result = service.predict_from_directory(tmp_path, county_id="19001", crop_type="corn")

    assert set(result["feature_forecasts_by_model"]) == {"lstm", "transformer_encoder", "gru", "tiny_mamba_ssm"}
    assert result["feature_forecasts_by_model"]["tiny_mamba_ssm"][0]["month_label"] == "2022-07"


def test_tiny_mamba_ssm_architecture_is_supported() -> None:
    state_dict = {
        "blocks.0.in_proj.weight": pd.DataFrame([[0.0] * 35] * 64).to_numpy(),
        "head.2.bias": pd.Series([0.0] * 35).to_numpy(),
    }

    params = infer_architecture_from_state_dict("tiny_mamba_ssm", state_dict)

    assert params["input_dim"] == 35
    assert params["output_dim"] == 35


def test_tiny_mamba_ssm_checkpoint_loads_without_legacy_script() -> None:
    model = CropNetModelFactory.load_checkpoint("weights/tiny_mamba_ssm_best.pt", model_name="tiny_mamba_ssm", device="cpu")

    assert model.__class__.__name__ == "MambaStyleForecaster"
