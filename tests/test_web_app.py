"""Tests for the browser app surface."""

from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

import crop_fusion_ai.web.app as web_app_module
from crop_fusion_ai.web.app import UploadedBlob, _build_upload_result, _stage_uploaded_folder, create_app


class _DummySummary:
    def to_payload(self) -> dict[str, object]:
        return {
            "model_name": "Dummy",
            "feature_count": 0,
            "target_units": "BU / ACRE",
            "holdout": {"rmse": 1.0, "mae": 1.0, "r2": 0.5},
            "best_model": {"model": "Dummy", "model_type": "ml", "rmse": 1.0, "mae": 1.0, "r2": 0.5, "mape": 0.0},
            "top_features": [],
        }


class _DummyService:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.summary = _DummySummary()

    def build_monthly_frame(self, root_dir, *, county_id: str, crop_type: str, progress=None):  # noqa: ANN001
        del root_dir, county_id, crop_type
        if progress is not None:
            progress("scan", 1, 1, "Scanning uploaded files")
        frame = pd.DataFrame(
            [
                {"county_id": "19001", "crop_type": "corn", "year": 2022, "month": 9},
            ]
        )
        return frame, []

    def prepare_monthly_frame(self, monthly_frame):  # noqa: ANN001
        return monthly_frame

    def predict_from_directory(self, root_dir, *, county_id: str, crop_type: str, progress=None, **kwargs):  # noqa: ANN001
        del root_dir, county_id, kwargs
        if progress is not None:
            progress("scan", 1, 1, "Scanning uploaded files")
        return {
            "headline": {"predicted_yield": 123.4, "unit": "bu/acre", "year": 2022, "month": 9, "month_label": "2022-09"},
            "yield_series": [],
            "yield_trajectory_by_model": {"current": [{"predicted_yield": 123.4, "month_label": "2022-09"}]},
            "summary": self.summary.to_payload(),
            "benchmark_summary": self.summary.to_payload(),
            "feature_groups": [],
            "feature_model_runs": [],
            "feature_model_best": {},
            "drivers": [],
            "monthly_features": [],
            "feature_importance": [],
            "feature_forecasts_by_model": {},
            "derived_drivers_by_model": {},
            "feature_forecast_models": [],
            "derived_driver_models": [],
            "crop_type": crop_type,
        }


def _install_chat_stubs(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "list_models", lambda: [{"name": "llama3.1", "model": "llama3.1", "details": {"parameter_size": "8B"}}])
    monkeypatch.setattr(
        web_app_module,
        "model_info",
        lambda model_name: {
            "model": model_name,
            "context_length": 8192,
            "parameters": "num_ctx 8192",
            "capabilities": ["chat"],
            "details": {"parameter_size": "8B"},
            "raw": {"parameters": "num_ctx 8192"},
        },
    )
    monkeypatch.setattr(
        web_app_module,
        "chat_with_ollama",
        lambda **kwargs: {
            "model": kwargs["model"],
            "reply": "The crop looks healthy.",
            "stats": {
                "input_tokens": 120,
                "output_tokens": 24,
                "tokens_per_second": 18.5,
                "context_length": 8192,
            },
        },
    )


def test_health_and_config_endpoints_work(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "YieldModelService", _DummyService)
    with TestClient(create_app()) as client:
        health = client.get("/healthz")
        config = client.get("/api/config")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert config.status_code == 200
    payload = config.json()
    assert payload["app_name"] == "TaoCrop"
    assert payload["model"]["model_name"]


def test_upload_endpoint_accepts_large_folders(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "YieldModelService", _DummyService)
    monkeypatch.setattr(
        web_app_module,
        "build_forecast_from_monthly_features",
        lambda *args, **kwargs: type(
            "DummyForecasts",
            (),
            {
                "forecast_by_model": {
                    key: pd.DataFrame([{"year": 2022, "month": 10, "month_label": "2022-10"}])
                    for key in ("lstm", "gru", "tiny_mamba_ssm", "transformer_encoder")
                },
                "derived_drivers_by_model": {},
                "predictor_by_model": {
                    key: type("DummyPredictor", (), {"model_name": key, "supports_latent_state": True})()
                    for key in ("lstm", "gru", "tiny_mamba_ssm", "transformer_encoder")
                },
            },
        )(),
    )
    relative_paths = ["nested/folder/file_0.txt", "nested/folder/file_1.txt"]
    uploads = [
        web_app_module.UploadedBlob(relative_path=path, content=b"x")
        for path in relative_paths
    ]
    services = {"corn": _DummyService(), "soybeans": _DummyService()}
    jobs = web_app_module.JobStore()
    job = jobs.create()
    payload = _build_upload_result(
        services,
        jobs,
        job.job_id,
        "19001",
        "corn",
        uploads=uploads,
    )
    assert payload["default_crop"] == "corn"
    assert set(payload["crops"]) == {"corn", "soybeans"}
    assert "current" in payload["crops"]["corn"]["yield_trajectory_by_model"]


def test_staged_upload_folder_is_stable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "UPLOAD_STAGE_DIR", tmp_path / "stage")
    uploads = [
        UploadedBlob(relative_path="ag/2017_12_21.png", content=b"ag-bytes"),
        UploadedBlob(relative_path="ndvi/2017_12_21.png", content=b"ndvi-bytes"),
    ]

    first = _stage_uploaded_folder(uploads, county_id="19001", crop_type="corn")
    second = _stage_uploaded_folder(uploads, county_id="19001", crop_type="corn")

    assert first == second
    assert (first / ".complete").exists()


def test_chat_endpoints_return_models_and_reply(monkeypatch) -> None:
    monkeypatch.setattr(web_app_module, "YieldModelService", _DummyService)
    _install_chat_stubs(monkeypatch)

    with TestClient(create_app()) as client:
        models = client.get("/api/chat/models")
        info = client.get("/api/chat/models/llama3.1")
        reply = client.post(
            "/api/chat",
            json={
                "model": "llama3.1",
                "messages": [{"role": "user", "content": "What do you see?"}],
                "dashboard_context": {
                    "headline": {"predicted_yield": 123.4},
                    "monthly_features": [],
                },
            },
        )

    assert models.status_code == 200
    assert models.json()["models"][0]["name"] == "llama3.1"
    assert info.status_code == 200
    assert info.json()["context_length"] == 8192
    assert reply.status_code == 200
    assert reply.json()["reply"] == "The crop looks healthy."
