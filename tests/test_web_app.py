"""Tests for the browser app surface."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

import crop_fusion_ai.web.app as web_app_module
from crop_fusion_ai.web.app import UploadedBlob, _stage_uploaded_folder, create_app


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
    def __init__(self) -> None:
        self.summary = _DummySummary()

    def predict_from_directory(self, root_dir, *, county_id: str, crop_type: str, progress=None):  # noqa: ANN001
        if progress is not None:
            progress("scan", 1, 1, "Scanning uploaded files")
        return {
            "headline": {"predicted_yield": 123.4, "unit": "bu/acre", "year": 2022, "month": 9, "month_label": "2022-09"},
            "yield_series": [],
            "feature_groups": [],
            "summary": self.summary.to_payload(),
            "drivers": [],
            "monthly_features": [],
            "feature_importance": [],
        }


def test_health_and_config_endpoints_work() -> None:
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
    relative_paths = [f"nested/folder/file_{index}.txt" for index in range(1001)]
    files = [
        ("files", (f"file_{index}.txt", b"x", "text/plain"))
        for index in range(1001)
    ]

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/predict/upload",
            data={
                "county_id": "19001",
                "crop_type": "corn",
                "relative_paths": json.dumps(relative_paths),
            },
            files=files,
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]
        job = None
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}")
            assert job.status_code == 200
            if job.json()["status"] == "completed":
                break
            time.sleep(0.05)
    assert job is not None
    assert job.status_code == 200
    assert job.json()["status"] == "completed"


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
