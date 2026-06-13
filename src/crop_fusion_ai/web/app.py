"""FastAPI application for the browser-first TaoCrop dashboard."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import uuid
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from crop_fusion_ai.web.chat import (
    build_dashboard_context,
    chat_with_ollama,
    list_models,
    model_info,
)
from crop_fusion_ai.web.feature_labels import FEATURE_GROUPS, label_payload
from crop_fusion_ai.web.service import YieldModelService


STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_STAGE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "web_uploads"
MAX_UPLOAD_FILES = 10000
MAX_UPLOAD_FIELDS = 1000
MAX_UPLOAD_PART_SIZE = 25 * 1024 * 1024


class ChatMessagePayload(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1)


class ChatRequestPayload(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessagePayload] = Field(default_factory=list)
    dashboard_context: dict[str, Any] | None = None
    base_url: str | None = None


@dataclass(slots=True)
class JobState:
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int
    total: int
    message: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class UploadedBlob:
    relative_path: str
    content: bytes


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    def create(self) -> JobState:
        now = self._now()
        job = JobState(
            job_id=uuid.uuid4().hex,
            status="queued",
            progress=0,
            total=100,
            message="Queued",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobState:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(job_id) from exc

    def update(self, job_id: str, **changes: Any) -> JobState:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = self._now()
            return job

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


class YieldWebApp:
    def __init__(self) -> None:
        self.service = YieldModelService()
        self.jobs = JobStore()

    def submit_directory_job(
        self,
        *,
        county_id: str,
        crop_type: str,
        builder,
    ) -> JobState:
        job = self.jobs.create()

        def run() -> None:
            try:
                self.jobs.update(job.job_id, status="running", progress=5, message="Starting analysis")
                result = builder(self.service, self.jobs, job.job_id, county_id, crop_type)
                self.jobs.update(
                    job.job_id,
                    status="completed",
                    progress=100,
                    message="Prediction complete",
                    result=result,
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001
                self.jobs.update(
                    job.job_id,
                    status="failed",
                    progress=100,
                    message="Prediction failed",
                    error=str(exc),
                    result=None,
                )

        threading.Thread(target=run, daemon=True).start()
        return job


def _root_html() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _job_payload(job: JobState) -> dict[str, Any]:
    return asdict(job)


def _write_uploaded_folder(upload_dir: Path, uploads: list[UploadedBlob]) -> None:
    if not uploads:
        raise ValueError("No files were uploaded")
    for upload in uploads:
        clean_path = Path(upload.relative_path)
        if clean_path.is_absolute() or ".." in clean_path.parts:
            raise ValueError(f"Invalid relative path: {upload.relative_path}")
        destination = upload_dir / clean_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(upload.content)


def _stage_uploaded_folder(uploads: list[UploadedBlob], *, county_id: str, crop_type: str) -> Path:
    if not uploads:
        raise ValueError("No files were uploaded")

    digest = hashlib.sha256()
    digest.update(county_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(crop_type.encode("utf-8"))
    for upload in uploads:
        digest.update(b"\0")
        digest.update(upload.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(upload.content)

    cache_dir = UPLOAD_STAGE_DIR / digest.hexdigest()
    marker = cache_dir / ".complete"
    if marker.exists():
        return cache_dir
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crop-fusion-stage-", dir=cache_dir.parent) as temp_root:
        staging_dir = Path(temp_root) / cache_dir.name
        _write_uploaded_folder(staging_dir, uploads)
        marker_path = staging_dir / ".complete"
        marker_path.write_text("ok", encoding="utf-8")
        staging_dir.replace(cache_dir)

    return cache_dir


def _collect_uploaded_blobs(files: list[UploadFile], relative_paths: list[str]) -> list[UploadedBlob]:
    if len(relative_paths) != len(files):
        raise ValueError("relative path count does not match uploaded files")
    uploads: list[UploadedBlob] = []
    for file, relative_path in zip(files, relative_paths, strict=True):
        try:
            content = file.file.read()
        finally:
            file.file.close()
        uploads.append(UploadedBlob(relative_path=relative_path, content=content))
    return uploads


def _build_upload_result(
    service: YieldModelService,
    jobs: JobStore,
    job_id: str,
    county_id: str,
    crop_type: str,
    *,
    uploads: list[UploadedBlob],
) -> dict[str, Any]:
    staged_dir = _stage_uploaded_folder(uploads, county_id=county_id, crop_type=crop_type)

    def progress(stage: str, current: int, total: int, message: str) -> None:
        percent = 10 if total <= 0 else int((current / total) * 80 / max(total, 1)) + 10
        jobs.update(job_id, progress=min(95, max(10, percent)), message=message)

    return service.predict_from_directory(staged_dir, county_id=county_id, crop_type=crop_type, progress=progress)


def create_app() -> FastAPI:
    web = YieldWebApp()
    app = FastAPI(title="TaoCrop", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root() -> FileResponse:
        return _root_html()

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return {
            "app_name": "TaoCrop",
            "tagline": "AI-powered yield predictor for farmers.",
            "brand": {
                "background": "#f7f1e6",
                "surface": "#fffaf2",
                "accent": "#d9772b",
                "accent_soft": "#f4d2b0",
                "ink": "#2b2118",
            },
            "feature_labels": label_payload(),
            "feature_groups": [
                {
                    "key": group,
                    "label": {
                        "canopy": "Crop canopy",
                        "vegetation": "Vegetation health",
                        "weather": "Weather conditions",
                        "season": "Season timing",
                    }.get(group, group.title()),
                }
                for group in FEATURE_GROUPS
            ],
            "defaults": {
                "county_id": "19001",
                "crop_type": "corn",
            },
            "model": web.service.summary.to_payload(),
        }

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        try:
            job = web.jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc
        return _job_payload(job)

    @app.post("/api/predict/upload")
    async def predict_upload(request: Request) -> dict[str, Any]:
        form = await request.form(
            max_files=MAX_UPLOAD_FILES,
            max_fields=MAX_UPLOAD_FIELDS,
            max_part_size=MAX_UPLOAD_PART_SIZE,
        )

        county_id = str(form.get("county_id") or "").strip()
        crop_type = str(form.get("crop_type") or "").strip()
        relative_paths = str(form.get("relative_paths") or "[]")
        files = form.getlist("files")

        if not county_id:
            raise HTTPException(status_code=400, detail="county_id is required")
        if not crop_type:
            raise HTTPException(status_code=400, detail="crop_type is required")
        if not files:
            raise HTTPException(status_code=400, detail="No files were uploaded")
        try:
            parsed_paths = json.loads(relative_paths)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="relative_paths must be JSON") from exc
        if not isinstance(parsed_paths, list) or not all(isinstance(item, str) for item in parsed_paths):
            raise HTTPException(status_code=400, detail="relative_paths must be a JSON array of strings")
        if len(parsed_paths) != len(files):
            raise HTTPException(status_code=400, detail="relative_paths must match the uploaded files")

        uploads = _collect_uploaded_blobs(files, parsed_paths)

        job = web.submit_directory_job(
            county_id=county_id,
            crop_type=crop_type,
            builder=lambda service, jobs, job_id, county_id, crop_type: _build_upload_result(
                service,
                jobs,
                job_id,
                county_id,
                crop_type,
                uploads=uploads,
            ),
        )
        return {"job_id": job.job_id, "status": job.status}

    @app.get("/api/chat/models")
    def api_chat_models() -> dict[str, Any]:
        try:
            models = list_models()
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "models": [
                {
                    "name": str(model.get("name") or model.get("model") or ""),
                    "model": str(model.get("model") or model.get("name") or ""),
                    "size": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "details": model.get("details") or {},
                }
                for model in models
                if model.get("name") or model.get("model")
            ]
        }

    @app.get("/api/chat/models/{model_name}")
    def api_chat_model_info(model_name: str) -> dict[str, Any]:
        try:
            return model_info(model_name)
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/chat")
    def api_chat(payload: ChatRequestPayload) -> dict[str, Any]:
        base_url = payload.base_url or "http://localhost:11434"
        try:
            response = chat_with_ollama(
                model=payload.model,
                messages=[message.model_dump() for message in payload.messages],
                dashboard_context=build_dashboard_context(payload.dashboard_context),
                base_url=base_url,
            )
        except ConnectionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    uvicorn.run(
        "crop_fusion_ai.web.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
