"""Controller helpers that keep GUI actions testable without Tkinter state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crop_fusion_ai.preprocessing.ag import extract_ag_features
from crop_fusion_ai.preprocessing.ndvi import extract_ndvi_features
from crop_fusion_ai.preprocessing.weather import extract_weather_features

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "gui_features"
CACHE_VERSION = 1


@dataclass(frozen=True, slots=True)
class UploadMetadata:
    """Common metadata supplied through the GUI."""

    county_id: str | None = None
    crop_type: str | None = None
    year: int | None = None
    month: int | None = None


class FeatureCache:
    """Tiny on-disk cache for per-file preprocessing results."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, *, modality: str, file_path: str | Path, metadata: UploadMetadata) -> str:
        path = Path(file_path)
        stat = path.stat()
        payload = {
            "version": CACHE_VERSION,
            "modality": modality,
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "county_id": metadata.county_id,
            "crop_type": metadata.crop_type,
            "year": metadata.year,
            "month": metadata.month,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return digest

    def _path_for(self, modality: str, key: str) -> Path:
        return self.cache_dir / modality / f"{key}.pkl"

    def load(self, modality: str, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame | None:
        key = self._cache_key(modality=modality, file_path=file_path, metadata=metadata)
        cache_path = self._path_for(modality, key)
        if not cache_path.exists():
            return None
        return pd.read_pickle(cache_path)

    def save(self, modality: str, file_path: str | Path, metadata: UploadMetadata, frame: pd.DataFrame) -> None:
        key = self._cache_key(modality=modality, file_path=file_path, metadata=metadata)
        cache_path = self._path_for(modality, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        frame.to_pickle(temp_path)
        temp_path.replace(cache_path)


class PreprocessingController:
    """Thin wrapper around modality extractors used by the GUI and tests."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.cache = FeatureCache(cache_dir)
        self.last_cache_hit: bool = False

    def _cached_extract(
        self,
        modality: str,
        file_path: str | Path,
        metadata: UploadMetadata,
        extractor,
    ) -> pd.DataFrame:
        cached = self.cache.load(modality, file_path, metadata)
        if cached is not None:
            self.last_cache_hit = True
            return cached.copy()

        self.last_cache_hit = False
        frame = extractor(file_path, metadata)
        self.cache.save(modality, file_path, metadata, frame)
        return frame.copy()

    def process_ag(self, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame:
        return self._cached_extract(
            "ag",
            file_path,
            metadata,
            lambda path, meta: extract_ag_features(
                path,
                county_id=meta.county_id,
                crop_type=meta.crop_type,
                year=meta.year,
                month=meta.month,
            ),
        )

    def process_ndvi(self, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame:
        return self._cached_extract(
            "ndvi",
            file_path,
            metadata,
            lambda path, meta: extract_ndvi_features(
                path,
                county_id=meta.county_id,
                crop_type=meta.crop_type,
                year=meta.year,
                month=meta.month,
            ),
        )

    def process_weather(self, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame:
        return self._cached_extract(
            "weather",
            file_path,
            metadata,
            lambda path, meta: extract_weather_features(
                path,
                county_id=meta.county_id,
                crop_type=meta.crop_type,
            ),
        )
