"""Controller helpers that keep GUI actions testable without Tkinter state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from crop_fusion_ai.preprocessing.ag import extract_ag_features
from crop_fusion_ai.preprocessing.ndvi import extract_ndvi_features
from crop_fusion_ai.preprocessing.weather import extract_weather_features


@dataclass(frozen=True, slots=True)
class UploadMetadata:
    """Common metadata supplied through the GUI."""

    county_id: str | None = None
    crop_type: str | None = None
    year: int | None = None
    month: int | None = None


class PreprocessingController:
    """Thin wrapper around modality extractors used by the GUI and tests."""

    def process_ag(self, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame:
        return extract_ag_features(
            file_path,
            county_id=metadata.county_id,
            crop_type=metadata.crop_type,
            year=metadata.year,
            month=metadata.month,
        )

    def process_ndvi(self, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame:
        return extract_ndvi_features(
            file_path,
            county_id=metadata.county_id,
            crop_type=metadata.crop_type,
            year=metadata.year,
            month=metadata.month,
        )

    def process_weather(self, file_path: str | Path, metadata: UploadMetadata) -> pd.DataFrame:
        return extract_weather_features(
            file_path,
            county_id=metadata.county_id,
            crop_type=metadata.crop_type,
        )
