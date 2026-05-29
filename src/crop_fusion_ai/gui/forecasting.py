"""Directory-based forecasting helpers for the desktop GUI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from crop_fusion_ai.gui.controller import PreprocessingController, UploadMetadata
from crop_fusion_ai.preprocessing import aggregate_monthly_feature_frame
from crop_fusion_ai.preprocessing import combine_modality_feature_frames
from cropnet_forecasting.data import prepare_monthly_features
from cropnet_forecasting.predictor import BlankFillPredictor

ModalityName = Literal["ag", "ndvi", "weather"]
ProgressCallback = Callable[[str, int, int, str], None]

_SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SUPPORTED_WEATHER_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather"}
_DATE_PATTERNS = (
    re.compile(r"(?P<year>20\d{2})[._-](?P<month>\d{1,2})[._-](?P<day>\d{1,2})"),
    re.compile(r"(?P<year>20\d{2})[._-](?P<month>\d{1,2})"),
    re.compile(r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})"),
)


@dataclass(frozen=True, slots=True)
class DirectorySample:
    """One file discovered inside a modality directory."""

    path: Path
    modality: ModalityName
    year: int | None
    month: int | None
    day: int | None = None


@dataclass(slots=True)
class DirectoryForecastResult:
    """Forecast artifacts produced from a directory analysis run."""

    monthly_features: pd.DataFrame
    forecast: pd.DataFrame
    predictor: BlankFillPredictor
    source_files: list[DirectorySample]


def _infer_modality(path: Path) -> ModalityName | None:
    parts = {part.lower() for part in path.parts}
    if "ag" in parts:
        return "ag"
    if "ndvi" in parts:
        return "ndvi"
    if "weather" in parts:
        return "weather"
    return None


def _parse_date_tokens(text: str) -> tuple[int | None, int | None, int | None]:
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        year = int(match.group("year"))
        month = int(match.group("month"))
        day_text = match.groupdict().get("day")
        day = int(day_text) if day_text else None
        return year, month, day
    return None, None, None


def _infer_sample_date(path: Path) -> tuple[int | None, int | None, int | None]:
    for candidate in (path.stem, path.name, *path.parts):
        year, month, day = _parse_date_tokens(candidate)
        if year is not None and month is not None:
            return year, month, day
    return None, None, None


def scan_directory(root_dir: str | Path) -> list[DirectorySample]:
    """Collect supported files from a directory tree with ag/ndvi/weather subfolders."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    samples: list[DirectorySample] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        modality = _infer_modality(path)
        if modality is None:
            continue

        suffix = path.suffix.lower()
        if modality in {"ag", "ndvi"} and suffix not in _SUPPORTED_IMAGE_SUFFIXES:
            continue
        if modality == "weather" and suffix not in _SUPPORTED_WEATHER_SUFFIXES:
            continue

        year, month, day = _infer_sample_date(path)
        samples.append(
            DirectorySample(
                path=path,
                modality=modality,
                year=year,
                month=month,
                day=day,
            )
        )
    return samples


def _extract_frames_for_sample(
    controller: PreprocessingController,
    sample: DirectorySample,
    *,
    county_id: str,
    crop_type: str,
) -> list[pd.DataFrame]:
    metadata = UploadMetadata(
        county_id=county_id,
        crop_type=crop_type,
        year=sample.year,
        month=sample.month,
    )
    if sample.modality == "ag":
        if sample.year is None or sample.month is None:
            raise ValueError(f"Could not infer year/month from AG file name: {sample.path.name}")
        return [controller.process_ag(sample.path, metadata)]
    if sample.modality == "ndvi":
        if sample.year is None or sample.month is None:
            raise ValueError(f"Could not infer year/month from NDVI file name: {sample.path.name}")
        return [controller.process_ndvi(sample.path, metadata)]
    return [controller.process_weather(sample.path, metadata)]


def build_monthly_features_from_directory(
    root_dir: str | Path,
    *,
    county_id: str,
    crop_type: str,
    controller: PreprocessingController | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, list[DirectorySample]]:
    """Build a merged monthly feature frame from ag/ndvi/weather files."""
    controller = controller or PreprocessingController()
    samples = scan_directory(root_dir)
    if not samples:
        raise ValueError("No supported AG, NDVI, or weather files were found in the directory.")

    if progress is not None:
        progress("scan", len(samples), len(samples), f"Discovered {len(samples)} supported files")

    modality_frames: dict[ModalityName, list[pd.DataFrame]] = {"ag": [], "ndvi": [], "weather": []}
    for index, sample in enumerate(samples, start=1):
        modality_frames[sample.modality].extend(
            _extract_frames_for_sample(
                controller,
                sample,
                county_id=county_id,
                crop_type=crop_type,
            )
        )
        if progress is not None:
            cache_note = "cache hit" if getattr(controller, "last_cache_hit", False) else "computed"
            progress(
                "preprocess",
                index,
                len(samples),
                f"Preprocessed {index}/{len(samples)} files ({cache_note})",
            )

    aggregated_frames: list[pd.DataFrame] = []
    modality_order: list[ModalityName] = ["ag", "ndvi", "weather"]
    for modality_index, modality in enumerate(modality_order, start=1):
        frames = modality_frames[modality]
        if not frames:
            if progress is not None:
                progress("aggregate", modality_index, len(modality_order), f"No {modality} rows to aggregate")
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)
        aggregated = aggregate_monthly_feature_frame(combined)
        if not aggregated.empty:
            aggregated_frames.append(aggregated)
        if progress is not None:
            progress(
                "aggregate",
                modality_index,
                len(modality_order),
                f"Aggregated {modality} features ({len(aggregated)} monthly rows)",
            )

    if not aggregated_frames:
        raise ValueError("No usable monthly features could be extracted from the directory.")

    monthly_features = combine_modality_feature_frames(*aggregated_frames)
    monthly_features = aggregate_monthly_feature_frame(monthly_features)
    if monthly_features.empty:
        raise ValueError("The merged monthly feature table is empty.")

    monthly_features = monthly_features.sort_values(["county_id", "crop_type", "year", "month"]).reset_index(drop=True)
    monthly_features["date"] = pd.to_datetime(
        monthly_features[["year", "month"]].assign(day=1)
    )
    if progress is not None:
        progress("prepare", len(monthly_features), len(monthly_features), f"Prepared {len(monthly_features)} monthly rows")
    return monthly_features, samples


def build_forecast_from_directory(
    root_dir: str | Path,
    *,
    county_id: str,
    crop_type: str,
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    config_path: str | Path,
    horizon: int = 12,
    device: str | None = None,
    controller: PreprocessingController | None = None,
    progress: ProgressCallback | None = None,
) -> DirectoryForecastResult:
    """Run the model on a directory of modality files and return forecast artifacts."""
    monthly_features, samples = build_monthly_features_from_directory(
        root_dir,
        county_id=county_id,
        crop_type=crop_type,
        controller=controller,
        progress=progress,
    )
    if progress is not None:
        progress("model_load", 0, 1, "Loading model and scaler")
    predictor = BlankFillPredictor.from_artifacts(
        checkpoint_path,
        scaler_path,
        config_path,
        device=device,
    )
    if progress is not None:
        progress(
            "feature_align",
            len(predictor.feature_names),
            len(predictor.feature_names),
            f"Aligned {len(predictor.feature_names)}/{len(predictor.feature_names)} model input features",
        )
    prepared = prepare_monthly_features(monthly_features, predictor.feature_names)
    if progress is not None:
        progress("forecast", 0, horizon, f"Forecasting next {horizon} months")
    forecast = predictor.predict_future_months(prepared, horizon=horizon, progress=progress)
    if progress is not None:
        progress("done", horizon, horizon, f"Forecast complete: {len(forecast)} rows")
    return DirectoryForecastResult(
        monthly_features=prepared,
        forecast=forecast,
        predictor=predictor,
        source_files=samples,
    )
