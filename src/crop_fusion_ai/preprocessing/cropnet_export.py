"""CropNet dataset export helpers for modality-specific JSONL generation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd

from crop_fusion_ai.preprocessing.ag import (
    derive_ag_time_series_features,
    extract_ag_features,
)
from crop_fusion_ai.preprocessing.ndvi import (
    derive_ndvi_time_series_features,
    extract_ndvi_features,
)
from crop_fusion_ai.preprocessing.usda_dataset import (
    SplitName,
    TargetKind,
    build_usda_records_from_frame,
    infer_usda_split,
    parse_usda_remote_path,
    select_usda_remote_files,
    write_jsonl_splits,
)
from crop_fusion_ai.preprocessing.weather import (
    derive_weather_time_series_features,
    extract_weather_features,
)

ModalityName = Literal["ag", "ndvi", "weather"]

_STATE_ABBR_BY_CODE: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

_MONTHLY_METADATA_COLUMNS = {"county_id", "crop_type", "year", "month"}


def get_state_abbr(state_code: str) -> str:
    """Return the two-letter state abbreviation for a numeric FIPS code."""
    try:
        return _STATE_ABBR_BY_CODE[str(state_code).zfill(2)]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"Unsupported state FIPS code: {state_code!r}"
        raise ValueError(msg) from exc


def normalize_fips_code(value: object) -> str:
    """Normalize a county FIPS value to a zero-padded five-character string."""
    text = str(value).strip()
    if not text:
        msg = "County FIPS code must not be empty"
        raise ValueError(msg)
    return text.zfill(5)


def extract_county_ids_from_usda_records(
    records: Sequence[Mapping[str, object]],
) -> list[str]:
    """Collect county ids from USDA records while preserving lexical order."""
    county_ids = {
        normalize_fips_code(record["county_id"])
        for record in records
        if record.get("county_id") is not None
    }
    return sorted(county_ids)


def resolve_usda_base_records(
    frame: pd.DataFrame,
    *,
    crop_type: str,
    year: int,
    split: SplitName,
    target_kind: TargetKind,
    source_path: str,
) -> list[dict[str, object]]:
    """Build compact USDA anchor records from a county-year crop frame."""
    return build_usda_records_from_frame(
        frame,
        crop_type=crop_type,
        year=year,
        split=split,
        target_kind=target_kind,
        source_path=source_path,
    )


def collect_usda_anchor_records(
    snapshot_root: Path,
    selected_paths: Sequence[str],
    *,
    target_kind: TargetKind,
) -> list[dict[str, object]]:
    """Load the selected USDA CSVs into JSONL-ready anchor records."""
    records: list[dict[str, object]] = []
    for remote_path in selected_paths:
        year, crop_type = parse_usda_remote_path(remote_path)
        split = infer_usda_split(year)
        local_path = snapshot_root / remote_path
        frame = pd.read_csv(local_path)
        records.extend(
            resolve_usda_base_records(
                frame,
                crop_type=crop_type,
                year=year,
                split=split,
                target_kind=target_kind,
                source_path=remote_path,
            )
        )
    return records


def build_usda_selection(
    remote_paths: Sequence[str],
    *,
    years: Sequence[int],
    crops: Sequence[str],
) -> list[str]:
    """Filter the repo tree down to the requested USDA crop-year slices."""
    return select_usda_remote_files(remote_paths, years=years, crops=crops)


def build_modality_allow_patterns(
    modality: ModalityName,
    *,
    years: Sequence[int],
    fips_codes: Sequence[str],
) -> list[str]:
    """Create Hugging Face allow-patterns for the requested modality files."""
    if not fips_codes:
        return []

    state_abbrs = sorted({get_state_abbr(code[:2]) for code in fips_codes})
    patterns: list[str] = []

    if modality in {"ag", "ndvi"}:
        image_type = "AG" if modality == "ag" else "NDVI"
        for year in years:
            for state_abbr in state_abbrs:
                patterns.extend(
                    [
                        f"Sentinel-2 Imagery/data/{image_type}/{year}/{state_abbr}/*.h5",
                    ]
                )
    elif modality == "weather":
        for year in years:
            for state_abbr in state_abbrs:
                patterns.extend(
                    [
                        f"WRF-HRRR Computed Dataset/data/{year}/{state_abbr}/*.csv",
                    ]
                )
    else:  # pragma: no cover - literal guard
        msg = f"Unsupported modality: {modality!r}"
        raise ValueError(msg)

    return patterns


def _normalize_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # pragma: no cover - defensive guard
        pass
    return value


def _json_safe_value(value: object) -> object:
    normalized = _normalize_scalar(value)
    if normalized is None:
        return None
    if isinstance(normalized, (bool, int, float, str)):
        return normalized
    if isinstance(normalized, Mapping):
        return {str(key): _json_safe_value(item) for key, item in normalized.items()}
    if isinstance(normalized, Sequence) and not isinstance(
        normalized, (bytes, bytearray, str)
    ):
        return [_json_safe_value(item) for item in normalized]
    return str(normalized)


def _aggregate_feature_rows(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("county_id", "crop_type", "year", "month"),
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    for column in group_columns:
        if column not in working.columns:
            working[column] = pd.NA

    numeric_columns = [
        column
        for column in working.columns
        if column not in group_columns
        and pd.api.types.is_numeric_dtype(working[column])
    ]
    if not numeric_columns:
        return (
            working.loc[:, list(group_columns)].drop_duplicates().reset_index(drop=True)
        )

    aggregated = (
        working.groupby(list(group_columns), dropna=False, as_index=False)[
            numeric_columns
        ]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    return aggregated


def _flatten_monthly_frame(
    monthly_frame: pd.DataFrame,
    *,
    prefix: str,
) -> dict[str, object]:
    flattened: dict[str, object] = {}
    if monthly_frame.empty:
        return flattened

    for row in monthly_frame.to_dict(orient="records"):
        month = int(_normalize_scalar(row.pop("month")))
        for column, value in row.items():
            if column in {"county_id", "crop_type", "year"}:
                continue
            flattened[f"{prefix}_m{month:02d}_{column}"] = _json_safe_value(value)

    flattened[f"{prefix}_month_count"] = int(monthly_frame["month"].nunique())
    flattened[f"{prefix}_observation_count"] = int(len(monthly_frame))
    return flattened


def _flatten_summary_frame(summary_frame: pd.DataFrame) -> dict[str, object]:
    if summary_frame.empty:
        return {}
    row = summary_frame.iloc[0].to_dict()
    return {
        str(column): _json_safe_value(value)
        for column, value in row.items()
        if column not in _MONTHLY_METADATA_COLUMNS and value is not None
    }


def _parse_month_from_date(date_key: object) -> int:
    text = str(date_key)
    if len(text) >= 7 and text[4] == "-":
        return int(text[5:7])
    if len(text) >= 2 and text[:2].isdigit():
        return int(text[5:7]) if len(text) >= 7 else int(text[:2])
    msg = f"Unrecognizable date key in Sentinel-2 file: {date_key!r}"
    raise ValueError(msg)


def _iter_sentinel_feature_rows(
    file_path: Path,
    *,
    fips_code: str,
    crop_type: str,
    year: int,
    modality: ModalityName,
) -> list[pd.DataFrame]:
    if modality not in {"ag", "ndvi"}:
        msg = f"Sentinel features are only valid for AG or NDVI, not {modality!r}"
        raise ValueError(msg)

    extractor = extract_ag_features if modality == "ag" else extract_ndvi_features
    rows: list[pd.DataFrame] = []

    with h5py.File(file_path, "r") as handle:
        if fips_code not in handle:
            return rows
        county_group = handle[fips_code]
        for date_key in county_group.keys():
            month = _parse_month_from_date(date_key)
            grids = np.asarray(county_group[date_key]["data"])
            for grid in grids:
                rows.append(
                    extractor(
                        grid,
                        county_id=fips_code,
                        crop_type=crop_type,
                        year=year,
                        month=month,
                    )
                )

    return rows


def _build_sentinel_feature_frame(
    snapshot_root: Path,
    *,
    modality: ModalityName,
    fips_code: str,
    crop_type: str,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_abbr = get_state_abbr(fips_code[:2])
    modality_dir = "AG" if modality == "ag" else "NDVI"
    candidates = sorted(
        snapshot_root.glob(
            f"**/Sentinel-2 Imagery/data/{modality_dir}/{year}/{state_abbr}/*.h5"
        )
    )
    if not candidates:
        return pd.DataFrame(), pd.DataFrame()

    feature_rows: list[pd.DataFrame] = []
    for file_path in candidates:
        feature_rows.extend(
            _iter_sentinel_feature_rows(
                file_path,
                fips_code=fips_code,
                crop_type=crop_type,
                year=year,
                modality=modality,
            )
        )

    if not feature_rows:
        return pd.DataFrame(), pd.DataFrame()

    per_scene = pd.concat(feature_rows, ignore_index=True)
    monthly = _aggregate_feature_rows(per_scene)
    if monthly.empty:
        return monthly, pd.DataFrame()

    summary = (
        derive_ag_time_series_features(monthly)
        if modality == "ag"
        else derive_ndvi_time_series_features(monthly)
    )
    return monthly, summary


def _read_weather_state_year_frame(
    snapshot_root: Path,
    *,
    fips_code: str,
    year: int,
) -> pd.DataFrame:
    state_abbr = get_state_abbr(fips_code[:2])
    candidates = sorted(
        snapshot_root.glob(
            f"**/WRF-HRRR Computed Dataset/data/{year}/{state_abbr}/*.csv"
        )
    )
    if not candidates:
        return pd.DataFrame()

    frames = [pd.read_csv(path) for path in candidates]
    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    frame["FIPS Code"] = frame["FIPS Code"].astype(str).str.zfill(5)
    return frame[frame["FIPS Code"] == fips_code].copy()


def _build_weather_feature_frame(
    snapshot_root: Path,
    *,
    fips_code: str,
    crop_type: str,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _read_weather_state_year_frame(
        snapshot_root,
        fips_code=fips_code,
        year=year,
    )
    if frame.empty:
        return frame, pd.DataFrame()

    monthly = extract_weather_features(
        frame,
        county_id=fips_code,
        crop_type=crop_type,
    )
    if monthly.empty:
        return monthly, pd.DataFrame()

    summary = derive_weather_time_series_features(monthly)
    return monthly, summary


def build_modality_features_for_record(
    snapshot_root: Path,
    record: Mapping[str, object],
    *,
    modality: ModalityName,
) -> dict[str, object] | None:
    """Build flattened modality features for a single USDA anchor record."""
    county_id_value = record.get("county_id")
    crop_type_value = record.get("crop_type")
    year_value = record.get("year")
    if county_id_value is None or crop_type_value is None or year_value is None:
        return None

    county_id = normalize_fips_code(county_id_value)
    crop_type = str(crop_type_value)
    year = int(year_value)

    if modality in {"ag", "ndvi"}:
        monthly_frame, summary_frame = _build_sentinel_feature_frame(
            snapshot_root,
            modality=modality,
            fips_code=county_id,
            crop_type=crop_type,
            year=year,
        )
    else:
        monthly_frame, summary_frame = _build_weather_feature_frame(
            snapshot_root,
            fips_code=county_id,
            crop_type=crop_type,
            year=year,
        )

    if monthly_frame.empty:
        return None

    features = _flatten_monthly_frame(monthly_frame, prefix=modality)
    features.update(_flatten_summary_frame(summary_frame))
    features[f"{modality}_available_months"] = int(monthly_frame["month"].nunique())
    features[f"{modality}_feature_rows"] = int(len(monthly_frame))

    payload: dict[str, object] = {
        "county_id": county_id,
        "crop_type": crop_type,
        "year": year,
        "split": record.get("split"),
        "target_kind": record.get("target_kind"),
        "target_value": record.get("target_value"),
        "target_unit": record.get("target_unit"),
        "source_path": record.get("source_path"),
    }
    for field in ("state_ansi", "county_ansi", "state_name", "county_name"):
        if record.get(field) is not None:
            payload[field] = record[field]
    payload["features"] = features
    return payload


def build_modality_records_by_split(
    snapshot_root: Path,
    usda_records: Sequence[Mapping[str, object]],
    *,
    modality: ModalityName,
) -> dict[str, list[dict[str, object]]]:
    """Build modality-specific records grouped by split."""
    records_by_split: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in usda_records:
        payload = build_modality_features_for_record(
            snapshot_root,
            record,
            modality=modality,
        )
        if payload is None:
            continue
        split = str(record.get("split"))
        if split not in records_by_split:
            continue
        records_by_split[split].append(payload)
    return records_by_split


def write_modality_jsonl_splits(
    records_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: Path,
) -> dict[str, int]:
    """Write train/validation/test JSONL files for a single modality."""
    return write_jsonl_splits(
        {
            "train": records_by_split.get("train", ()),
            "validation": records_by_split.get("validation", ()),
            "test": records_by_split.get("test", ()),
        },
        output_dir,
    )


def build_usda_records_from_snapshot(
    snapshot_root: Path,
    selected_paths: Sequence[str],
    *,
    target_kind: TargetKind,
) -> list[dict[str, object]]:
    """Backward-compatible helper for loading USDA-only anchor records."""
    return collect_usda_anchor_records(
        snapshot_root,
        selected_paths,
        target_kind=target_kind,
    )


def build_usda_records_from_dataframe(
    frame: pd.DataFrame,
    *,
    crop_type: str,
    year: int,
    split: SplitName,
    target_kind: TargetKind,
    source_path: str,
) -> list[dict[str, object]]:
    """Convert a local USDA dataframe into JSONL-ready anchor records."""
    return build_usda_records_from_frame(
        frame,
        crop_type=crop_type,
        year=year,
        split=split,
        target_kind=target_kind,
        source_path=source_path,
    )


def build_modality_records_from_frame(
    monthly_frame: pd.DataFrame,
    *,
    summary_frame: pd.DataFrame | None = None,
    modality: ModalityName,
    split: SplitName,
    crop_type: str,
    year: int,
    county_id: str,
    source_path: str,
    target_kind: TargetKind | None = None,
    target_value: float | None = None,
    target_unit: str | None = None,
    state_ansi: str | None = None,
    county_ansi: str | None = None,
    state_name: str | None = None,
    county_name: str | None = None,
) -> dict[str, object] | None:
    """Build a single modality record from a precomputed feature frame."""
    if monthly_frame.empty:
        return None

    if "month" not in monthly_frame.columns:
        return None

    features = _flatten_monthly_frame(monthly_frame, prefix=modality)
    if summary_frame is None:
        summary_frame = pd.DataFrame()
    features.update(_flatten_summary_frame(summary_frame))
    features[f"{modality}_available_months"] = int(monthly_frame["month"].nunique())
    features[f"{modality}_feature_rows"] = int(len(monthly_frame))

    payload: dict[str, object] = {
        "county_id": normalize_fips_code(county_id),
        "crop_type": crop_type,
        "year": int(year),
        "split": split,
        "source_path": source_path,
        "features": features,
    }
    if target_kind is not None:
        payload["target_kind"] = target_kind
    if target_value is not None:
        payload["target_value"] = target_value
    if target_unit is not None:
        payload["target_unit"] = target_unit
    if state_ansi is not None:
        payload["state_ansi"] = state_ansi
    if county_ansi is not None:
        payload["county_ansi"] = county_ansi
    if state_name is not None:
        payload["state_name"] = state_name
    if county_name is not None:
        payload["county_name"] = county_name
    return payload


def build_modality_record_from_sentinel_tensor(
    tensor: np.ndarray,
    *,
    modality: ModalityName,
    split: SplitName,
    crop_type: str,
    year: int,
    county_id: str,
    source_path: str,
    target_kind: TargetKind | None = None,
    target_value: float | None = None,
    target_unit: str | None = None,
    state_ansi: str | None = None,
    county_ansi: str | None = None,
    state_name: str | None = None,
    county_name: str | None = None,
) -> dict[str, object] | None:
    """Build a single AG/NDVI record from a retrieved CropNet tensor."""
    if modality not in {"ag", "ndvi"}:
        return None

    array = np.asarray(tensor)
    if array.ndim != 5:
        return None

    extractor = extract_ag_features if modality == "ag" else extract_ndvi_features
    feature_rows: list[pd.DataFrame] = []

    for month_index, temporal_slice in enumerate(array, start=1):
        if temporal_slice.ndim != 4:
            continue
        for grid in temporal_slice:
            feature_rows.append(
                extractor(
                    grid,
                    county_id=county_id,
                    crop_type=crop_type,
                    year=year,
                    month=month_index,
                )
            )

    if not feature_rows:
        return None

    monthly_frame = _aggregate_feature_rows(pd.concat(feature_rows, ignore_index=True))
    if monthly_frame.empty:
        return None

    summary_frame = (
        derive_ag_time_series_features(monthly_frame)
        if modality == "ag"
        else derive_ndvi_time_series_features(monthly_frame)
    )
    return build_modality_record_from_frame(
        monthly_frame,
        summary_frame=summary_frame,
        modality=modality,
        split=split,
        crop_type=crop_type,
        year=year,
        county_id=county_id,
        source_path=source_path,
        target_kind=target_kind,
        target_value=target_value,
        target_unit=target_unit,
        state_ansi=state_ansi,
        county_ansi=county_ansi,
        state_name=state_name,
        county_name=county_name,
    )


def build_modality_record_from_weather_frame(
    frame: pd.DataFrame,
    *,
    split: SplitName,
    crop_type: str,
    year: int,
    county_id: str,
    source_path: str,
    target_kind: TargetKind | None = None,
    target_value: float | None = None,
    target_unit: str | None = None,
    state_ansi: str | None = None,
    county_ansi: str | None = None,
    state_name: str | None = None,
    county_name: str | None = None,
) -> dict[str, object] | None:
    """Build a single weather record from a retrieved CropNet dataframe."""
    if frame.empty:
        return None

    monthly_frame = extract_weather_features(
        frame,
        county_id=county_id,
        crop_type=crop_type,
    )
    if monthly_frame.empty:
        return None

    summary_frame = derive_weather_time_series_features(monthly_frame)
    return build_modality_record_from_frame(
        monthly_frame,
        summary_frame=summary_frame,
        modality="weather",
        split=split,
        crop_type=crop_type,
        year=year,
        county_id=county_id,
        source_path=source_path,
        target_kind=target_kind,
        target_value=target_value,
        target_unit=target_unit,
        state_ansi=state_ansi,
        county_ansi=county_ansi,
        state_name=state_name,
        county_name=county_name,
    )
