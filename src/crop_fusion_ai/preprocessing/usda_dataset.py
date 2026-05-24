"""Helpers for assembling the USDA tutorial dataset into JSONL records."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "validation", "test"]
TargetKind = Literal["yield", "production"]

_CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "corn": ("corn",),
    "cotton": ("cotton",),
    "soybeans": ("soybean", "soybeans"),
    "winter wheat": ("winterwheat", "winter wheat", "winter_wheat"),
}

_CROP_TARGET_COLUMNS: dict[str, dict[TargetKind, str]] = {
    "corn": {
        "yield": "YIELD, MEASURED IN BU / ACRE",
        "production": "PRODUCTION, MEASURED IN BU",
    },
    "cotton": {
        "yield": "YIELD, MEASURED IN LB / ACRE",
        "production": "PRODUCTION, MEASURED IN 480 LB BALES",
    },
    "soybeans": {
        "yield": "YIELD, MEASURED IN BU / ACRE",
        "production": "PRODUCTION, MEASURED IN BU",
    },
    "winter wheat": {
        "yield": "YIELD, MEASURED IN BU / ACRE",
        "production": "PRODUCTION, MEASURED IN BU",
    },
}

_CROP_TARGET_UNITS: dict[str, dict[TargetKind, str]] = {
    "corn": {
        "yield": "BU / ACRE",
        "production": "BU",
    },
    "cotton": {
        "yield": "LB / ACRE",
        "production": "480 LB BALES",
    },
    "soybeans": {
        "yield": "BU / ACRE",
        "production": "BU",
    },
    "winter wheat": {
        "yield": "BU / ACRE",
        "production": "BU",
    },
}

_SPLIT_BY_YEAR: dict[int, SplitName] = {
    2017: "train",
    2018: "train",
    2019: "train",
    2020: "train",
    2021: "validation",
    2022: "test",
}

_TOP_LEVEL_FIELDS = {
    "split",
    "county_id",
    "crop_type",
    "year",
    "target_kind",
    "target_value",
    "target_unit",
    "source_path",
    "state_ansi",
    "county_ansi",
    "state_name",
    "county_name",
    "commodity_desc",
    "agg_level_desc",
    "domain_desc",
    "source_desc",
}


@dataclass(frozen=True, slots=True)
class UsdaCropSpec:
    """Metadata for a USDA crop family."""

    canonical_name: str
    aliases: tuple[str, ...]
    target_columns: dict[TargetKind, str]
    target_units: dict[TargetKind, str]


def _normalize_key(value: str) -> str:
    return "".join(
        character for character in value.strip().lower() if character.isalnum()
    )


def get_usda_crop_spec(crop_type: str) -> UsdaCropSpec:
    """Return the canonical USDA crop spec for a crop label or alias."""
    normalized = _normalize_key(crop_type)
    for canonical_name, aliases in _CROP_ALIASES.items():
        if normalized == _normalize_key(canonical_name) or normalized in {
            _normalize_key(alias) for alias in aliases
        }:
            return UsdaCropSpec(
                canonical_name=canonical_name,
                aliases=aliases,
                target_columns=_CROP_TARGET_COLUMNS[canonical_name],
                target_units=_CROP_TARGET_UNITS[canonical_name],
            )

    msg = f"Unsupported USDA crop type: {crop_type!r}"
    raise ValueError(msg)


def normalize_crop_type(crop_type: str) -> str:
    """Return the canonical crop label used in JSONL records."""
    return get_usda_crop_spec(crop_type).canonical_name


def infer_usda_split(year: int) -> SplitName:
    """Map a USDA year to the train/validation/test split requested by the user."""
    try:
        return _SPLIT_BY_YEAR[year]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"Unsupported USDA year for splitting: {year}"
        raise ValueError(msg) from exc


def select_usda_remote_files(
    remote_paths: Iterable[str],
    *,
    years: Sequence[int],
    crops: Sequence[str],
) -> list[str]:
    """Filter repo file paths down to the requested USDA year/crop slices."""
    selected_years = {str(year) for year in years}
    selected_crops = {
        alias
        for crop_type in crops
        for alias in get_usda_crop_spec(crop_type).aliases
    }

    selected: list[str] = []
    for remote_path in remote_paths:
        parts = Path(remote_path).parts
        file_token = _normalize_key(parts[-1])
        year: str | None = None
        folder_token = ""

        if len(parts) >= 4 and parts[0] == "USDA" and parts[1] == "data":
            year = parts[2]
            folder_token = _normalize_key(parts[3])
        elif len(parts) >= 4 and parts[0] == "USDA Crop Dataset":
            year = parts[2]
            folder_token = _normalize_key(parts[1])
        elif len(parts) >= 5 and parts[0] == "USDA" and parts[1] == "Crop Dataset":
            year = parts[3]
            folder_token = _normalize_key(parts[2])

        if year is None or year not in selected_years:
            continue

        folder_match = any(
            _normalize_key(alias) in folder_token for alias in selected_crops
        )
        file_match = any(
            _normalize_key(alias) in file_token for alias in selected_crops
        )
        if folder_match or file_match:
            selected.append(remote_path)

    return sorted(set(selected))


def parse_usda_remote_path(remote_path: str) -> tuple[int, str]:
    """Extract the USDA year and canonical crop label from a repo path."""
    parts = Path(remote_path).parts
    if len(parts) >= 4 and parts[0] == "USDA" and parts[1] == "data":
        year = int(parts[2])
        crop_type = normalize_crop_type(parts[3])
        return year, crop_type

    if len(parts) >= 4 and parts[0] == "USDA Crop Dataset":
        year = int(parts[2])
        crop_type = normalize_crop_type(parts[1])
        return year, crop_type

    if len(parts) >= 5 and parts[0] == "USDA" and parts[1] == "Crop Dataset":
        year = int(parts[3])
        crop_type = normalize_crop_type(parts[2])
        return year, crop_type

    msg = f"Not a USDA CropNet data path: {remote_path}"
    raise ValueError(msg)


def infer_county_id(row: Mapping[str, object]) -> str | None:
    """Build a county FIPS code from the USDA row when possible."""
    if "county_id" in row and row["county_id"] is not None:
        county_id = _normalize_scalar(row["county_id"])
        if county_id is not None:
            return str(county_id).zfill(5)

    state_ansi = row.get("state_ansi")
    county_ansi = row.get("county_ansi")
    if state_ansi is not None and county_ansi is not None:
        state_code = _to_zero_padded_code(state_ansi, 2)
        county_code = _to_zero_padded_code(county_ansi, 3)
        return f"{state_code}{county_code}"

    return None


def _to_zero_padded_code(value: object, width: int) -> str:
    normalized = _normalize_scalar(value)
    if normalized is None:
        return ""
    try:
        return str(int(float(str(normalized)))).zfill(width)
    except ValueError:
        return str(normalized).zfill(width)


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

    if isinstance(normalized, (bool, int, str)):
        return normalized

    if isinstance(normalized, float):
        return normalized if math.isfinite(normalized) else None

    if isinstance(normalized, Path):
        return str(normalized)

    if isinstance(normalized, pd.Timestamp):
        return normalized.isoformat()

    if isinstance(normalized, Mapping):
        return {str(key): _json_safe_value(item) for key, item in normalized.items()}

    if isinstance(normalized, Sequence) and not isinstance(
        normalized, (bytes, bytearray, str)
    ):
        return [_json_safe_value(item) for item in normalized]

    return str(normalized)


def _coerce_numeric(value: object, *, field_name: str) -> float:
    normalized = _normalize_scalar(value)
    if normalized is None:
        msg = f"Missing USDA target value for {field_name}"
        raise ValueError(msg)

    try:
        numeric = float(normalized)
    except (TypeError, ValueError) as exc:
        msg = f"USDA target column {field_name!r} must be numeric"
        raise ValueError(msg) from exc

    if not math.isfinite(numeric):
        msg = f"USDA target column {field_name!r} must be finite"
        raise ValueError(msg)

    return numeric


def _copy_present_fields(
    row: Mapping[str, object],
    fields: Sequence[str],
) -> dict[str, object]:
    copied: dict[str, object] = {}
    for field in fields:
        if field in row and row[field] is not None:
            normalized = _json_safe_value(row[field])
            if normalized is not None:
                copied[field] = normalized
    return copied


def resolve_usda_target_column(crop_type: str, target_kind: TargetKind) -> str:
    """Return the USDA column name used for the requested crop/target pair."""
    return get_usda_crop_spec(crop_type).target_columns[target_kind]


def resolve_usda_target_unit(crop_type: str, target_kind: TargetKind) -> str:
    """Return the human-readable unit for a USDA crop/target pair."""
    return get_usda_crop_spec(crop_type).target_units[target_kind]


def filter_usda_rows(frame: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    """Keep the county-total rows that are actually labeled for training."""
    filtered = frame.copy()
    if target_column not in filtered.columns:
        msg = f"Required USDA target column {target_column!r} is missing"
        raise ValueError(msg)

    filtered = filtered[filtered[target_column].notna()].copy()
    if filtered.empty:
        return filtered

    if "agg_level_desc" in filtered.columns:
        county_mask = filtered["agg_level_desc"].astype(str).str.upper() == "COUNTY"
        county_rows = filtered[county_mask]
        if not county_rows.empty:
            filtered = county_rows

    if "domain_desc" in filtered.columns:
        total_mask = filtered["domain_desc"].astype(str).str.upper() == "TOTAL"
        total_rows = filtered[total_mask]
        if not total_rows.empty:
            filtered = total_rows

    return filtered.reset_index(drop=True)


def build_usda_record(
    row: Mapping[str, object],
    *,
    crop_type: str,
    year: int,
    split: SplitName,
    target_kind: TargetKind,
    source_path: str,
) -> dict[str, object]:
    """Convert a single USDA row into a JSONL-ready training record."""
    spec = get_usda_crop_spec(crop_type)
    target_column = spec.target_columns[target_kind]
    target_value = _coerce_numeric(row.get(target_column), field_name=target_column)

    record: dict[str, object] = {
        "split": split,
        "county_id": infer_county_id(row),
        "crop_type": spec.canonical_name,
        "year": int(year),
        "target_kind": target_kind,
        "target_value": target_value,
        "target_unit": spec.target_units[target_kind],
        "source_path": source_path,
    }
    record.update(
        _copy_present_fields(
            row,
            (
                "state_name",
                "county_name",
                "commodity_desc",
                "agg_level_desc",
                "domain_desc",
                "source_desc",
            ),
        )
    )
    if row.get("state_ansi") is not None:
        record["state_ansi"] = _to_zero_padded_code(row["state_ansi"], 2)
    if row.get("county_ansi") is not None:
        record["county_ansi"] = _to_zero_padded_code(row["county_ansi"], 3)

    excluded_fields = _TOP_LEVEL_FIELDS | {target_column}
    features = {
        str(key): _json_safe_value(value)
        for key, value in row.items()
        if key not in excluded_fields
    }
    features = {key: value for key, value in features.items() if value is not None}
    if features:
        record["features"] = features

    return record


def build_usda_records_from_frame(
    frame: pd.DataFrame,
    *,
    crop_type: str,
    year: int,
    split: SplitName,
    target_kind: TargetKind,
    source_path: str,
) -> list[dict[str, object]]:
    """Turn a filtered USDA dataframe into JSONL-ready records."""
    target_column = resolve_usda_target_column(crop_type, target_kind)
    filtered = filter_usda_rows(frame, target_column=target_column)
    return [
        build_usda_record(
            row,
            crop_type=crop_type,
            year=year,
            split=split,
            target_kind=target_kind,
            source_path=source_path,
        )
        for row in filtered.to_dict(orient="records")
    ]


def write_jsonl_records(
    records: Iterable[Mapping[str, object]],
    output_path: Path,
) -> int:
    """Write JSONL records and return the number of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe_value(record), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def write_jsonl_splits(
    records_by_split: Mapping[SplitName, Sequence[Mapping[str, object]]],
    output_dir: Path,
) -> dict[str, int]:
    """Write the standard train/validation/test JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        split_path = output_dir / f"{split}.jsonl"
        counts[split] = write_jsonl_records(records_by_split.get(split, ()), split_path)
    return counts
