from __future__ import annotations

import json
import os
import multiprocessing as mp
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from crop_fusion_ai.preprocessing import (
    aggregate_monthly_feature_frame,
    combine_modality_feature_frames,
    extract_ag_features,
    extract_ndvi_features,
    extract_weather_features,
)
from crop_fusion_ai.preprocessing.usda_dataset import (
    build_usda_records_from_frame,
    infer_usda_split,
    parse_usda_remote_path,
    select_usda_remote_files,
)

from .data import read_table
from .features import FEATURE_COLS, META_COLS, selected_feature_columns

DEFAULT_SOURCE_TABLE = Path("data/raw/cropnet/monthly_feature_table.parquet")
DEFAULT_RAW_ROOT = Path("data/raw/cropnet")
DEFAULT_OUTPUT_DIR = Path("data/training")
DEFAULT_CACHE_DIR = Path("data/cache/cropnet_prepare")
DEFAULT_REPO_ID = "CropNet/CropNet"
DEFAULT_CROP_TYPE = "corn"
DEFAULT_STATE_CODES = ("IA",)
DEFAULT_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)

_STATE_ABBR_BY_CODE = {
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

GENERATED_FORECAST_COLUMNS = {
    "forecast_step",
    "known_months",
    "source_note",
    "y_pred",
}


@dataclass(frozen=True, slots=True)
class PreparedDatasetPaths:
    output_dir: Path
    all_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    scaler_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class PreparedRawDataset:
    raw_dir: Path
    download_manifest_path: Path
    monthly_table_path: Path
    counties: list[str]
    years: list[int]
    state_codes: list[str]
    crop_type: str


def _normalize_state_code(value: str) -> str:
    text = str(value).strip().upper()
    if not text:
        raise ValueError("State code must not be empty.")
    if len(text) == 2 and text.isalpha():
        return text
    if len(text) <= 2 and text.isdigit():
        try:
            return _STATE_ABBR_BY_CODE[text.zfill(2)]
        except KeyError as exc:
            raise ValueError(f"Unsupported state FIPS code: {value!r}") from exc
    raise ValueError(f"Unsupported state code: {value!r}")


def normalize_state_codes(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return list(DEFAULT_STATE_CODES)
    normalized = {_normalize_state_code(value) for value in values if str(value).strip()}
    return sorted(normalized)


def _normalize_years(values: Iterable[int] | None) -> list[int]:
    if values is None:
        return list(DEFAULT_YEARS)
    years = sorted({int(value) for value in values})
    if not years:
        raise ValueError("At least one year is required.")
    return years


def _slugify_tokens(tokens: Iterable[str]) -> str:
    cleaned = [str(token).strip().replace(" ", "_").replace("-", "_") for token in tokens]
    cleaned = [token for token in cleaned if token]
    if not cleaned:
        raise ValueError("Cannot build a slug from empty tokens.")
    return "-".join(cleaned)


def build_raw_dataset_dir(
    raw_root: str | Path,
    *,
    crop_type: str,
    state_codes: Iterable[str] | None,
    years: Iterable[int] | None,
) -> Path:
    normalized_states = normalize_state_codes(state_codes)
    normalized_years = _normalize_years(years)
    slug = _slugify_tokens(
        [
            normalize_crop_type(crop_type or DEFAULT_CROP_TYPE) or DEFAULT_CROP_TYPE,
            "_".join(normalized_states),
            f"{normalized_years[0]}-{normalized_years[-1]}",
        ]
    )
    return Path(raw_root) / slug


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def normalize_county_id(values: pd.Series) -> pd.Series:
    coerced = values.astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    return coerced.str.zfill(5)


def normalize_crop_type(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized or None


def _build_cropnet_allow_patterns(*, crop_type: str, years: list[int], state_codes: list[str]) -> list[str]:
    crop = normalize_crop_type(crop_type) or crop_type
    crop_folder = {
        "corn": ("Corn", "Corn"),
        "cotton": ("Cotton", "Cotton"),
        "soybeans": ("Soybeans", "Soybean"),
        "soybean": ("Soybeans", "Soybean"),
        "winter wheat": ("WinterWheat", "WinterWheat"),
        "winterwheat": ("WinterWheat", "WinterWheat"),
        "winter_wheat": ("WinterWheat", "WinterWheat"),
    }[crop]
    crop_folder_name, crop_file_name = crop_folder

    patterns = [
        f"USDA Crop Dataset/{crop_folder_name}/{year}/USDA_{crop_file_name}_County_{year}.csv"
        for year in years
    ]
    for year in years:
        for state_code in state_codes:
            patterns.append(f"Sentinel-2 Imagery/data/AG/{year}/{state_code}/*.h5")
            patterns.append(f"Sentinel-2 Imagery/data/NDVI/{year}/{state_code}/*.h5")
            patterns.append(f"WRF-HRRR Computed Dataset/data/{year}/{state_code}/*.csv")
    return patterns


def _download_cropnet_snapshot(
    raw_dir: str | Path,
    *,
    repo_id: str,
    crop_type: str,
    years: list[int],
    state_codes: list[str],
    cache_dir: str | Path,
) -> Path:
    raw_dir = Path(raw_dir)
    cache_dir = Path(cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        cache_dir=str(cache_dir / "huggingface"),
        allow_patterns=_build_cropnet_allow_patterns(
            crop_type=crop_type,
            years=years,
            state_codes=state_codes,
        ),
        ignore_patterns=["**/.DS_Store"],
    )
    return raw_dir


def _selected_usda_paths(raw_dir: Path, *, crop_type: str, years: list[int]) -> list[Path]:
    all_paths = sorted(
        path.relative_to(raw_dir).as_posix()
        for path in raw_dir.rglob("USDA_*_County_*.csv")
        if path.is_file()
    )
    selected = select_usda_remote_files(all_paths, years=years, crops=[crop_type])
    return [raw_dir / Path(path) for path in selected]


def _load_selected_counties(
    raw_dir: Path,
    usda_paths: list[Path],
    *,
    crop_type: str,
) -> tuple[list[str], list[dict[str, object]]]:
    counties: set[str] = set()
    records: list[dict[str, object]] = []
    for path in usda_paths:
        remote_path = path.relative_to(raw_dir).as_posix()
        year, crop = parse_usda_remote_path(remote_path)
        frame = pd.read_csv(path)
        split = infer_usda_split(year)
        path_records = build_usda_records_from_frame(
            frame,
            crop_type=crop,
            year=year,
            split=split,
            target_kind="yield",
            source_path=remote_path,
        )
        for record in path_records:
            county_id = record.get("county_id")
            if county_id is None:
                continue
            counties.add(str(county_id).zfill(5))
        records.extend(path_records)
    if not counties:
        raise ValueError("No county IDs were found in the downloaded USDA files.")
    return sorted(counties), records


def _county_weather_frame(path: Path, selected_counties: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "FIPS Code" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["FIPS Code"] = frame["FIPS Code"].astype(str).str.zfill(5)
    if {"Year", "Month", "Day"} <= set(frame.columns):
        frame["date"] = pd.to_datetime(
            dict(
                year=pd.to_numeric(frame["Year"], errors="coerce"),
                month=pd.to_numeric(frame["Month"], errors="coerce"),
                day=pd.to_numeric(frame["Day"], errors="coerce").fillna(1),
            ),
            errors="coerce",
        )
    return frame[frame["FIPS Code"].isin(selected_counties)].copy()


def _month_from_date_key(date_key: object) -> int | None:
    text = str(date_key)
    if len(text) >= 7 and text[4] == "-":
        return int(text[5:7])
    if len(text) >= 2 and text[:2].isdigit():
        return int(text[:2])
    return None


def _extract_ag_ndvi_file(
    path: Path,
    *,
    year: int,
    selected_counties: set[str],
    crop_type: str,
    modality: str,
) -> tuple[list[pd.DataFrame], int]:
    frames: list[pd.DataFrame] = []
    processed_grids = 0
    if not path.is_file():
        return frames, processed_grids
    extractor = extract_ag_features if modality == "ag" else extract_ndvi_features
    try:
        with h5py.File(path, "r") as handle:
            for county_id in handle.keys():
                county_id = str(county_id).zfill(5)
                if county_id not in selected_counties:
                    continue
                county_group = handle[county_id]
                for date_key in county_group.keys():
                    node = county_group[date_key]
                    if "data" not in node:
                        continue
                    month = _month_from_date_key(date_key)
                    if month is None:
                        continue
                    grids = np.asarray(node["data"])
                    if grids.ndim == 2:
                        grids = grids[None, ...]
                    for grid in grids:
                        frames.append(
                            extractor(
                                grid,
                                county_id=county_id,
                                crop_type=crop_type,
                                year=year,
                                month=month,
                            )
                        )
                        processed_grids += 1
    except Exception as exc:
        warnings.warn(f"Skipping unreadable {modality.upper()} file {path}: {exc}", RuntimeWarning)
    return frames, processed_grids


def _extract_weather_file(
    path: Path,
    *,
    selected_counties: set[str],
    crop_type: str,
) -> tuple[list[pd.DataFrame], int]:
    if not path.is_file():
        return [], 1
    frame = _county_weather_frame(path, selected_counties)
    if frame.empty:
        return [], 1
    extracted_frames: list[pd.DataFrame] = []
    for county_id, county_frame in frame.groupby("FIPS Code", sort=True):
        extracted = extract_weather_features(
            county_frame,
            county_id=str(county_id).zfill(5),
            crop_type=crop_type,
        )
        if not extracted.empty:
            extracted_frames.append(extracted)
    return extracted_frames, 1


def _extract_ag_ndvi_frames(
    raw_dir: Path,
    *,
    modality: str,
    years: list[int],
    state_codes: list[str],
    selected_counties: set[str],
    crop_type: str,
    num_workers: int | None = None,
) -> pd.DataFrame:
    modality_dir = "AG" if modality == "ag" else "NDVI"
    candidate_paths: list[tuple[Path, int]] = []
    for year in years:
        for state_code in state_codes:
            candidate_paths.extend(
                (path, year)
                for path in sorted((raw_dir / f"Sentinel-2 Imagery/data/{modality_dir}/{year}/{state_code}").glob("*.h5"))
            )
    frames: list[pd.DataFrame] = []
    worker_count = max(1, int(num_workers or (os.cpu_count() or 1)))
    print(f"Working on {len(candidate_paths)} files with {worker_count} workers...")
    with tqdm(desc=f"{modality.upper()} images", unit="img") as progress_bar:
        if worker_count == 1 or len(candidate_paths) <= 1:
            for path, year in candidate_paths:
                extracted, processed = _extract_ag_ndvi_file(
                    path,
                    year=year,
                    selected_counties=selected_counties,
                    crop_type=crop_type,
                    modality=modality,
                )
                frames.extend(extracted)
                progress_bar.update(processed)
        else:
            with ProcessPoolExecutor(
                max_workers=min(len(candidate_paths), worker_count),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(
                        _extract_ag_ndvi_file,
                        path,
                        year=year,
                        selected_counties=selected_counties,
                        crop_type=crop_type,
                        modality=modality,
                    )
                    for path, year in candidate_paths
                ]
                for future in as_completed(futures):
                    extracted, processed = future.result()
                    frames.extend(extracted)
                    progress_bar.update(processed)
    if not frames:
        return pd.DataFrame()
    return aggregate_monthly_feature_frame(pd.concat(frames, ignore_index=True, sort=False))


def _extract_weather_frames(
    raw_dir: Path,
    *,
    years: list[int],
    state_codes: list[str],
    selected_counties: set[str],
    crop_type: str,
    num_workers: int | None = None,
) -> pd.DataFrame:
    candidate_paths: list[Path] = []
    for year in years:
        for state_code in state_codes:
            weather_dir = raw_dir / f"WRF-HRRR Computed Dataset/data/{year}/{state_code}"
            candidate_paths.extend(sorted(weather_dir.glob("*.csv")))
    frames: list[pd.DataFrame] = []
    worker_count = max(1, int(num_workers or (os.cpu_count() or 1)))
    with tqdm(total=len(candidate_paths), desc="WEATHER files", unit="file") as progress_bar:
        if worker_count == 1 or len(candidate_paths) <= 1:
            for path in candidate_paths:
                extracted, processed = _extract_weather_file(
                    path,
                    selected_counties=selected_counties,
                    crop_type=crop_type,
                )
                frames.extend(extracted)
                progress_bar.update(processed)
        else:
            with ProcessPoolExecutor(
                max_workers=min(len(candidate_paths), worker_count),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(
                        _extract_weather_file,
                        path,
                        selected_counties=selected_counties,
                        crop_type=crop_type,
                    )
                    for path in candidate_paths
                ]
                for future in as_completed(futures):
                    extracted, processed = future.result()
                    frames.extend(extracted)
                    progress_bar.update(processed)
    if not frames:
        return pd.DataFrame()
    return aggregate_monthly_feature_frame(pd.concat(frames, ignore_index=True, sort=False))


def build_monthly_table_from_raw_download(
    raw_dir: str | Path,
    *,
    crop_type: str,
    years: list[int],
    state_codes: list[str],
    num_workers: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw_dir = Path(raw_dir)
    usda_paths = _selected_usda_paths(raw_dir, crop_type=crop_type, years=years)
    if not usda_paths:
        raise FileNotFoundError("No USDA files were found in the downloaded CropNet snapshot.")

    counties, usda_records = _load_selected_counties(raw_dir, usda_paths, crop_type=crop_type)
    selected_counties = set(counties)

    ag_frame = _extract_ag_ndvi_frames(
        raw_dir,
        modality="ag",
        years=years,
        state_codes=state_codes,
        selected_counties=selected_counties,
        crop_type=normalize_crop_type(crop_type) or crop_type,
        num_workers=num_workers,
    )
    ndvi_frame = _extract_ag_ndvi_frames(
        raw_dir,
        modality="ndvi",
        years=years,
        state_codes=state_codes,
        selected_counties=selected_counties,
        crop_type=normalize_crop_type(crop_type) or crop_type,
        num_workers=num_workers,
    )
    weather_frame = _extract_weather_frames(
        raw_dir,
        years=years,
        state_codes=state_codes,
        selected_counties=selected_counties,
        crop_type=normalize_crop_type(crop_type) or crop_type,
        num_workers=num_workers,
    )

    monthly = combine_modality_feature_frames(ag_frame, ndvi_frame, weather_frame)
    if monthly.empty:
        raise ValueError("No monthly feature rows could be extracted from the downloaded CropNet data.")

    monthly = monthly.sort_values(META_COLS).reset_index(drop=True)
    monthly["county_id"] = normalize_county_id(monthly["county_id"])
    monthly["crop_type"] = monthly["crop_type"].astype(str).map(normalize_crop_type)
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce").astype(int)
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce").astype(int)
    before_year_filter = len(monthly)
    monthly = monthly[monthly["year"].isin(years)].copy()
    dropped_year_rows = before_year_filter - len(monthly)
    for feature in FEATURE_COLS:
        if feature not in monthly.columns:
            monthly[feature] = np.nan

    available_features = [col for col in FEATURE_COLS if monthly[col].notna().any()]
    if not available_features:
        raise ValueError("No canonical CropNet features were extracted from the raw snapshot.")

    monthly = monthly[META_COLS + FEATURE_COLS].copy()
    monthly = monthly.sort_values(META_COLS).reset_index(drop=True)

    diagnostic = {
        "raw_dir": raw_dir,
        "selected_counties": counties,
        "row_count": int(len(monthly)),
        "feature_columns": FEATURE_COLS,
        "available_features": available_features,
        "usda_source_count": int(len(usda_paths)),
        "usda_record_count": int(len(usda_records)),
        "years": years,
        "state_codes": state_codes,
        "crop_type": normalize_crop_type(crop_type) or crop_type,
        "dropped_rows_outside_requested_years": int(dropped_year_rows),
    }
    return monthly, available_features, diagnostic


def prepare_training_dataset_from_download(
    *,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    crop_type: str = DEFAULT_CROP_TYPE,
    state_codes: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    num_workers: int | None = None,
    overwrite: bool = False,
) -> PreparedDatasetPaths:
    normalized_states = normalize_state_codes(state_codes)
    normalized_years = _normalize_years(years)
    raw_dir = build_raw_dataset_dir(
        raw_root,
        crop_type=crop_type,
        state_codes=normalized_states,
        years=normalized_years,
    )
    _download_cropnet_snapshot(
        raw_dir,
        repo_id=repo_id,
        crop_type=crop_type,
        years=normalized_years,
        state_codes=normalized_states,
        cache_dir=cache_dir,
    )
    monthly, _, download_diagnostic = build_monthly_table_from_raw_download(
        raw_dir,
        crop_type=crop_type,
        years=normalized_years,
        state_codes=normalized_states,
        num_workers=num_workers,
    )
    source_path = raw_dir / "official_monthly_feature_table.parquet"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(source_path, index=False)

    prepared = prepare_training_dataset(
        source_path,
        output_dir,
        feature_group="all",
        crop_type=crop_type,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        overwrite=overwrite,
    )
    metadata = load_prepared_metadata(prepared.output_dir)
    metadata["raw_dir"] = raw_dir
    metadata["download_diagnostic"] = download_diagnostic
    save_json(metadata, prepared.metadata_path)
    return prepared


def validate_ground_truth_monthly_features(frame: pd.DataFrame, source_path: Path) -> None:
    generated_columns = sorted(GENERATED_FORECAST_COLUMNS.intersection(frame.columns))
    if generated_columns:
        raise ValueError(
            "Monthly feature table appears to contain generated forecast columns "
            f"{generated_columns}. Use a ground-truth extraction table instead."
        )
    lowered_path = str(source_path).lower()
    if "blank_fill" in lowered_path or "forecast" in lowered_path:
        raise ValueError(
            "Monthly feature table path looks like a forecast artifact. "
            "Use the ground-truth official monthly feature table."
        )


def resolve_year_splits(years: list[int]) -> tuple[list[int], list[int], list[int]]:
    unique_years = sorted({int(year) for year in years})
    if len(unique_years) < 3:
        raise ValueError("At least three distinct years are required to build train/val/test splits.")
    train_years = unique_years[:-2]
    val_years = unique_years[-2:-1]
    test_years = unique_years[-1:]
    return train_years, val_years, test_years


def save_scaler(frame: pd.DataFrame, feature_cols: list[str], path: Path) -> pd.DataFrame:
    means = frame[feature_cols].mean(numeric_only=True).fillna(0.0).to_numpy(dtype=float)
    stds = (
        frame[feature_cols]
        .std(numeric_only=True)
        .replace(0, 1.0)
        .fillna(1.0)
        .to_numpy(dtype=float)
    )
    stats = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean": means,
            "std": stds,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)
    return stats


def load_scaler(path: str | Path) -> tuple[list[str], pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    required = {"feature", "mean", "std"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Scaler csv must contain columns {sorted(required)}")
    feature_cols = frame["feature"].astype(str).tolist()
    mu = pd.Series(frame["mean"].astype(float).to_numpy(), index=feature_cols)
    sigma = pd.Series(frame["std"].astype(float).replace(0, 1.0).fillna(1.0).to_numpy(), index=feature_cols)
    return feature_cols, mu, sigma


def _coerce_monthly_frame(frame: pd.DataFrame, feature_group: str, crop_type: str | None) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = selected_feature_columns(feature_group)
    required_cols = list(dict.fromkeys(META_COLS + feature_cols))
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Monthly feature table is missing required columns: {missing}")

    out = frame[required_cols].copy()
    out["county_id"] = normalize_county_id(out["county_id"])
    out["crop_type"] = out["crop_type"].astype(str).map(normalize_crop_type)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype(int)
    if crop_type is not None:
        normalized_crop = normalize_crop_type(crop_type)
        out = out[out["crop_type"].eq(normalized_crop)].copy()
    out = out.sort_values(META_COLS).reset_index(drop=True)
    return out, feature_cols


def _assign_split(year: int, train_years: set[int], val_years: set[int], test_years: set[int]) -> str:
    if year in train_years:
        return "train"
    if year in val_years:
        return "val"
    if year in test_years:
        return "test"
    raise ValueError(f"Year {year} is not assigned to train, val, or test.")


def prepare_training_dataset(
    source_path: str | Path = DEFAULT_SOURCE_TABLE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    feature_group: str = "all",
    crop_type: str | None = "corn",
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
    overwrite: bool = False,
) -> PreparedDatasetPaths:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Monthly feature table not found: {source_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Training dataset directory already exists and is not empty: {output_dir}. "
            "Re-run with overwrite=True or choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_table(source_path)
    validate_ground_truth_monthly_features(raw, source_path)
    frame, feature_cols = _coerce_monthly_frame(raw, feature_group, crop_type)
    if frame.empty:
        raise ValueError("No rows remain after applying the crop filter.")

    years = sorted(frame["year"].dropna().astype(int).unique().tolist())
    if train_years is None or val_years is None or test_years is None:
        inferred_train, inferred_val, inferred_test = resolve_year_splits(years)
        train_years = train_years or inferred_train
        val_years = val_years or inferred_val
        test_years = test_years or inferred_test

    train_set, val_set, test_set = set(train_years), set(val_years), set(test_years)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train_years, val_years, and test_years must be disjoint.")

    assigned_years = train_set | val_set | test_set
    missing_years = [year for year in years if year not in assigned_years]
    if missing_years:
        raise ValueError(
            "The split definition does not cover every year present in the source data. "
            f"Missing years: {missing_years}"
        )

    out = frame.copy()
    out["split"] = [
        _assign_split(int(year), train_set, val_set, test_set)
        for year in out["year"].to_numpy()
    ]

    all_path = output_dir / "all.parquet"
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    scaler_path = output_dir / "scaler.csv"
    metadata_path = output_dir / "metadata.json"

    out.to_parquet(all_path, index=False)
    out[out["split"].eq("train")].to_parquet(train_path, index=False)
    out[out["split"].eq("val")].to_parquet(val_path, index=False)
    out[out["split"].eq("test")].to_parquet(test_path, index=False)
    save_scaler(out[out["split"].eq("train")], feature_cols, scaler_path)

    metadata = {
        "source_path": source_path,
        "output_dir": output_dir,
        "feature_group": feature_group,
        "crop_type": crop_type or "mixed",
        "feature_columns": feature_cols,
        "years": years,
        "train_years": sorted(train_set),
        "val_years": sorted(val_set),
        "test_years": sorted(test_set),
        "row_count": int(len(out)),
        "split_counts": out["split"].value_counts().to_dict(),
        "paths": {
            "all": all_path,
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "scaler": scaler_path,
        },
    }
    save_json(metadata, metadata_path)

    return PreparedDatasetPaths(
        output_dir=output_dir,
        all_path=all_path,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
    )


def load_prepared_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prepared dataset metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_prepared_dataset(dataset_dir: str | Path) -> dict[str, pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    paths = {
        "all": dataset_dir / "all.parquet",
        "train": dataset_dir / "train.parquet",
        "val": dataset_dir / "val.parquet",
        "test": dataset_dir / "test.parquet",
    }
    frames = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Prepared dataset file not found: {path}")
        frames[name] = pd.read_parquet(path)
    return frames
