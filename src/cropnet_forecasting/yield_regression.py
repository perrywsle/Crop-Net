from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional dependency.
    from lightgbm import LGBMRegressor

    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_LIGHTGBM = False

try:  # Optional dependency.
    from xgboost import XGBRegressor

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_XGBOOST = False

if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from cropnet_forecasting.data import read_table
    from cropnet_forecasting.features import FEATURE_GROUP_SELECTIONS, META_COLS, selected_feature_columns
else:  # pragma: no cover - exercised when run as a module
    from .data import read_table
    from .features import FEATURE_GROUP_SELECTIONS, META_COLS, selected_feature_columns

GROWING_SEASON_MONTHS = (4, 5, 6, 7, 8, 9)
TARGET_GRAINS = ("monthly", "annual")
DEFAULT_MIN_OVERLAP_ROWS = 12
DEFAULT_OUTPUT_SUBDIR = "yield_baseline"
DEFAULT_MAX_MISSING_FRACTION = 0.20
DEFAULT_CORRELATION_THRESHOLD = 0.995
FEATURE_GROUP_BENCHMARKS = ("ag", "ndvi", "weather", "ag_ndvi", "ndvi_weather", "all")
TIMING_FEATURES = ("month", "month_sin", "month_cos")
TARGET_COLUMNS = {"yield_bu_acre", "target_unit"}
ANNUAL_FEATURE_SUFFIXES = ("_slope", "_delta", "_amplitude")
GENERATED_FORECAST_COLUMNS = {
    "forecast_step",
    "known_months",
    "source_note",
    "y_pred",
}
YIELD_COLUMN_CANDIDATES = (
    "YIELD, MEASURED IN BU / ACRE",
    "yield_bu_acre",
    "yield",
    "target_value",
)
_YIELD_UNIT_BY_CROP: dict[str, str] = {
    "corn": "BU / ACRE",
    "soybean": "BU / ACRE",
    "soybeans": "BU / ACRE",
    "winter wheat": "BU / ACRE",
    "cotton": "LB / ACRE",
}
_STATE_ABBR_BY_FIPS_PREFIX: dict[str, str] = {
    "19": "ia",
}


@dataclass(frozen=True)
class RegressionArtifacts:
    results: pd.DataFrame
    split_mode: str
    train_rows: int
    test_rows: int
    overlap_rows: int
    overlap_counties: int
    overlap_years: int
    best_model_name: str
    fitted_model: Pipeline
    output_dir: Path
    results_path: Path
    training_frame_path: Path
    model_path: Path
    metadata_path: Path
    importance_path: Path
    summary_path: Path
    feature_group_benchmark_path: Path
    year_cv_benchmark_path: Path
    month_benchmark_path: Path
    window_benchmark_path: Path
    pruning_report_path: Path
    residuals_path: Path


def _json_default(value: Any) -> Any:
    """Serialize common numpy/pandas/path values into JSON primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def normalize_county_id(values: pd.Series) -> pd.Series:
    """Convert county identifiers to 5-digit FIPS strings."""
    coerced = values.astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    return coerced.str.zfill(5)


def normalize_crop_type(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in {"winterwheat", "winter wheat"}:
        return "winter wheat"
    return normalized or None


def infer_crop_type_from_path(path: Path) -> str | None:
    match = re.search(r"USDA_([^_]+(?:_[^_]+)*)_County_\d{4}\.csv$", path.name)
    if not match:
        return None
    return normalize_crop_type(match.group(1))


def detect_yield_column(columns: list[str]) -> str:
    for candidate in YIELD_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    lower_lookup = {col.lower(): col for col in columns}
    for col in columns:
        lowered = col.lower()
        if "yield" in lowered and "acre" in lowered:
            return col
    if "target_value" in lower_lookup:
        return lower_lookup["target_value"]
    raise ValueError(
        "Could not find a USDA yield column. Looked for one of: "
        + ", ".join(YIELD_COLUMN_CANDIDATES)
    )


def infer_target_unit(yield_column: str, crop_type: str | None) -> str:
    """Infer the unit used by the selected USDA yield column."""
    lowered = yield_column.lower()
    if "bu / acre" in lowered or "bu/acre" in lowered:
        return "BU / ACRE"
    if "lb / acre" in lowered or "lb/acre" in lowered:
        return "LB / ACRE"
    if crop_type is not None:
        return _YIELD_UNIT_BY_CROP.get(normalize_crop_type(crop_type) or "", "unknown")
    return "unknown"


def validate_ground_truth_monthly_features(frame: pd.DataFrame, source_path: Path) -> None:
    """Reject forecast or blank-fill outputs masquerading as monthly features."""
    generated_columns = sorted(GENERATED_FORECAST_COLUMNS.intersection(frame.columns))
    if generated_columns:
        raise ValueError(
            "Monthly feature table appears to contain generated forecast columns "
            f"{generated_columns}. Use the ground-truth official_monthly_feature_table "
            "from extraction-only preprocessing instead."
        )
    lowered_path = str(source_path).lower()
    if "blank_fill" in lowered_path or "forecast" in lowered_path:
        raise ValueError(
            "Monthly feature table path looks like a forecast/blank-fill artifact. "
            "Use the extraction artifact official_monthly_feature_table.parquet."
        )


def read_monthly_features(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    validate_ground_truth_monthly_features(frame, path)
    required = {"county_id", "year", "month"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly feature table is missing required columns: {missing}")

    out = frame.copy()
    out["county_id"] = normalize_county_id(out["county_id"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    if "crop_type" in out.columns:
        out["crop_type"] = out["crop_type"].astype(str).map(normalize_crop_type)
    return out


def aggregate_growing_season_features(
    monthly_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Aggregate monthly features into county-year seasonal summaries.

    This keeps the original growing-season mean, but also adds simple growth
    descriptors so the model can see whether a feature is rising or falling
    across April-September instead of only seeing an average.
    """
    if monthly_df.empty:
        return pd.DataFrame(columns=META_COLS + feature_columns)

    frame = monthly_df.copy()
    if "month" not in frame.columns:
        raise ValueError("Monthly feature table must include a 'month' column.")

    frame = frame[frame["month"].isin(GROWING_SEASON_MONTHS)].copy()
    if frame.empty:
        raise ValueError("No April-September records were found in the monthly feature table.")

    present_features = [col for col in feature_columns if col in frame.columns]
    if not present_features:
        raise ValueError("None of the expected CropNet feature columns were present.")

    group_cols = ["county_id", "year"]
    if "crop_type" in frame.columns:
        group_cols.append("crop_type")

    for col in present_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, Any] = dict(zip(group_cols, keys, strict=False))
        group = group.sort_values("month").copy()
        months = pd.to_numeric(group["month"], errors="coerce").to_numpy(dtype=float)

        for col in present_features:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            valid_mask = np.isfinite(values) & np.isfinite(months)
            valid_values = values[valid_mask]
            valid_months = months[valid_mask]

            if valid_values.size == 0:
                row[col] = np.nan
                row[f"{col}_slope"] = np.nan
                row[f"{col}_delta"] = np.nan
                row[f"{col}_amplitude"] = np.nan
                continue

            row[col] = float(np.nanmean(valid_values))
            row[f"{col}_delta"] = float(valid_values[-1] - valid_values[0])
            row[f"{col}_amplitude"] = float(np.nanmax(valid_values) - np.nanmin(valid_values))

            if valid_values.size >= 2 and np.unique(valid_months).size >= 2:
                slope = np.polyfit(valid_months, valid_values, deg=1)[0]
                row[f"{col}_slope"] = float(slope)
            else:
                row[f"{col}_slope"] = np.nan

        records.append(row)

    annual = pd.DataFrame(records)
    annual["county_id"] = normalize_county_id(annual["county_id"])
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
    if "crop_type" in annual.columns:
        annual["crop_type"] = annual["crop_type"].map(normalize_crop_type)
    annual = annual.dropna(subset=["county_id", "year"]).reset_index(drop=True)
    annual["year"] = annual["year"].astype(int)
    return annual


def load_usda_yield_table(path: Path, crop_type: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"state_ansi", "county_ansi", "year"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"USDA file is missing required columns: {missing}")

    yield_col = detect_yield_column(list(frame.columns))
    out = frame.copy()
    out["state_ansi"] = pd.to_numeric(out["state_ansi"], errors="coerce").astype("Int64")
    out["county_ansi"] = pd.to_numeric(out["county_ansi"], errors="coerce").astype("Int64")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["county_id"] = (
        out["state_ansi"].astype(str).str.zfill(2) + out["county_ansi"].astype(str).str.zfill(3)
    )
    out["yield_bu_acre"] = pd.to_numeric(out[yield_col], errors="coerce")
    out["crop_type"] = normalize_crop_type(crop_type or infer_crop_type_from_path(path))
    out["target_unit"] = infer_target_unit(yield_col, out["crop_type"].dropna().iloc[0] if out["crop_type"].notna().any() else crop_type)
    keep_cols = ["county_id", "year", "yield_bu_acre", "target_unit"]
    if out["crop_type"].notna().any():
        keep_cols.append("crop_type")
    out = out[keep_cols].dropna(subset=["county_id", "year", "yield_bu_acre"]).reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    return out


def load_usda_yield_tables(paths: list[Path], crop_type: str | None = None) -> pd.DataFrame:
    tables = [load_usda_yield_table(path, crop_type=crop_type) for path in paths]
    if not tables:
        raise ValueError("No USDA yield files were provided.")
    combined = pd.concat(tables, ignore_index=True)
    combined["county_id"] = normalize_county_id(combined["county_id"])
    return combined.drop_duplicates(subset=[col for col in ["county_id", "year", "crop_type"] if col in combined.columns])


def discover_monthly_candidates(root: Path) -> list[Path]:
    candidates = sorted(
        {
            *root.rglob("official_monthly_feature_table.parquet"),
            *root.rglob("official_monthly_feature_table.csv"),
            *root.rglob("sample_monthly_features.csv"),
        }
    )
    return [path for path in candidates if path.is_file()]


def discover_usda_candidates(root: Path) -> list[Path]:
    candidates = sorted(root.rglob("USDA_*_County_*.csv"))
    return [path for path in candidates if path.is_file()]


def select_best_usda_paths(
    candidates: list[Path],
    monthly_frame: pd.DataFrame | None,
    crop_type: str | None,
) -> list[Path]:
    if not candidates:
        return []
    if monthly_frame is None or monthly_frame.empty:
        if crop_type is None:
            return candidates
        filtered = [path for path in candidates if normalize_crop_type(infer_crop_type_from_path(path)) == crop_type]
        return filtered or candidates

    monthly_counties = set(monthly_frame["county_id"].dropna().astype(str))
    monthly_years = set(pd.to_numeric(monthly_frame["year"], errors="coerce").dropna().astype(int))
    monthly_crop_types = set()
    if "crop_type" in monthly_frame.columns:
        monthly_crop_types = {
            normalize_crop_type(value)
            for value in monthly_frame["crop_type"].dropna().astype(str).tolist()
        }
    monthly_crop_types.discard(None)

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        path_crop = normalize_crop_type(infer_crop_type_from_path(path))
        score = 0
        if crop_type is not None and path_crop == crop_type:
            score += 1000
        if monthly_crop_types and path_crop in monthly_crop_types:
            score += 500
        match = re.search(r"_(\d{4})\.csv$", path.name)
        if match and int(match.group(1)) in monthly_years:
            score += 50
        try:
            sample = load_usda_yield_table(path, crop_type=path_crop)
        except Exception:
            sample = pd.DataFrame()
        if not sample.empty:
            county_overlap = len(set(sample["county_id"]).intersection(monthly_counties))
            year_overlap = len(set(sample["year"]).intersection(monthly_years))
            score += county_overlap * 10 + year_overlap
        scored.append((score, path))

    max_score = max(score for score, _ in scored)
    if max_score <= 0:
        return candidates
    chosen = [path for score, path in scored if score == max_score]
    return chosen


def build_training_frame(
    monthly_frame: pd.DataFrame,
    usda_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    annual = aggregate_growing_season_features(monthly_frame, feature_columns)
    if annual.empty:
        raise ValueError("No annual feature rows could be created from the monthly table.")

    merge_keys = [key for key in ["county_id", "year", "crop_type"] if key in annual.columns and key in usda_frame.columns]
    if "crop_type" in merge_keys:
        annual["crop_type"] = annual["crop_type"].map(normalize_crop_type)
        usda_frame = usda_frame.copy()
        usda_frame["crop_type"] = usda_frame["crop_type"].map(normalize_crop_type)

    merged = annual.merge(usda_frame, on=merge_keys, how="inner", suffixes=("", "_usda"))
    if merged.empty:
        return merged, merge_keys, (
            "No county-year overlap was found between the server monthly features and USDA yield labels."
        )

    feature_cols = [
        col
        for col in merged.columns
        if col not in set(META_COLS).union(TARGET_COLUMNS)
        and pd.api.types.is_numeric_dtype(merged[col])
    ]
    if not feature_cols:
        raise ValueError("The merged dataset does not contain any model feature columns.")

    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.reset_index(drop=True), feature_cols, ""


def add_month_timing_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype(int)
    radians = 2.0 * np.pi * out["month"].to_numpy(dtype=float) / 12.0
    out["month_sin"] = np.sin(radians)
    out["month_cos"] = np.cos(radians)
    return out


def build_monthly_training_frame(
    monthly_frame: pd.DataFrame,
    usda_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    frame = monthly_frame.copy()
    present_features = [col for col in feature_columns if col in frame.columns]
    if not present_features:
        raise ValueError("None of the expected CropNet monthly feature columns were present.")

    for col in present_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = add_month_timing_features(frame)

    merge_keys = [key for key in ["county_id", "year", "crop_type"] if key in frame.columns and key in usda_frame.columns]
    if "crop_type" in merge_keys:
        frame["crop_type"] = frame["crop_type"].map(normalize_crop_type)
        usda_frame = usda_frame.copy()
        usda_frame["crop_type"] = usda_frame["crop_type"].map(normalize_crop_type)

    merged = frame.merge(usda_frame, on=merge_keys, how="inner", suffixes=("", "_usda"))
    if merged.empty:
        return merged, merge_keys, (
            "No county-year-month overlap was found between monthly features and USDA yield labels."
        )

    feature_cols = [
        col
        for col in present_features + list(TIMING_FEATURES)
        if col in merged.columns and pd.api.types.is_numeric_dtype(merged[col])
    ]
    if not feature_cols:
        raise ValueError("The merged monthly dataset does not contain any model feature columns.")

    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged.reset_index(drop=True), feature_cols, ""


def base_feature_name(feature_name: str) -> str:
    for suffix in ANNUAL_FEATURE_SUFFIXES:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    return feature_name


def annualized_columns_for_group(feature_cols: list[str], feature_group: str) -> list[str]:
    base_features = set(selected_feature_columns(feature_group))
    return [col for col in feature_cols if base_feature_name(col) in base_features]


def model_columns_for_group(feature_cols: list[str], feature_group: str) -> list[str]:
    base_features = set(selected_feature_columns(feature_group))
    selected = [col for col in feature_cols if base_feature_name(col) in base_features]
    selected.extend([col for col in TIMING_FEATURES if col in feature_cols])
    return list(dict.fromkeys(selected))


def prune_feature_columns(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> tuple[list[str], dict[str, Any]]:
    """Drop deterministic low-information columns before fitting yield models."""
    if not feature_cols:
        raise ValueError("No feature columns were provided for pruning.")

    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    missing_fraction = numeric.isna().mean()
    sparse = [
        col
        for col in feature_cols
        if float(missing_fraction.get(col, 1.0)) > max_missing_fraction
    ]
    candidates = [col for col in feature_cols if col not in sparse]
    zero_variance = [
        col
        for col in candidates
        if numeric[col].dropna().nunique() <= 1
    ]
    candidates = [col for col in candidates if col not in zero_variance]

    correlated: list[dict[str, Any]] = []
    correlated_drop: set[str] = set()
    if len(candidates) >= 2:
        corr = numeric[candidates].corr().abs()
        for idx, left in enumerate(candidates):
            if left in correlated_drop:
                continue
            for right in candidates[idx + 1 :]:
                if right in correlated_drop:
                    continue
                value = corr.loc[left, right]
                if pd.notna(value) and float(value) >= correlation_threshold:
                    correlated_drop.add(right)
                    correlated.append(
                        {
                            "dropped": right,
                            "kept": left,
                            "correlation": float(value),
                        }
                    )

    kept = [col for col in candidates if col not in correlated_drop]
    if not kept:
        raise ValueError("Feature pruning removed all candidate columns.")

    report = {
        "input_feature_count": len(feature_cols),
        "kept_feature_count": len(kept),
        "max_missing_fraction": max_missing_fraction,
        "correlation_threshold": correlation_threshold,
        "dropped_sparse": sparse,
        "dropped_zero_variance": zero_variance,
        "dropped_correlated": correlated,
        "kept_features": kept,
    }
    return kept, report


def overlap_summary(monthly_frame: pd.DataFrame, usda_frame: pd.DataFrame) -> dict[str, Any]:
    monthly_pairs_frame = monthly_frame[["county_id", "year"]].dropna().copy()
    monthly_pairs_frame["county_id"] = monthly_pairs_frame["county_id"].astype(str)
    monthly_pairs_frame["year"] = pd.to_numeric(monthly_pairs_frame["year"], errors="coerce").astype(int)
    monthly_pairs = set(map(tuple, monthly_pairs_frame.to_numpy()))

    usda_pairs_frame = usda_frame[["county_id", "year"]].dropna().copy()
    usda_pairs_frame["county_id"] = usda_pairs_frame["county_id"].astype(str)
    usda_pairs_frame["year"] = pd.to_numeric(usda_pairs_frame["year"], errors="coerce").astype(int)
    usda_pairs = set(map(tuple, usda_pairs_frame.to_numpy()))

    counties = set(monthly_frame["county_id"].astype(str)).intersection(set(usda_frame["county_id"].astype(str)))
    years = set(pd.to_numeric(monthly_frame["year"], errors="coerce").dropna().astype(int)).intersection(set(usda_frame["year"].astype(int)))
    return {
        "county_year_overlap": len(monthly_pairs.intersection(usda_pairs)),
        "county_overlap": len(counties),
        "year_overlap": len(years),
    }


def suggest_fips_codes(
    usda_frame: pd.DataFrame,
    monthly_frame: pd.DataFrame,
    limit: int = 20,
) -> list[str]:
    monthly_counties = set(monthly_frame["county_id"].astype(str))
    ranked = (
        usda_frame["county_id"]
        .astype(str)
        .value_counts()
        .rename_axis("county_id")
        .reset_index(name="count")
    )
    suggestion = [cid for cid in ranked["county_id"].tolist() if cid not in monthly_counties]
    if not suggestion:
        suggestion = ranked["county_id"].tolist()
    return suggestion[:limit]


def build_model_pipelines(
    random_state: int = 42,
    *,
    include_optional_models: bool = False,
) -> dict[str, Pipeline]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    models: dict[str, Pipeline] = {
        "Ridge": Pipeline(base_steps + [("model", Ridge(alpha=5.0))]),
        "RandomForest": Pipeline(
            base_steps
            + [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "ExtraTrees": Pipeline(
            base_steps
            + [
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }
    if include_optional_models and _HAS_XGBOOST:
        models["XGBoost"] = Pipeline(
            base_steps
            + [
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                        reg_lambda=1.0,
                        verbosity=0,
                    ),
                )
            ]
        )
    if include_optional_models and _HAS_LIGHTGBM:
        models["LightGBM"] = Pipeline(
            base_steps
            + [
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                )
            ]
        )
    return models


def build_model_candidate_pipelines(random_state: int = 42) -> dict[str, list[Pipeline]]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "Ridge": [
            Pipeline(base_steps + [("model", Ridge(alpha=alpha))])
            for alpha in (0.1, 1.0, 5.0, 20.0)
        ],
        "RandomForest": [
            Pipeline(
                base_steps
                + [
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    )
                ]
            )
            for max_depth, min_samples_leaf in [(None, 1), (8, 2), (12, 3)]
        ],
        "ExtraTrees": [
            Pipeline(
                base_steps
                + [
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=300,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    )
                ]
            )
            for max_depth, min_samples_leaf in [(None, 1), (8, 2), (12, 3)]
        ],
    }


def choose_validation_split(
    train_df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if len(train_df) < 6:
        return None
    years = sorted(pd.to_numeric(train_df["year"], errors="coerce").dropna().astype(int).unique())
    if len(years) >= 2:
        validation_year = years[-1]
        tune_train = train_df[train_df["year"].astype(int) < validation_year].copy()
        tune_valid = train_df[train_df["year"].astype(int) == validation_year].copy()
        if not tune_train.empty and not tune_valid.empty:
            return tune_train, tune_valid
    tune_train, tune_valid = train_test_split(
        train_df,
        test_size=0.25,
        random_state=random_state,
    )
    if tune_train.empty or tune_valid.empty:
        return None
    return tune_train.copy(), tune_valid.copy()


def select_model_pipelines(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    random_state: int = 42,
    include_optional_models: bool = False,
    tune_hyperparameters: bool = True,
) -> dict[str, Pipeline]:
    if not tune_hyperparameters:
        return build_model_pipelines(
            random_state=random_state,
            include_optional_models=include_optional_models,
        )

    split = choose_validation_split(train_df, random_state=random_state)
    if split is None:
        return build_model_pipelines(
            random_state=random_state,
            include_optional_models=include_optional_models,
        )
    tune_train, tune_valid = split
    selected: dict[str, Pipeline] = {}
    for family, candidates in build_model_candidate_pipelines(random_state).items():
        best_score = float("inf")
        best_pipeline = candidates[0]
        for candidate in candidates:
            candidate.fit(tune_train[feature_cols], tune_train[target_col].to_numpy(dtype=float))
            predictions = candidate.predict(tune_valid[feature_cols])
            score = float(
                np.sqrt(
                    mean_squared_error(
                        tune_valid[target_col].to_numpy(dtype=float),
                        predictions,
                    )
                )
            )
            if score < best_score:
                best_score = score
                best_pipeline = candidate
        selected[family] = best_pipeline
    if include_optional_models:
        optional = build_model_pipelines(
            random_state=random_state,
            include_optional_models=True,
        )
        for name, pipeline in optional.items():
            selected.setdefault(name, pipeline)
    return selected


def split_dataset(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any]]:
    unique_years = sorted(pd.unique(df["year"].astype(int)).tolist())
    unique_counties = sorted(pd.unique(df["county_id"].astype(str)).tolist())

    if len(unique_years) >= 2:
        test_year = unique_years[-1]
        train = df[df["year"].astype(int) < test_year].copy()
        test = df[df["year"].astype(int) == test_year].copy()
        if not train.empty and not test.empty:
            return train, test, f"year_split (test_year={test_year})", {
                "strategy": "year_split",
                "test_year": test_year,
            }

    if len(unique_counties) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df["county_id"].astype(str)))
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        if not train.empty and not test.empty:
            return train, test, "county_split", {
                "strategy": "county_split",
                "test_year": None,
            }

    raise ValueError(
        "Not enough independent years or counties to create a leakage-safe train/test split. "
        f"Found {len(unique_years)} year(s), {len(unique_counties)} county/counties, and {len(df)} row(s)."
    )


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str,
    model_type: str,
    n_train: int,
    n_test: int,
    feature_group: str = "all",
) -> dict[str, Any]:
    return {
        "model": model,
        "model_type": model_type,
        "feature_group": feature_group,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "mape": compute_mape(y_true, y_pred),
        "n_train": n_train,
        "n_test": n_test,
    }


def evaluate_naive_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    feature_group: str = "all",
) -> pd.DataFrame:
    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)
    train_mean = float(np.nanmean(y_train))
    rows = [
        regression_metric_row(
            y_test,
            np.full(shape=len(test_df), fill_value=train_mean, dtype=float),
            model="BaselineTrainMean",
            model_type="baseline",
            feature_group=feature_group,
            n_train=len(train_df),
            n_test=len(test_df),
        )
    ]

    lookup = {
        (str(row.county_id), int(row.year)): float(getattr(row, target_col))
        for row in train_df[["county_id", "year", target_col]].itertuples(index=False)
    }
    previous_year_predictions = [
        lookup.get((str(row.county_id), int(row.year) - 1), train_mean)
        for row in test_df[["county_id", "year"]].itertuples(index=False)
    ]
    rows.append(
        regression_metric_row(
            y_test,
            np.asarray(previous_year_predictions, dtype=float),
            model="BaselinePreviousYearSameCounty",
            model_type="baseline",
            feature_group=feature_group,
            n_train=len(train_df),
            n_test=len(test_df),
        )
    )
    return pd.DataFrame(rows)


def baseline_prediction_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    train_mean = float(np.nanmean(train_df[target_col].to_numpy(dtype=float)))
    lookup = {
        (str(row.county_id), int(row.year)): float(getattr(row, target_col))
        for row in train_df[["county_id", "year", target_col]].drop_duplicates().itertuples(index=False)
    }
    out = pd.DataFrame(index=test_df.index)
    out["BaselineTrainMean"] = train_mean
    out["BaselinePreviousYearSameCounty"] = [
        lookup.get((str(row.county_id), int(row.year) - 1), train_mean)
        for row in test_df[["county_id", "year"]].itertuples(index=False)
    ]
    return out


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    include_optional_models: bool = False,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    feature_group: str = "all",
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].to_numpy(dtype=float)

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    for name, pipe in select_model_pipelines(
        train_df,
        feature_cols,
        target_col,
        random_state=random_state,
        include_optional_models=include_optional_models,
        tune_hyperparameters=tune_hyperparameters,
    ).items():
        try:
            pipe.fit(X_train, y_train)
            predictions = pipe.predict(X_test)
            results.append(
                regression_metric_row(
                    y_test,
                    predictions,
                    model=name,
                    model_type="ml",
                    feature_group=feature_group,
                    n_train=len(train_df),
                    n_test=len(test_df),
                )
            )
            fitted_models[name] = pipe
        except Exception as exc:  # pragma: no cover - model-specific failures
            print(f"Model {name} failed: {exc}")

    if not results:
        raise RuntimeError("All benchmark models failed to train.")

    result_frame = pd.DataFrame(results).sort_values(["rmse", "mae", "model"]).reset_index(drop=True)
    return result_frame, fitted_models


def evaluate_feature_group_benchmarks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    random_state: int = 42,
    include_optional_models: bool = False,
    tune_hyperparameters: bool = True,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group_name in FEATURE_GROUP_BENCHMARKS:
        group_cols = model_columns_for_group(feature_cols, group_name)
        if not group_cols:
            continue
        group_results, _ = evaluate_models(
            train_df,
            test_df,
            group_cols,
            target_col,
            include_optional_models=include_optional_models,
            random_state=random_state,
            tune_hyperparameters=tune_hyperparameters,
            feature_group=group_name,
        )
        rows.append(group_results)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["feature_group", "rmse", "mae", "model"]
    ).reset_index(drop=True)


def evaluate_year_cv(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    random_state: int = 42,
    include_optional_models: bool = False,
    tune_hyperparameters: bool = True,
    feature_group: str = "all",
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    years = sorted(pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int).unique())
    for test_year in years:
        train_df = frame[frame["year"].astype(int) != test_year].copy()
        test_df = frame[frame["year"].astype(int) == test_year].copy()
        if train_df.empty or test_df.empty:
            continue
        model_results, _ = evaluate_models(
            train_df,
            test_df,
            feature_cols,
            target_col,
            include_optional_models=include_optional_models,
            random_state=random_state,
            tune_hyperparameters=tune_hyperparameters,
            feature_group=feature_group,
        )
        baseline_results = evaluate_naive_baselines(
            train_df,
            test_df,
            target_col,
            feature_group=feature_group,
        )
        fold = pd.concat([model_results, baseline_results], ignore_index=True)
        fold.insert(0, "fold_test_year", test_year)
        rows.append(fold)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["fold_test_year", "model_type", "rmse", "mae", "model"]
    ).reset_index(drop=True)


def build_residual_frame(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    predictions = model.predict(test_df[feature_cols])
    base_cols = ["county_id", "year", target_col]
    if "month" in test_df.columns:
        base_cols.insert(2, "month")
    residuals = test_df[base_cols].copy()
    if "crop_type" in test_df.columns:
        residuals.insert(1, "crop_type", test_df["crop_type"].to_numpy())
    residuals["prediction"] = predictions
    residuals["residual"] = residuals[target_col] - residuals["prediction"]
    residuals["abs_error"] = residuals["residual"].abs()
    residuals["ape"] = np.where(
        residuals[target_col].ne(0),
        residuals["abs_error"] / residuals[target_col].abs() * 100.0,
        np.nan,
    )
    sort_cols = [col for col in ["year", "county_id", "month"] if col in residuals.columns]
    return residuals.sort_values(sort_cols).reset_index(drop=True)


def build_prediction_frame(
    model: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_model_name: str,
) -> pd.DataFrame:
    base_cols = ["county_id", "year", "month", target_col]
    if "crop_type" in test_df.columns:
        base_cols.insert(1, "crop_type")
    predictions = test_df[base_cols].copy()
    predictions[best_model_name] = model.predict(test_df[feature_cols])
    baselines = baseline_prediction_columns(train_df, test_df, target_col)
    for col in baselines.columns:
        predictions[col] = baselines[col].to_numpy(dtype=float)
    return predictions


def benchmark_prediction_groups(
    prediction_frame: pd.DataFrame,
    *,
    group_column: str,
    target_col: str,
    model_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in prediction_frame.groupby(group_column, sort=True):
        y_true = group[target_col].to_numpy(dtype=float)
        for model_name in model_columns:
            rows.append(
                regression_metric_row(
                    y_true,
                    group[model_name].to_numpy(dtype=float),
                    model=model_name,
                    model_type="baseline" if model_name.startswith("Baseline") else "ml",
                    feature_group="all",
                    n_train=0,
                    n_test=len(group),
                )
                | {group_column: group_value}
            )
    return pd.DataFrame(rows)


def build_month_benchmark(
    prediction_frame: pd.DataFrame,
    *,
    target_col: str,
    model_columns: list[str],
) -> pd.DataFrame:
    if "month" not in prediction_frame.columns:
        return pd.DataFrame()
    return benchmark_prediction_groups(
        prediction_frame,
        group_column="month",
        target_col=target_col,
        model_columns=model_columns,
    ).sort_values(["month", "model_type", "rmse", "model"]).reset_index(drop=True)


def build_window_benchmark(
    prediction_frame: pd.DataFrame,
    *,
    target_col: str,
    model_columns: list[str],
) -> pd.DataFrame:
    if "month" not in prediction_frame.columns:
        return pd.DataFrame()
    windows = {
        "Jan": (1,),
        "Jan-Mar": (1, 2, 3),
        "Apr-Jun": (4, 5, 6),
        "Apr-Sep": (4, 5, 6, 7, 8, 9),
        "FullYear": tuple(range(1, 13)),
    }
    rows: list[dict[str, Any]] = []
    for window_name, months in windows.items():
        group = prediction_frame[prediction_frame["month"].isin(months)]
        if group.empty:
            continue
        y_true = group[target_col].to_numpy(dtype=float)
        for model_name in model_columns:
            rows.append(
                regression_metric_row(
                    y_true,
                    group[model_name].to_numpy(dtype=float),
                    model=model_name,
                    model_type="baseline" if model_name.startswith("Baseline") else "ml",
                    feature_group="all",
                    n_train=0,
                    n_test=len(group),
                )
                | {"window": window_name, "months": " ".join(str(month) for month in months)}
            )
    return pd.DataFrame(rows).sort_values(["window", "model_type", "rmse", "model"]).reset_index(drop=True)


def get_feature_importance(
    model_pipe: Pipeline,
    feature_cols: list[str],
    X_reference: pd.DataFrame | None = None,
    y_reference: np.ndarray | None = None,
) -> pd.DataFrame:
    estimator = model_pipe.named_steps["model"]

    importance: np.ndarray | None = None
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()

    if importance is None and X_reference is not None and y_reference is not None and len(X_reference) >= 2:
        from sklearn.inspection import permutation_importance

        try:
            result = permutation_importance(
                model_pipe,
                X_reference,
                y_reference,
                n_repeats=10,
                random_state=42,
                scoring="neg_root_mean_squared_error",
            )
            importance = np.asarray(result.importances_mean, dtype=float)
        except Exception:
            importance = None

    if importance is None or len(importance) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = np.nan_to_num(np.abs(importance), nan=0.0, posinf=0.0, neginf=0.0)
    if len(importance) != len(feature_cols):
        feature_cols = feature_cols[: len(importance)]
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_plot(
    importance_frame: pd.DataFrame,
    output_path: Path,
    title: str = "Top 15 Features Driving Yield Prediction",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 8))

    if importance_frame.empty:
        plt.text(
            0.5,
            0.5,
            "Feature importance unavailable\nNot enough data or model support",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        plt.axis("off")
    else:
        plot_frame = importance_frame.head(15).sort_values("importance", ascending=True)
        ax = sns.barplot(data=plot_frame, x="importance", y="feature", color="#4C72B0")
        ax.set_title(title, pad=14)
        ax.set_xlabel("importance")
        ax.set_ylabel("feature")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        colors = sns.color_palette("viridis", n_colors=len(plot_frame))
        for patch, color in zip(ax.patches, colors):
            patch.set_facecolor(color)
        for patch, value in zip(ax.patches, plot_frame["importance"].tolist()):
            ax.text(
                patch.get_width(),
                patch.get_y() + patch.get_height() / 2.0,
                f"  {value:.6f}",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return output_path


def infer_state_label(county_ids: pd.Series) -> str | None:
    prefixes = set(county_ids.dropna().astype(str).str.zfill(5).str[:2])
    if len(prefixes) != 1:
        return None
    return _STATE_ABBR_BY_FIPS_PREFIX.get(next(iter(prefixes)))


def default_output_dir(root: Path, merged: pd.DataFrame, crop_type: str | None) -> Path:
    years = sorted(pd.to_numeric(merged["year"], errors="coerce").dropna().astype(int).unique())
    year_label = f"{years[0]}_{years[-1]}" if years else "unknown_years"
    crop_label = (normalize_crop_type(crop_type) or "mixed").replace(" ", "_")
    state_label = infer_state_label(merged["county_id"])
    parts = [crop_label]
    if state_label is not None:
        parts.append(state_label)
    parts.append(year_label)
    return root / "outputs" / DEFAULT_OUTPUT_SUBDIR / "_".join(parts)


def resolve_monthly_path(root: Path, monthly_path: Path | None) -> Path:
    if monthly_path is not None:
        if monthly_path.exists():
            return monthly_path
        raise FileNotFoundError(f"Monthly feature table not found: {monthly_path}")

    candidates = discover_monthly_candidates(root)
    if not candidates:
        raise FileNotFoundError(
            "Could not find a monthly feature table. "
            "Pass --monthly-path or run the feature forecasting server first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_usda_paths(
    root: Path,
    usda_paths: list[Path] | None,
    monthly_frame: pd.DataFrame | None,
    crop_type: str | None,
) -> list[Path]:
    if usda_paths:
        existing = [path for path in usda_paths if path.exists()]
        missing = [path for path in usda_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"USDA yield file(s) not found: {missing}")
        return existing

    candidates = discover_usda_candidates(root)
    if not candidates:
        raise FileNotFoundError(
            "Could not find any USDA yield CSV files under the repository. "
            "Pass --usda-path explicitly."
        )
    selected = select_best_usda_paths(candidates, monthly_frame, crop_type)
    return selected or candidates


def validate_overlap_or_raise(
    monthly_frame: pd.DataFrame,
    usda_frame: pd.DataFrame,
    overlap_rows: int,
    min_overlap_rows: int,
) -> None:
    summary = overlap_summary(monthly_frame, usda_frame)
    if overlap_rows >= min_overlap_rows:
        return

    suggestions = suggest_fips_codes(usda_frame, monthly_frame, limit=20)
    suggestion_text = " ".join(suggestions) if suggestions else "(no suggestion available)"
    raise ValueError(
        "Matched county-year overlap is too small for reliable training.\n"
        f"  monthly counties: {monthly_frame['county_id'].nunique()}\n"
        f"  monthly years: {monthly_frame['year'].nunique()}\n"
        f"  USDA counties: {usda_frame['county_id'].nunique()}\n"
        f"  USDA years: {usda_frame['year'].nunique()}\n"
        f"  county overlap: {summary['county_overlap']}\n"
        f"  year overlap: {summary['year_overlap']}\n"
        f"  county-year overlap: {summary['county_year_overlap']}\n"
        f"  required rows: {min_overlap_rows}\n\n"
        "To fix the mismatch, rerun the feature server with a matching county list, for example:\n"
        f"  --fips-codes {suggestion_text}"
    )


def run_yield_regression(
    monthly_path: Path | None = None,
    usda_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    crop_type: str | None = None,
    feature_group: str = "all",
    target_grain: str = "monthly",
    min_overlap_rows: int = DEFAULT_MIN_OVERLAP_ROWS,
    random_state: int = 42,
    include_optional_models: bool = False,
) -> RegressionArtifacts:
    if target_grain not in TARGET_GRAINS:
        raise ValueError(f"target_grain must be one of {TARGET_GRAINS}; got {target_grain!r}.")
    root = Path(__file__).resolve().parents[2]
    resolved_monthly_path = resolve_monthly_path(root, monthly_path)
    monthly = read_monthly_features(resolved_monthly_path)
    if crop_type is not None:
        crop_type = normalize_crop_type(crop_type)
        if "crop_type" in monthly.columns:
            monthly = monthly[monthly["crop_type"].map(normalize_crop_type).eq(crop_type)]
        else:
            print("Warning: monthly feature table has no crop_type column; skipping crop filter.")
    elif "crop_type" in monthly.columns and monthly["crop_type"].notna().nunique() == 1:
        crop_type = normalize_crop_type(monthly["crop_type"].dropna().iloc[0])

    if monthly.empty:
        raise ValueError(
            "No monthly rows remain after applying the crop filter. "
            "Check that --crop-type matches the server output."
        )

    selected_features = selected_feature_columns(feature_group)
    resolved_usda_paths = resolve_usda_paths(root, usda_paths, monthly, crop_type)
    usda = load_usda_yield_tables(resolved_usda_paths, crop_type=crop_type)
    if crop_type is not None and "crop_type" in usda.columns:
        usda = usda[usda["crop_type"].map(normalize_crop_type).eq(crop_type)]

    if target_grain == "monthly":
        merged, feature_cols, overlap_error = build_monthly_training_frame(monthly, usda, selected_features)
        overlap_frame = monthly
        label_strategy = "annual_yield_copied_to_months"
        training_frame_filename = "merged_monthly_training_frame.csv"
    else:
        annual = aggregate_growing_season_features(monthly, selected_features)
        merged, feature_cols, overlap_error = build_training_frame(monthly, usda, selected_features)
        overlap_frame = annual
        label_strategy = "annualized_april_september_features"
        training_frame_filename = "merged_annual_training_frame.csv"
    if merged.empty:
        suggestions = suggest_fips_codes(usda, monthly, limit=20)
        suggestion_text = " ".join(suggestions) if suggestions else "(no suggestion available)"
        raise ValueError(
            overlap_error
            + "\n"
            + "Suggested server retry:\n"
            + f"  --fips-codes {suggestion_text}"
        )

    overlap_rows = len(merged)
    summary = overlap_summary(overlap_frame, usda)
    validate_overlap_or_raise(monthly, usda, overlap_rows, min_overlap_rows)

    if output_dir is None:
        output_dir = default_output_dir(root, merged, crop_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols_before_pruning = list(feature_cols)
    feature_cols, pruning_report = prune_feature_columns(merged, feature_cols)
    pruning_report_path = save_json(
        pruning_report,
        output_dir / "feature_pruning_report.json",
    )

    train_df, test_df, split_mode, split_metadata = split_dataset(merged, random_state=random_state)
    ml_results, fitted_models = evaluate_models(
        train_df,
        test_df,
        feature_cols,
        "yield_bu_acre",
        include_optional_models=include_optional_models,
        random_state=random_state,
        tune_hyperparameters=True,
        feature_group=feature_group,
    )
    baseline_results = evaluate_naive_baselines(
        train_df,
        test_df,
        "yield_bu_acre",
        feature_group=feature_group,
    )
    results = pd.concat([ml_results, baseline_results], ignore_index=True).sort_values(
        ["model_type", "rmse", "mae", "model"]
    ).reset_index(drop=True)
    best_model_name = ml_results.iloc[0]["model"]

    holdout_model = fitted_models[best_model_name]
    final_model = clone(holdout_model)
    final_model.fit(merged[feature_cols], merged["yield_bu_acre"].to_numpy(dtype=float))

    importance = get_feature_importance(
        final_model,
        feature_cols,
        X_reference=merged[feature_cols],
        y_reference=merged["yield_bu_acre"].to_numpy(dtype=float),
    )
    importance_path = save_feature_importance_plot(
        importance,
        output_dir / "yield_feature_importance.png",
    )
    importance_csv_path = output_dir / "yield_feature_importance.csv"
    importance.to_csv(importance_csv_path, index=False)

    results_path = output_dir / "yield_model_benchmark.csv"
    results.to_csv(results_path, index=False)

    feature_group_benchmark = evaluate_feature_group_benchmarks(
        train_df,
        test_df,
        feature_cols,
        "yield_bu_acre",
        random_state=random_state,
        include_optional_models=include_optional_models,
        tune_hyperparameters=True,
    )
    feature_group_benchmark_path = output_dir / "feature_group_benchmark.csv"
    feature_group_benchmark.to_csv(feature_group_benchmark_path, index=False)

    year_cv_benchmark = evaluate_year_cv(
        merged,
        feature_cols,
        "yield_bu_acre",
        random_state=random_state,
        include_optional_models=include_optional_models,
        tune_hyperparameters=True,
        feature_group=feature_group,
    )
    year_cv_benchmark_path = output_dir / "year_cv_benchmark.csv"
    year_cv_benchmark.to_csv(year_cv_benchmark_path, index=False)

    prediction_frame = build_prediction_frame(
        holdout_model,
        train_df,
        test_df,
        feature_cols,
        "yield_bu_acre",
        best_model_name,
    )
    month_benchmark = build_month_benchmark(
        prediction_frame,
        target_col="yield_bu_acre",
        model_columns=[best_model_name, "BaselineTrainMean", "BaselinePreviousYearSameCounty"],
    )
    month_benchmark_path = output_dir / "month_benchmark.csv"
    month_benchmark.to_csv(month_benchmark_path, index=False)

    window_benchmark = build_window_benchmark(
        prediction_frame,
        target_col="yield_bu_acre",
        model_columns=[best_model_name, "BaselineTrainMean", "BaselinePreviousYearSameCounty"],
    )
    window_benchmark_path = output_dir / "window_benchmark.csv"
    window_benchmark.to_csv(window_benchmark_path, index=False)

    residuals = build_residual_frame(holdout_model, test_df, feature_cols, "yield_bu_acre")
    residuals_path = output_dir / "prediction_residuals.csv"
    residuals.to_csv(residuals_path, index=False)

    training_frame_path = output_dir / training_frame_filename
    merged.to_csv(training_frame_path, index=False)

    model_path = output_dir / "best_yield_model.joblib"
    joblib.dump(final_model, model_path)

    target_units = sorted(str(value) for value in merged["target_unit"].dropna().unique()) if "target_unit" in merged.columns else []
    metadata = {
        "monthly_path": resolved_monthly_path,
        "usda_paths": resolved_usda_paths,
        "crop_type": crop_type or "mixed",
        "feature_group": feature_group,
        "target_grain": target_grain,
        "label_strategy": label_strategy,
        "feature_columns": feature_cols,
        "feature_columns_before_pruning": feature_cols_before_pruning,
        "target_column": "yield_bu_acre",
        "target_units": target_units,
        "split_mode": split_mode,
        "split_strategy": split_metadata["strategy"],
        "test_year": split_metadata["test_year"],
        "rows": len(merged),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "county_overlap": summary["county_overlap"],
        "year_overlap": summary["year_overlap"],
        "county_year_overlap": summary["county_year_overlap"],
        "best_model_name": best_model_name,
        "models_benchmarked": results["model"].tolist(),
        "random_state": random_state,
        "pruning": pruning_report,
        "uses_forecast_generated_features": False,
        "results_csv": results_path,
        "training_frame_csv": training_frame_path,
        "feature_importance_csv": importance_csv_path,
        "feature_importance_png": importance_path,
        "feature_group_benchmark_csv": feature_group_benchmark_path,
        "year_cv_benchmark_csv": year_cv_benchmark_path,
        "month_benchmark_csv": month_benchmark_path,
        "window_benchmark_csv": window_benchmark_path,
        "feature_pruning_report_json": pruning_report_path,
        "prediction_residuals_csv": residuals_path,
        "model_path": model_path,
    }
    metadata_path = save_json(metadata, output_dir / "yield_model_metadata.json")

    summary_path = output_dir / "yield_dataset_summary.txt"
    summary_lines = [
        f"monthly_path={resolved_monthly_path}",
        f"usda_paths={', '.join(str(path) for path in resolved_usda_paths)}",
        f"crop_type={crop_type or 'mixed'}",
        f"feature_group={feature_group}",
        f"target_grain={target_grain}",
        f"label_strategy={label_strategy}",
        f"split_mode={split_mode}",
        f"rows={len(merged)}",
        f"train_rows={len(train_df)}",
        f"test_rows={len(test_df)}",
        f"feature_count_before_pruning={len(feature_cols_before_pruning)}",
        f"feature_count_after_pruning={len(feature_cols)}",
        f"county_overlap={summary['county_overlap']}",
        f"year_overlap={summary['year_overlap']}",
        f"county_year_overlap={summary['county_year_overlap']}",
        f"best_model={best_model_name}",
        f"results_csv={results_path}",
        f"feature_group_benchmark_csv={feature_group_benchmark_path}",
        f"year_cv_benchmark_csv={year_cv_benchmark_path}",
        f"month_benchmark_csv={month_benchmark_path}",
        f"window_benchmark_csv={window_benchmark_path}",
        f"feature_pruning_report_json={pruning_report_path}",
        f"prediction_residuals_csv={residuals_path}",
        f"training_frame_csv={training_frame_path}",
        f"model_path={model_path}",
        f"metadata_json={metadata_path}",
        f"importance_png={importance_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n=== Yield Regression Benchmark ===")
    print(f"Monthly table: {resolved_monthly_path}")
    print(f"USDA file(s): {', '.join(str(path) for path in resolved_usda_paths)}")
    print(f"Split mode: {split_mode}")
    print(results.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Feature importance plot: {importance_path}")
    print(f"Feature group benchmark: {feature_group_benchmark_path}")
    print(f"Year CV benchmark: {year_cv_benchmark_path}")
    print(f"Month benchmark: {month_benchmark_path}")
    print(f"Window benchmark: {window_benchmark_path}")

    return RegressionArtifacts(
        results=results,
        split_mode=split_mode,
        train_rows=len(train_df),
        test_rows=len(test_df),
        overlap_rows=overlap_rows,
        overlap_counties=summary["county_overlap"],
        overlap_years=summary["year_overlap"],
        best_model_name=best_model_name,
        fitted_model=final_model,
        output_dir=output_dir,
        results_path=results_path,
        training_frame_path=training_frame_path,
        model_path=model_path,
        metadata_path=metadata_path,
        importance_path=importance_path,
        summary_path=summary_path,
        feature_group_benchmark_path=feature_group_benchmark_path,
        year_cv_benchmark_path=year_cv_benchmark_path,
        month_benchmark_path=month_benchmark_path,
        window_benchmark_path=window_benchmark_path,
        pruning_report_path=pruning_report_path,
        residuals_path=residuals_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark CropNet yield regression from monthly or annualized features."
    )
    parser.add_argument(
        "--monthly-path",
        type=Path,
        default=None,
        help="Monthly feature table from the feature server. If omitted, the newest matching file is discovered.",
    )
    parser.add_argument(
        "--usda-path",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help="One or more USDA county yield CSV files. If omitted, matching USDA files are discovered automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for benchmark CSVs and the feature importance plot.",
    )
    parser.add_argument(
        "--crop-type",
        type=str,
        default=None,
        help="Optional crop filter, for example corn or soybeans.",
    )
    parser.add_argument(
        "--feature-group",
        type=str,
        default="all",
        choices=sorted({"all", "ag", "ndvi", "weather", "ag_ndvi", "ag_weather", "ndvi_weather"}),
        help="Feature subset to benchmark.",
    )
    parser.add_argument(
        "--target-grain",
        type=str,
        default="monthly",
        choices=TARGET_GRAINS,
        help="Use monthly rows with copied annual labels, or legacy annualized rows.",
    )
    parser.add_argument(
        "--min-overlap-rows",
        type=int,
        default=DEFAULT_MIN_OVERLAP_ROWS,
        help="Minimum matched county-year rows required before training starts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split and model training.",
    )
    parser.add_argument(
        "--include-optional-models",
        action="store_true",
        help="Also benchmark optional XGBoost/LightGBM models when installed.",
    )
    return parser


def flatten_usda_path_groups(usda_path_groups: list[list[Path]] | None) -> list[Path] | None:
    if not usda_path_groups:
        return None
    return [path for group in usda_path_groups for path in group]


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    usda_paths = flatten_usda_path_groups(args.usda_path)

    try:
        run_yield_regression(
            monthly_path=args.monthly_path,
            usda_paths=usda_paths,
            output_dir=args.output_dir,
            crop_type=args.crop_type,
            feature_group=args.feature_group,
            target_grain=args.target_grain,
            min_overlap_rows=args.min_overlap_rows,
            random_state=args.random_state,
            include_optional_models=args.include_optional_models,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
