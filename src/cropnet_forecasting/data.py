from __future__ import annotations

from pathlib import Path

import pandas as pd

from .features import META_COLS, selected_feature_columns

def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table suffix: {suffix}")

def load_monthly_features(path: str | Path, feature_group: str = "all") -> pd.DataFrame:
    return coerce_monthly_features(read_table(path), feature_group=feature_group)

def coerce_monthly_features(frame: pd.DataFrame, feature_group: str = "all") -> pd.DataFrame:
    feature_names = selected_feature_columns(feature_group)
    missing = [col for col in META_COLS + feature_names if col not in frame.columns]
    if missing:
        raise ValueError(f"Monthly feature table is missing required columns: {missing}")
    out = frame[META_COLS + feature_names].copy()
    out["county_id"] = out["county_id"].astype(str).str.zfill(5)
    out["crop_type"] = out["crop_type"].astype(str)
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out.sort_values(["county_id", "crop_type", "year", "month"]).reset_index(drop=True)


def prepare_monthly_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Coerce and impute a monthly feature frame for inference."""
    missing = [col for col in META_COLS + feature_names if col not in frame.columns]
    if missing:
        raise ValueError(f"Monthly feature table is missing required model columns: {missing}")

    coerced = frame[META_COLS + feature_names].copy()
    coerced["county_id"] = coerced["county_id"].astype(str).str.zfill(5)
    coerced["crop_type"] = coerced["crop_type"].astype(str)
    coerced["year"] = coerced["year"].astype(int)
    coerced["month"] = coerced["month"].astype(int)

    if coerced.empty:
        return coerced

    filled_groups: list[pd.DataFrame] = []
    for _, group in coerced.groupby(["county_id", "crop_type"], sort=True):
        group = group.sort_values(["year", "month"]).reset_index(drop=True).copy()
        group[feature_names] = (
            group[feature_names].interpolate(limit_direction="both").ffill().bfill()
        )
        medians = group[feature_names].median(numeric_only=True)
        group[feature_names] = group[feature_names].fillna(medians)
        group[feature_names] = group[feature_names].fillna(0.0)
        filled_groups.append(group)

    return pd.concat(filled_groups, ignore_index=True).sort_values(
        ["county_id", "crop_type", "year", "month"]
    ).reset_index(drop=True)
