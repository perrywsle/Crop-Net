from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import Callable

from .features import META_COLS
from .data import prepare_monthly_features

@dataclass(slots=True)
class BlankFillResult:
    predictions: pd.DataFrame
    prediction_long: pd.DataFrame


ProgressCallback = Callable[[str, int, int, str], None]

def build_seasonal_lookup(monthly_features: pd.DataFrame, feature_names: list[str]) -> dict[tuple[str, str, int, int], np.ndarray]:
    lookup: dict[tuple[str, str, int, int], np.ndarray] = {}
    for row in monthly_features[META_COLS + feature_names].itertuples(index=False):
        values = np.asarray(row[4:], dtype=float)
        lookup[(str(row[0]).zfill(5), str(row[1]), int(row[2]), int(row[3]))] = values
    return lookup

def _pad_window(values: np.ndarray, seq_len: int) -> np.ndarray:
    if len(values) >= seq_len:
        return values[-seq_len:]
    if len(values) == 0:
        raise ValueError("Cannot build a prediction window from an empty history.")
    pad = np.repeat(values[[0]], seq_len - len(values), axis=0)
    return np.vstack([pad, values])

def _next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def rollout_blank_fill(
    predictor,
    monthly_features: pd.DataFrame,
    year: int,
    known_months: int,
) -> BlankFillResult:
    feature_names = predictor.feature_names
    prepared = prepare_monthly_features(monthly_features, feature_names)
    seasonal_lookup = build_seasonal_lookup(prepared, feature_names)
    rows = []
    for (county_id, crop_type), group in prepared.groupby(["county_id", "crop_type"], sort=True):
        group = group.sort_values(["year", "month"]).reset_index(drop=True)
        known_history = group[(group["year"] < year) | ((group["year"] == year) & (group["month"] <= known_months))].copy()
        if known_history.empty:
            continue
        history_rows = known_history[META_COLS + feature_names].copy()
        for month in range(known_months + 1, 13):
            window = _pad_window(history_rows[feature_names].to_numpy(dtype=float), predictor.seq_len)
            seasonal_base = seasonal_lookup.get((str(county_id).zfill(5), str(crop_type), year - 1, month))
            if seasonal_base is None and len(history_rows) > 0:
                seasonal_base = history_rows[feature_names].to_numpy(dtype=float)[-1]
                source_note = "fallback_last_history"
            else:
                source_note = "seasonal_last_year"
            forecast = predictor.predict_next(window, seasonal_base=seasonal_base)
            predicted_row = {
                "county_id": str(county_id).zfill(5),
                "crop_type": str(crop_type),
                "year": int(year),
                "month": int(month),
                "known_months": int(known_months),
                "source_note": source_note,
            }
            for name, value in zip(feature_names, forecast, strict=True):
                predicted_row[name] = float(value)
            rows.append(predicted_row)
            history_rows = pd.concat([history_rows, pd.DataFrame([{k: predicted_row[k] for k in META_COLS + feature_names}])], ignore_index=True)
    predictions = pd.DataFrame(rows)
    long_records = []
    if not predictions.empty:
        for row in predictions.itertuples(index=False):
            for feature_name in feature_names:
                long_records.append({
                    "county_id": row.county_id,
                    "crop_type": row.crop_type,
                    "year": row.year,
                    "month": row.month,
                    "known_months": row.known_months,
                    "feature": feature_name,
                    "y_pred": getattr(row, feature_name),
                    "source_note": row.source_note,
                })
    return BlankFillResult(predictions=predictions, prediction_long=pd.DataFrame(long_records))


def rollout_autoregressive(
    predictor,
    monthly_features: pd.DataFrame,
    horizon: int = 12,
    progress: ProgressCallback | None = None,
) -> BlankFillResult:
    feature_names = predictor.feature_names
    prepared = prepare_monthly_features(monthly_features, feature_names)
    seasonal_lookup = build_seasonal_lookup(prepared, feature_names)
    rows = []

    for (county_id, crop_type), group in prepared.groupby(["county_id", "crop_type"], sort=True):
        group = group.sort_values(["year", "month"]).reset_index(drop=True)
        if group.empty:
            continue

        history_rows = group[META_COLS + feature_names].copy()
        current_year = int(group.iloc[-1]["year"])
        current_month = int(group.iloc[-1]["month"])

        for step in range(1, horizon + 1):
            current_year, current_month = _next_month(current_year, current_month)
            window = _pad_window(history_rows[feature_names].to_numpy(dtype=float), predictor.seq_len)
            seasonal_base = seasonal_lookup.get((str(county_id).zfill(5), str(crop_type), current_year - 1, current_month))
            if seasonal_base is None and len(history_rows) > 0:
                seasonal_base = history_rows[feature_names].to_numpy(dtype=float)[-1]
                source_note = "fallback_last_history"
            else:
                source_note = "seasonal_last_year"
            forecast = predictor.predict_next(window, seasonal_base=seasonal_base)
            predicted_row = {
                "county_id": str(county_id).zfill(5),
                "crop_type": str(crop_type),
                "year": int(current_year),
                "month": int(current_month),
                "forecast_step": int(step),
                "source_note": source_note,
            }
            for name, value in zip(feature_names, forecast, strict=True):
                predicted_row[name] = float(value)
            rows.append(predicted_row)
            history_rows = pd.concat(
                [
                    history_rows,
                    pd.DataFrame([{k: predicted_row[k] for k in META_COLS + feature_names}]),
                ],
                ignore_index=True,
            )
            if progress is not None:
                progress(
                    "forecast",
                    step,
                    horizon,
                    f"Forecasted {step}/{horizon} months",
                )

    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions["date"] = pd.to_datetime(
            predictions[["year", "month"]].assign(day=1)
        )

    long_records = []
    if not predictions.empty:
        for row in predictions.itertuples(index=False):
            for feature_name in feature_names:
                long_records.append(
                    {
                        "county_id": row.county_id,
                        "crop_type": row.crop_type,
                        "year": row.year,
                        "month": row.month,
                        "forecast_step": row.forecast_step,
                        "date": row.date,
                        "feature": feature_name,
                        "y_pred": getattr(row, feature_name),
                        "source_note": row.source_note,
                    }
                )

    return BlankFillResult(predictions=predictions, prediction_long=pd.DataFrame(long_records))
