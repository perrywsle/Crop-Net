"""Feature extraction for CropNet weather time-series data."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crop_fusion_ai.preprocessing.common import (
    FrameLike,
    load_weather_frame,
    longest_true_streak,
)


@dataclass(frozen=True, slots=True)
class WeatherFeatureConfig:
    """Thresholds used by weather preprocessing."""

    base_temperature_c: float = 10.0
    heat_threshold_c: float = 35.0
    frost_threshold_c: float = 0.0
    dry_day_precip_mm: float = 1.0
    rainy_day_precip_mm: float = 1.0
    heavy_rain_day_precip_mm: float = 10.0


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "datetime", "day"),
    "temperature_mean": (
        "temperature_mean",
        "temp_mean",
        "tmean",
        "mean_temp",
        "avg_temp",
        "avg temperature (k)",
    ),
    "temperature_max": (
        "temperature_max",
        "temp_max",
        "tmax",
        "max_temp",
        "max temperature (k)",
    ),
    "temperature_min": (
        "temperature_min",
        "temp_min",
        "tmin",
        "min_temp",
        "min temperature (k)",
    ),
    "precipitation": (
        "precipitation",
        "rainfall",
        "precip",
        "prcp",
        "precipitation (kg m**-2)",
    ),
    "humidity": ("humidity", "relative_humidity", "rh", "relative humidity (%)"),
    "wind_speed": ("wind_speed", "wind", "ws", "wind speed (m s**-1)"),
    "solar_radiation": (
        "solar_radiation",
        "radiation",
        "swrad",
        "shortwave_radiation",
        "downward shortwave radiation flux (w m**-2)",
    ),
}


def _first_present_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized_lookup = {column.strip().lower(): column for column in frame.columns}
    for column in aliases:
        normalized = column.strip().lower()
        if normalized in normalized_lookup:
            return normalized_lookup[normalized]
    return None


def _normalize_weather_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    for canonical, aliases in _COLUMN_ALIASES.items():
        source = _first_present_column(renamed, aliases)
        if source is not None and source != canonical:
            renamed = renamed.rename(columns={source: canonical})
    return renamed


def _saturation_vapor_pressure(temp_c: pd.Series) -> pd.Series:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def _safe_series_mean(series: pd.Series) -> float:
    values = series.astype(np.float64)
    return float(values.mean()) if not values.empty else 0.0


def _safe_series_sum(series: pd.Series) -> float:
    values = series.astype(np.float64)
    return float(values.sum()) if not values.empty else 0.0


def _safe_series_max(series: pd.Series) -> float:
    values = series.astype(np.float64)
    return float(values.max()) if not values.empty else 0.0


def _safe_series_min(series: pd.Series) -> float:
    values = series.astype(np.float64)
    return float(values.min()) if not values.empty else 0.0


def extract_weather_features(
    weather_input: FrameLike,
    *,
    county_id: str | None = None,
    crop_type: str | None = None,
    config: WeatherFeatureConfig | None = None,
    climatology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate daily weather records into monthly CropNet features."""
    config = config or WeatherFeatureConfig()
    frame = _normalize_weather_columns(load_weather_frame(weather_input))

    if "date" not in frame.columns:
        msg = "Weather data must include a date column"
        raise ValueError(msg)
    if "temperature_mean" not in frame.columns and not (
        "temperature_max" in frame.columns and "temperature_min" in frame.columns
    ):
        msg = (
            "Weather data must include temperature_mean or "
            "temperature_min/temperature_max"
        )
        raise ValueError(msg)
    if "precipitation" not in frame.columns:
        frame["precipitation"] = 0.0

    if "Daily/Monthly" in frame.columns:
        daily_mask = frame["Daily/Monthly"].astype(str).str.strip().str.lower() == "daily"
        if daily_mask.any():
            frame = frame[daily_mask].copy()

    if "date" not in frame.columns:
        year_column = _first_present_column(frame, ("year",))
        month_column = _first_present_column(frame, ("month",))
        day_column = _first_present_column(frame, ("day",))
        if year_column is not None and month_column is not None and day_column is not None:
            frame["date"] = pd.to_datetime(
                frame[[year_column, month_column, day_column]].rename(
                    columns={
                        year_column: "year",
                        month_column: "month",
                        day_column: "day",
                    }
                ),
                errors="coerce",
            )
        else:
            msg = "Weather data must include a date column or year/month/day columns"
            raise ValueError(msg)
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    if frame.empty:
        msg = "Weather data does not contain any parseable dates"
        raise ValueError(msg)

    if "temperature_mean" not in frame.columns:
        frame["temperature_mean"] = frame[["temperature_max", "temperature_min"]].mean(
            axis=1
        )
    if "temperature_max" not in frame.columns:
        frame["temperature_max"] = frame["temperature_mean"]
    if "temperature_min" not in frame.columns:
        frame["temperature_min"] = frame["temperature_mean"]

    if (
        frame[["temperature_mean", "temperature_max", "temperature_min"]]
        .astype(np.float64)
        .to_numpy()
        .max(initial=0.0)
        > 100.0
    ):
        for column in ("temperature_mean", "temperature_max", "temperature_min"):
            frame[column] = frame[column].astype(np.float64) - 273.15

    if "humidity" not in frame.columns:
        frame["humidity"] = np.nan
    if "solar_radiation" not in frame.columns:
        frame["solar_radiation"] = np.nan
    if "wind_speed" not in frame.columns:
        frame["wind_speed"] = np.nan

    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["days_in_month"] = frame["date"].dt.days_in_month
    frame["gdd"] = np.maximum(
        0.0, frame["temperature_mean"].astype(np.float64) - config.base_temperature_c
    )
    frame["extreme_heat_day"] = (
        frame["temperature_max"].astype(np.float64) >= config.heat_threshold_c
    )
    frame["frost_day"] = (
        frame["temperature_min"].astype(np.float64) <= config.frost_threshold_c
    )
    frame["dry_day"] = (
        frame["precipitation"].astype(np.float64) < config.dry_day_precip_mm
    )
    frame["rainy_day"] = (
        frame["precipitation"].astype(np.float64) >= config.rainy_day_precip_mm
    )
    if frame["humidity"].notna().any():
        es = _saturation_vapor_pressure(frame["temperature_mean"].astype(np.float64))
        ea = es * (frame["humidity"].astype(np.float64) / 100.0)
        frame["vpd"] = (es - ea).clip(lower=0.0)
    else:
        frame["vpd"] = np.nan

    monthly_rows: list[dict[str, float | int | str | None]] = []
    for (year, month), group in frame.groupby(["year", "month"], sort=True):
        temp_mean = group["temperature_mean"].astype(np.float64)
        temp_max = group["temperature_max"].astype(np.float64)
        temp_min = group["temperature_min"].astype(np.float64)
        precipitation = group["precipitation"].astype(np.float64)
        humidity = group["humidity"].astype(np.float64)
        solar_radiation = group["solar_radiation"].astype(np.float64)
        wind_speed = group["wind_speed"].astype(np.float64)
        vpd = group["vpd"].astype(np.float64)

        days_in_month = int(monthrange(int(year), int(month))[1])
        rainy_days = int(group["rainy_day"].sum())
        dry_days = int(group["dry_day"].sum())
        heavy_rain_days = int((precipitation >= config.heavy_rain_day_precip_mm).sum())
        monthly_precipitation = _safe_series_sum(precipitation)
        mean_vpd = float(vpd.mean()) if vpd.notna().any() else 0.0
        potential_evap = max(0.0, mean_vpd) * days_in_month * 10.0
        drought_index = float(
            max(0.0, potential_evap - monthly_precipitation)
            / (potential_evap + monthly_precipitation + 1e-6)
        )
        solar_sum = _safe_series_sum(solar_radiation.fillna(0.0))
        temp_range_series = temp_max - temp_min
        heat_stress_days = int(group["extreme_heat_day"].sum())
        cold_stress_days = int(group["frost_day"].sum())

        monthly_rows.append(
            {
                "county_id": county_id,
                "crop_type": crop_type,
                "year": int(year),
                "month": int(month),
                "weather_temp_mean": _safe_series_mean(temp_mean),
                "weather_temp_max": _safe_series_max(temp_max),
                "weather_temp_min": _safe_series_min(temp_min),
                "weather_temp_range": float(
                    _safe_series_max(temp_max) - _safe_series_min(temp_min)
                ),
                "weather_gdd": float(group["gdd"].sum()),
                "weather_extreme_heat_days": heat_stress_days,
                "weather_heat_stress_days": heat_stress_days,
                "weather_cold_stress_degree_days": float(
                    np.maximum(0.0, config.frost_threshold_c - temp_min).sum()
                ),
                "weather_cold_stress_days": cold_stress_days,
                "weather_total_precipitation": monthly_precipitation,
                "weather_rainy_days": rainy_days,
                "weather_precipitation_days": rainy_days,
                "weather_dry_days": dry_days,
                "weather_heavy_rain_days": heavy_rain_days,
                "weather_max_dry_streak": int(
                    longest_true_streak(group["dry_day"].tolist())
                ),
                "weather_humidity_mean": float(humidity.mean())
                if humidity.notna().any()
                else np.nan,
                "weather_vpd": mean_vpd,
                "weather_vpd_mean": mean_vpd,
                "weather_wind_speed_mean": float(wind_speed.mean())
                if wind_speed.notna().any()
                else np.nan,
                "weather_wind_mean": float(wind_speed.mean())
                if wind_speed.notna().any()
                else np.nan,
                "weather_wind_speed_max": float(wind_speed.max())
                if wind_speed.notna().any()
                else np.nan,
                "weather_solar_radiation_mean": float(solar_radiation.mean())
                if solar_radiation.notna().any()
                else np.nan,
                "weather_cumulative_solar_radiation": solar_sum,
                "weather_precipitation_intensity": float(
                    monthly_precipitation / max(rainy_days, 1)
                ),
                "weather_weather_volatility_score": float(
                    np.nanstd(temp_mean)
                    + np.nanstd(precipitation)
                    + np.nanstd(solar_radiation)
                ),
                "weather_heat_stress_degree_days": float(
                    np.maximum(0.0, temp_mean - config.heat_threshold_c).sum()
                ),
                "weather_drought_index": drought_index,
                "weather_days_in_month": days_in_month,
                "weather_temp_range_mean": float(temp_range_series.mean()),
            }
        )

    monthly_frame = (
        pd.DataFrame(monthly_rows).sort_values(["year", "month"]).reset_index(drop=True)
    )

    if climatology is None:
        baseline = monthly_frame.groupby("month", as_index=False).agg(
            weather_temperature_baseline=("weather_temp_mean", "mean"),
            weather_rainfall_baseline=("weather_total_precipitation", "mean"),
        )
    else:
        baseline = climatology.copy()
        baseline_columns = set(baseline.columns)
        if {
            "month",
            "weather_temperature_baseline",
            "weather_rainfall_baseline",
        } <= baseline_columns:
            baseline = baseline.loc[
                :,
                ["month", "weather_temperature_baseline", "weather_rainfall_baseline"],
            ]
        elif {
            "month",
            "weather_temp_mean",
            "weather_total_precipitation",
        } <= baseline_columns:
            baseline = baseline.groupby("month", as_index=False).agg(
                weather_temperature_baseline=("weather_temp_mean", "mean"),
                weather_rainfall_baseline=("weather_total_precipitation", "mean"),
            )
        else:
            msg = (
                "Climatology must include month plus either baseline columns "
                "or raw weather_temp_mean/weather_total_precipitation columns"
            )
            raise ValueError(msg)
    monthly_frame = monthly_frame.merge(baseline, on="month", how="left")
    monthly_frame["weather_temperature_anomaly"] = (
        monthly_frame["weather_temp_mean"]
        - monthly_frame["weather_temperature_baseline"]
    )
    monthly_frame["weather_rainfall_anomaly"] = (
        monthly_frame["weather_total_precipitation"]
        - monthly_frame["weather_rainfall_baseline"]
    )
    monthly_frame = monthly_frame.drop(
        columns=["weather_temperature_baseline", "weather_rainfall_baseline"]
    )

    monthly_frame["weather_rainfall_lag_1"] = monthly_frame.groupby("year")[
        "weather_total_precipitation"
    ].shift(1)
    monthly_frame["weather_rainfall_lag_2"] = monthly_frame.groupby("year")[
        "weather_total_precipitation"
    ].shift(2)
    monthly_frame["weather_temp_lag_1"] = monthly_frame.groupby("year")[
        "weather_temp_mean"
    ].shift(1)
    monthly_frame["weather_vpd_lag_1"] = monthly_frame.groupby("year")[
        "weather_vpd"
    ].shift(1)

    monthly_frame["weather_vpd_mean"] = monthly_frame["weather_vpd"]
    monthly_frame["weather_wind_mean"] = monthly_frame["weather_wind_speed_mean"]
    monthly_frame["weather_heat_stress_days"] = monthly_frame["weather_extreme_heat_days"]
    monthly_frame["weather_precipitation_days"] = monthly_frame["weather_rainy_days"]

    return monthly_frame


def derive_weather_time_series_features(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """Derive higher-level weather sequence features from monthly weather records."""
    if monthly_features.empty:
        return pd.DataFrame()

    frame = monthly_features.sort_values(["year", "month"]).reset_index(drop=True)
    months = frame["month"].astype(np.float64)
    temp_series = frame["weather_temp_mean"].astype(np.float64)
    precip_series = frame["weather_total_precipitation"].astype(np.float64)
    vpd_series = frame["weather_vpd"].astype(np.float64)

    temp_slope = (
        float(np.polyfit(months.to_numpy(), temp_series.to_numpy(), 1)[0])
        if len(frame) > 1
        else 0.0
    )
    precip_slope = (
        float(np.polyfit(months.to_numpy(), precip_series.to_numpy(), 1)[0])
        if len(frame) > 1
        else 0.0
    )
    vpd_slope = (
        float(np.polyfit(months.to_numpy(), vpd_series.fillna(0.0).to_numpy(), 1)[0])
        if len(frame) > 1
        else 0.0
    )

    summary = {
        "weather_temperature_trend_slope": temp_slope,
        "weather_rainfall_trend_slope": precip_slope,
        "weather_vpd_trend_slope": vpd_slope,
        "weather_total_precipitation_auc": float(
            np.trapezoid(precip_series.to_numpy(), months.to_numpy())
        )
        if len(frame) > 1
        else float(precip_series.iloc[0]),
        "weather_peak_rainfall_month": int(
            frame.loc[int(precip_series.idxmax()), "month"]
        ),
        "weather_rainfall_seasonality_index": float(
            precip_series.std() / (precip_series.mean() + 1e-6)
        ),
        "weather_heat_stress_total": float(
            frame["weather_heat_stress_degree_days"].sum()
        ),
        "weather_drought_severity_max": float(frame["weather_drought_index"].max()),
    }
    return pd.DataFrame([summary])
