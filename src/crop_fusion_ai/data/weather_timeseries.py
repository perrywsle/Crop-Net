"""Utilities for reducing weather time-series records into model features."""

from collections.abc import Mapping, Sequence

WeatherRecord = Mapping[str, float]


def summarize_weather_sequence(records: Sequence[WeatherRecord]) -> dict[str, float]:
    """Summarize previous weather records into fixed tabular features.

    This is the bridge between CropNet's time-series weather modality and the
    current sklearn baseline. A future recurrent/Transformer weather model can
    replace this reducer without changing the surrounding training flow.
    """
    if not records:
        msg = "At least one weather record is required"
        raise ValueError(msg)

    temperatures = _values_for_key(records, "temperature")
    rainfall = _values_for_key(records, "rainfall")
    humidity = _values_for_key(records, "humidity")
    solar_radiation = _values_for_key(records, "solar_radiation")

    features: dict[str, float] = {
        "weather_steps": float(len(records)),
    }
    _add_summary(features, "temperature", temperatures)
    _add_summary(features, "rainfall", rainfall)
    _add_summary(features, "humidity", humidity)
    _add_summary(features, "solar_radiation", solar_radiation)
    return features


def _values_for_key(records: Sequence[WeatherRecord], key: str) -> list[float]:
    """Extract numeric values for one weather key."""
    return [float(record[key]) for record in records if key in record]


def _add_summary(
    features: dict[str, float],
    prefix: str,
    values: list[float],
) -> None:
    """Add mean/min/max/sum summary features when values are present."""
    if not values:
        return
    features[f"{prefix}_mean"] = sum(values) / len(values)
    features[f"{prefix}_min"] = min(values)
    features[f"{prefix}_max"] = max(values)
    features[f"{prefix}_sum"] = sum(values)
