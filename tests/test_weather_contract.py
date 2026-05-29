"""Tests for weather feature names used by the forecasting model."""

from __future__ import annotations

import pandas as pd

from crop_fusion_ai.preprocessing.weather import extract_weather_features
from cropnet_forecasting.features import WEATHER_CORE


def test_extract_weather_features_exports_model_core_columns() -> None:
    """Weather preprocessing must expose the exact model input feature names."""
    frame = pd.DataFrame(
        [
            {
                "date": "2017-12-01",
                "temp_mean": 24.0,
                "temp_max": 31.0,
                "temp_min": 18.0,
                "precipitation": 0.0,
                "humidity": 73.0,
                "solar_radiation": 138.0,
                "wind_speed": 2.2,
            },
            {
                "date": "2017-12-02",
                "temp_mean": 25.0,
                "temp_max": 32.0,
                "temp_min": 19.0,
                "precipitation": 12.0,
                "humidity": 71.0,
                "solar_radiation": 141.0,
                "wind_speed": 2.3,
            },
        ]
    )

    monthly = extract_weather_features(frame, county_id="01003", crop_type="corn")

    assert not monthly.empty
    for column in WEATHER_CORE:
        assert column in monthly.columns
