"""Tests for the browser-first yield web service."""

from __future__ import annotations

import math

import pandas as pd

from crop_fusion_ai.web.feature_labels import label_payload
from crop_fusion_ai.web.service import YieldModelService
from cropnet_forecasting.data import prepare_monthly_features


def _synthetic_monthly_frame(service: YieldModelService) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in [4, 5, 6]:
        row: dict[str, object] = {
            "county_id": "19001",
            "crop_type": "corn",
            "year": 2022,
            "month": month,
        }
        for index, feature_name in enumerate(service.feature_names):
            if feature_name == "month":
                row[feature_name] = month
            elif feature_name == "month_sin":
                row[feature_name] = math.sin(2.0 * math.pi * month / 12.0)
            elif feature_name == "month_cos":
                row[feature_name] = math.cos(2.0 * math.pi * month / 12.0)
            else:
                row[feature_name] = (index + month) / 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_labels_are_farmer_friendly() -> None:
    payload = label_payload()

    assert payload["ag_green_pixel_ratio"]["label"] == "Green canopy cover"
    assert payload["ag_green_pixel_ratio"]["group"] == "canopy"
    assert payload["month_sin"]["hidden"] == "true"


def test_prepare_monthly_features_deduplicates_month_column() -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service)

    prepared = prepare_monthly_features(frame, service.feature_names)

    assert list(prepared.columns).count("month") == 1
    assert prepared.shape[0] == 3


def test_prepare_monthly_features_derives_month_sin_cos() -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service).drop(columns=["month_sin", "month_cos"])

    prepared = prepare_monthly_features(frame, service.feature_names)

    assert "month_sin" in prepared.columns
    assert "month_cos" in prepared.columns
    assert prepared[["month_sin", "month_cos"]].notna().all().all()


def test_predict_from_monthly_frame_returns_friendly_payload() -> None:
    service = YieldModelService()
    frame = _synthetic_monthly_frame(service)

    result = service.predict_from_monthly_frame(frame)

    assert result["headline"]["model_name"] == service.model_name
    assert result["headline"]["predicted_yield"] > 0
    assert len(result["prediction_rows"]) == 3
    assert len(result["feature_groups"]) >= 1
    assert result["monthly_features"][-1]["predicted_yield"] > 0
    assert result["drivers"][0]["label"]
