"""Tests for the optional CropNet API wrapper."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from crop_fusion_ai.config.schemas import ImagePrediction, WeatherFeatures
from crop_fusion_ai.data_sources import (
    CropNetClient,
    CropNetDependencyError,
    CropNetQuery,
    CropNetSample,
    cropnet_sample_to_yield_input,
)


class FakeDownloader:
    """Small fake for the official CropNet DataDownloader API."""

    def __init__(self, *, target_dir: str) -> None:
        """Record the target directory used to construct the fake."""
        self.target_dir = target_dir
        self.calls: list[tuple[str, object]] = []

    def download_USDA(
        self,
        crop_type: str,
        *,
        fips_codes: list[str],
        years: list[str],
    ) -> object:
        """Record a USDA download request."""
        self.calls.append(
            (
                "USDA",
                {
                    "crop_type": crop_type,
                    "fips_codes": fips_codes,
                    "years": years,
                },
            )
        )
        return None

    def download_HRRR(self, *, fips_codes: list[str], years: list[str]) -> object:
        """Record an HRRR download request."""
        self.calls.append(("HRRR", {"fips_codes": fips_codes, "years": years}))
        return None

    def download_Sentinel2(
        self,
        *,
        fips_codes: list[str],
        years: list[str],
        image_type: str,
    ) -> object:
        """Record a Sentinel-2 download request."""
        self.calls.append(
            (
                "Sentinel2",
                {
                    "fips_codes": fips_codes,
                    "years": years,
                    "image_type": image_type,
                },
            )
        )
        return None


def test_cropnet_query_validates_fips_and_dataset_years() -> None:
    """CropNet queries should stay bounded to documented years and FIPS codes."""
    with pytest.raises(ValidationError):
        CropNetQuery(crop_type="Corn", fips_codes=["1003"], years=[2022])

    with pytest.raises(ValidationError):
        CropNetQuery(crop_type="Corn", fips_codes=["01003"], years=[2026])


def test_cropnet_client_uses_injected_downloader_for_selected_modalities() -> None:
    """The client should call only the selected official API modalities."""
    fake_downloader = FakeDownloader(target_dir="unused")

    def factory(*, target_dir: str) -> FakeDownloader:
        assert target_dir == "data/cropnet_cache"
        return fake_downloader

    client = CropNetClient(downloader_factory=factory)
    query = CropNetQuery(
        crop_type="Soybean",
        fips_codes=["10003", "22007"],
        years=[2022],
        image_type="NDVI",
        include_sentinel2=True,
    )

    result = client.download_query(query)

    assert result.target_dir == Path("data/cropnet_cache")
    assert result.requested_modalities == ["USDA", "HRRR", "Sentinel2:NDVI"]
    assert fake_downloader.calls == [
        (
            "USDA",
            {
                "crop_type": "Soybean",
                "fips_codes": ["10003", "22007"],
                "years": ["2022"],
            },
        ),
        ("HRRR", {"fips_codes": ["10003", "22007"], "years": ["2022"]}),
        (
            "Sentinel2",
            {
                "fips_codes": ["10003", "22007"],
                "years": ["2022"],
                "image_type": "NDVI",
            },
        ),
    ]


def test_cropnet_client_raises_clear_error_when_dependency_missing() -> None:
    """Real API access should fail with a setup-focused message if missing."""

    def missing_import(_: str) -> object:
        raise ImportError("missing cropnet")

    client = CropNetClient(import_module_func=missing_import)
    query = CropNetQuery(crop_type="Corn", fips_codes=["01003"], years=[2022])

    with pytest.raises(CropNetDependencyError, match="optional 'cropnet' package"):
        client.download_query(query)


def test_cropnet_sample_to_yield_input_combines_sample_and_image_prediction() -> None:
    """Normalized CropNet samples should convert into existing fusion schemas."""
    sample = CropNetSample(
        fips_code="01003",
        year=2022,
        crop_type="Corn",
        yield_value=142.5,
        weather=WeatherFeatures(
            temperature_mean=27.4,
            rainfall_total=155.0,
            humidity_mean=72.0,
        ),
        region="Baldwin County",
    )
    image_prediction = ImagePrediction(
        disease_class="healthy",
        health_score=0.96,
        confidence=0.9,
    )

    yield_input = cropnet_sample_to_yield_input(sample, image_prediction)

    assert yield_input.crop.crop_type == "Corn"
    assert yield_input.crop.region == "Baldwin County"
    assert yield_input.weather.rainfall_total == pytest.approx(155.0)
    assert yield_input.image_prediction.health_score == pytest.approx(0.96)
