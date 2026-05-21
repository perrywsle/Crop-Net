"""Tests for real-data multistage dataset building and training."""

from pathlib import Path

import pandas as pd

from crop_fusion_ai.data_sources.cropnet_client import CropNetDownloadResult
from crop_fusion_ai.data_sources.cropnet_schemas import CropNetQuery
from crop_fusion_ai.training.build_multistage_cropnet_dataset import (
    NextYearDatasetConfig,
    build_and_save_next_year_dataset,
    build_next_year_multistage_dataframe,
    download_bounded_cropnet_inputs,
)
from crop_fusion_ai.training.train_multistage_real import (
    train_multistage_model_from_csv,
)


class FakeExtractor:
    """Deterministic image feature extractor for tests."""

    def extract(self, image_path: Path) -> list[float]:
        """Return features based on the image filename."""
        offset = 1.0 if "b" in image_path.name else 0.0
        return [1.0 + offset, 3.0 + offset]


class FakeCropNetClient:
    """Record bounded CropNet queries without network access."""

    calls: list[CropNetQuery] = []

    def __init__(self, target_dir: Path) -> None:
        """Record the target dir and reset calls for each test construction."""
        self.target_dir = target_dir
        FakeCropNetClient.calls = []

    def download_query(self, query: CropNetQuery) -> CropNetDownloadResult:
        """Record a query and return a lightweight result."""
        FakeCropNetClient.calls.append(query)
        return CropNetDownloadResult(
            target_dir=self.target_dir,
            requested_modalities=[],
            fips_codes=query.fips_codes,
            years=query.years,
        )


def test_build_next_year_multistage_dataframe_uses_prior_features_and_next_yield(
    tmp_path: Path,
) -> None:
    """Dataset builder should combine 2021 NDVI/weather with 2022 yield."""
    weather_csv, yield_csv, image_manifest = _write_minimal_manifests(tmp_path)
    config = NextYearDatasetConfig(fips_codes=("01003",), raw_cache_dir=tmp_path)

    dataframe = build_next_year_multistage_dataframe(
        config=config,
        weather_csv=weather_csv,
        yield_csv=yield_csv,
        image_manifest=image_manifest,
        extractor=FakeExtractor(),
    )

    row = dataframe.iloc[0]
    assert row["crop_type"] == "corn"
    assert row["image_type"] == "NDVI"
    assert row["input_year"] == 2021
    assert row["target_year"] == 2022
    assert row["yield"] == 152.0
    assert row["weather_steps"] == 2.0
    assert row["temperature_mean"] == 21.0
    assert row["rainfall_sum"] == 5.0
    assert row["image_feature_000"] == 1.5
    assert row["image_feature_001"] == 3.5


def test_build_and_save_next_year_dataset_deletes_disposable_raw_cache(
    tmp_path: Path,
) -> None:
    """Raw cache should be disposable once compact features are persisted."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "raw.bin").write_bytes(b"raw")
    weather_csv, yield_csv, image_manifest = _write_minimal_manifests(tmp_path)

    result = build_and_save_next_year_dataset(
        config=NextYearDatasetConfig(fips_codes=("01003",), raw_cache_dir=cache_dir),
        weather_csv=weather_csv,
        yield_csv=yield_csv,
        image_manifest=image_manifest,
        output_path=tmp_path / "features.csv",
        metadata_path=tmp_path / "metadata.json",
        extractor=FakeExtractor(),
        keep_raw_cache=False,
    )

    assert result.row_count == 1
    assert result.output_path.exists()
    assert result.metadata_path.exists()
    assert not cache_dir.exists()


def test_download_bounded_cropnet_inputs_separates_input_and_target_modalities(
    tmp_path: Path,
) -> None:
    """Only input-year HRRR/NDVI and target-year USDA should be requested."""
    config = NextYearDatasetConfig(
        crop_type="corn",
        image_type="NDVI",
        input_year=2021,
        target_year=2022,
        fips_codes=("01003", "01005"),
        raw_cache_dir=tmp_path,
    )

    download_bounded_cropnet_inputs(config, client_factory=FakeCropNetClient)

    assert len(FakeCropNetClient.calls) == 2
    input_query, target_query = FakeCropNetClient.calls
    assert input_query.years == [2021]
    assert input_query.include_hrrr is True
    assert input_query.include_sentinel2 is True
    assert input_query.include_usda is False
    assert target_query.years == [2022]
    assert target_query.include_usda is True
    assert target_query.include_hrrr is False
    assert target_query.include_sentinel2 is False


def test_train_multistage_model_from_csv_saves_artifacts(tmp_path: Path) -> None:
    """Real trainer should consume processed CSV and persist artifacts."""
    csv_path = tmp_path / "features.csv"
    _write_training_csv(csv_path)
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_multistage_model_from_csv(
        csv_path=csv_path,
        target_column="yield",
        model_path=model_path,
        metrics_path=metrics_path,
        holdout_fraction=0.25,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["row_count"] == 8
    assert metrics["holdout_rows"] == 2
    assert "mae" in metrics


def _write_minimal_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create small manifests for one FIPS county."""
    weather_csv = tmp_path / "weather.csv"
    yield_csv = tmp_path / "yield.csv"
    image_manifest = tmp_path / "images.csv"
    pd.DataFrame(
        [
            {"fips_code": "01003", "year": 2021, "temperature": 20.0, "rainfall": 2.0},
            {"fips_code": "01003", "year": 2021, "temperature": 22.0, "rainfall": 3.0},
        ]
    ).to_csv(weather_csv, index=False)
    pd.DataFrame(
        [
            {
                "fips_code": "01003",
                "year": 2022,
                "yield": 152.0,
                "yield_unit": "bushels_per_acre",
            }
        ]
    ).to_csv(yield_csv, index=False)
    pd.DataFrame(
        [
            {
                "fips_code": "01003",
                "year": 2021,
                "image_type": "NDVI",
                "image_path": str(tmp_path / "tile_a.png"),
            },
            {
                "fips_code": "01003",
                "year": 2021,
                "image_type": "NDVI",
                "image_path": str(tmp_path / "tile_b.png"),
            },
        ]
    ).to_csv(image_manifest, index=False)
    return weather_csv, yield_csv, image_manifest


def _write_training_csv(csv_path: Path) -> None:
    """Create a small processed multistage dataset with multiple regions."""
    rows: list[dict[str, float | int | str]] = []
    for index in range(8):
        rows.append(
            {
                "crop_type": "corn",
                "region": f"0100{index}",
                "input_year": 2021,
                "target_year": 2022,
                "image_type": "NDVI",
                "yield_unit": "bushels_per_acre",
                "weather_steps": 2.0,
                "temperature_mean": 20.0 + float(index),
                "rainfall_sum": 5.0 + float(index),
                "image_feature_000": float(index) / 10.0,
                "image_feature_001": float(index) / 20.0,
                "yield": 120.0 + float(index) * 3.0,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
