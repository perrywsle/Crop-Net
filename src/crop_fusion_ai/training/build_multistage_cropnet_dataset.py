"""Build a compact next-year CropNet-style multistage training dataset."""

import argparse
import json
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from crop_fusion_ai.data.weather_timeseries import summarize_weather_sequence
from crop_fusion_ai.data_sources.cropnet_client import CropNetClient
from crop_fusion_ai.data_sources.cropnet_schemas import CropNetImageType, CropNetQuery
from crop_fusion_ai.models.mobilenet_feature_extractor import MobileNetFeatureExtractor

DEFAULT_OUTPUT_PATH = Path("data/processed/multistage_cropnet_features.csv")
DEFAULT_CACHE_DIR = Path("data/cropnet_cache")
DEFAULT_METADATA_PATH = Path("reports/metrics/multistage_dataset_metadata.json")
BYTES_PER_GIB = 1024**3


class ImageFeatureExtractor(Protocol):
    """Small interface shared by MobileNet and test doubles."""

    def extract(self, image_path: Path) -> list[float]:
        """Extract one feature vector from an image."""
        ...


@dataclass(frozen=True)
class NextYearDatasetConfig:
    """Configuration for a bounded one-year-ahead crop experiment."""

    crop_type: str = "corn"
    image_type: CropNetImageType = "NDVI"
    input_year: int = 2021
    target_year: int = 2022
    fips_codes: tuple[str, ...] = ("01003",)
    raw_cache_dir: Path = DEFAULT_CACHE_DIR
    max_cache_gb: float = 3.0


@dataclass(frozen=True)
class DatasetBuildResult:
    """Summary of a processed dataset build."""

    output_path: Path
    metadata_path: Path
    row_count: int
    feature_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the dataset builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact corn/NDVI next-year dataset from bounded CropNet "
            "downloads and prepared local manifests."
        )
    )
    parser.add_argument("--crop-type", default="corn")
    parser.add_argument("--image-type", choices=["AG", "NDVI"], default="NDVI")
    parser.add_argument("--input-year", type=int, default=2021)
    parser.add_argument("--target-year", type=int, default=2022)
    parser.add_argument(
        "--fips",
        nargs="+",
        required=True,
        help="Five-digit FIPS county codes to include in the bounded request.",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        required=True,
        help=(
            "CSV with fips_code, year, and weather rows containing temperature, "
            "rainfall, humidity, and/or solar_radiation columns."
        ),
    )
    parser.add_argument(
        "--yield-csv",
        type=Path,
        required=True,
        help="CSV with fips_code, year, yield, and optional yield_unit columns.",
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        required=True,
        help="CSV with fips_code, year, image_path, and optional image_type columns.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-cache-gb", type=float, default=3.0)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run bounded official CropNet download calls before reading manifests.",
    )
    parser.add_argument(
        "--keep-raw-cache",
        action="store_true",
        help="Keep raw CropNet cache after processed features are written.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained TorchVision MobileNet weights for image embeddings.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for MobileNet feature extraction, e.g. 'cpu' or 'cuda'.",
    )
    return parser.parse_args(argv)


def download_bounded_cropnet_inputs(
    config: NextYearDatasetConfig,
    *,
    client_factory: Callable[[Path], CropNetClient] = CropNetClient,
) -> None:
    """Download only the modalities needed for the one-year-ahead experiment."""
    client = client_factory(config.raw_cache_dir)
    input_year_query = CropNetQuery(
        crop_type=config.crop_type,
        fips_codes=list(config.fips_codes),
        years=[config.input_year],
        image_type=config.image_type,
        include_usda=False,
        include_hrrr=True,
        include_sentinel2=True,
    )
    client.download_query(input_year_query)
    _raise_if_cache_too_large(config.raw_cache_dir, config.max_cache_gb)

    target_year_query = CropNetQuery(
        crop_type=config.crop_type,
        fips_codes=list(config.fips_codes),
        years=[config.target_year],
        image_type=config.image_type,
        include_usda=True,
        include_hrrr=False,
        include_sentinel2=False,
    )
    client.download_query(target_year_query)
    _raise_if_cache_too_large(config.raw_cache_dir, config.max_cache_gb)


def build_next_year_multistage_dataframe(
    *,
    config: NextYearDatasetConfig,
    weather_csv: Path,
    yield_csv: Path,
    image_manifest: Path,
    extractor: ImageFeatureExtractor,
) -> pd.DataFrame:
    """Build one row per FIPS with image, weather, crop, and next-year yield."""
    weather_df = _read_csv_required(weather_csv, "weather")
    yield_df = _read_csv_required(yield_csv, "yield")
    image_df = _read_csv_required(image_manifest, "image manifest")
    rows: list[dict[str, float | int | str]] = []

    for fips_code in config.fips_codes:
        weather_features = _summarize_weather_for_fips(
            weather_df,
            fips_code=fips_code,
            input_year=config.input_year,
        )
        yield_value, yield_unit = _yield_for_fips(
            yield_df,
            fips_code=fips_code,
            target_year=config.target_year,
        )
        image_features = _average_image_features_for_fips(
            image_df,
            fips_code=fips_code,
            input_year=config.input_year,
            image_type=config.image_type,
            extractor=extractor,
        )
        row: dict[str, float | int | str] = {
            "crop_type": config.crop_type,
            "region": fips_code,
            "input_year": config.input_year,
            "target_year": config.target_year,
            "image_type": config.image_type,
            "yield_unit": yield_unit,
            "yield": yield_value,
        }
        row.update(weather_features)
        for index, value in enumerate(image_features):
            row[f"image_feature_{index:03d}"] = value
        rows.append(row)

    if not rows:
        msg = "No dataset rows were produced for the selected FIPS codes"
        raise ValueError(msg)
    return pd.DataFrame(rows)


def build_and_save_next_year_dataset(
    *,
    config: NextYearDatasetConfig,
    weather_csv: Path,
    yield_csv: Path,
    image_manifest: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    extractor: ImageFeatureExtractor,
    keep_raw_cache: bool = False,
) -> DatasetBuildResult:
    """Build the processed CSV, write metadata, and optionally delete raw cache."""
    dataframe = build_next_year_multistage_dataframe(
        config=config,
        weather_csv=weather_csv,
        yield_csv=yield_csv,
        image_manifest=image_manifest,
        extractor=extractor,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    image_feature_count = len(
        [column for column in dataframe.columns if column.startswith("image_feature_")]
    )
    _write_metadata(
        metadata_path,
        config=config,
        output_path=output_path,
        row_count=len(dataframe),
        image_feature_count=image_feature_count,
    )

    if not keep_raw_cache:
        _delete_raw_cache(config.raw_cache_dir)

    return DatasetBuildResult(
        output_path=output_path,
        metadata_path=metadata_path,
        row_count=len(dataframe),
        feature_count=len(dataframe.columns) - 1,
    )


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    """Read a required CSV with an actionable error."""
    if not path.exists():
        msg = f"{label.title()} CSV does not exist: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def _summarize_weather_for_fips(
    weather_df: pd.DataFrame,
    *,
    fips_code: str,
    input_year: int,
) -> dict[str, float]:
    """Filter weather records for one county/year and summarize them."""
    _require_columns(weather_df, {"fips_code", "year"}, "weather CSV")
    rows = cast(
        pd.DataFrame,
        weather_df[
            (weather_df["fips_code"].astype(str).str.zfill(5) == fips_code)
            & (weather_df["year"].astype(int) == input_year)
        ],
    )
    if rows.empty:
        msg = f"No weather rows found for FIPS {fips_code} in {input_year}"
        raise ValueError(msg)

    records: list[dict[str, float]] = []
    weather_keys = ("temperature", "rainfall", "humidity", "solar_radiation")
    for row in _dataframe_records(rows):
        record: dict[str, float] = {}
        for key in weather_keys:
            if key in row:
                value = _finite_float_or_none(row[key])
                if value is not None:
                    record[key] = value
        records.append(record)
    return summarize_weather_sequence(records)


def _yield_for_fips(
    yield_df: pd.DataFrame,
    *,
    fips_code: str,
    target_year: int,
) -> tuple[float, str]:
    """Return the next-year yield label for one county."""
    _require_columns(yield_df, {"fips_code", "year", "yield"}, "yield CSV")
    rows = cast(
        pd.DataFrame,
        yield_df[
            (yield_df["fips_code"].astype(str).str.zfill(5) == fips_code)
            & (yield_df["year"].astype(int) == target_year)
        ],
    )
    if rows.empty:
        msg = f"No yield row found for FIPS {fips_code} in {target_year}"
        raise ValueError(msg)
    first_row = _dataframe_records(rows)[0]
    unit = "bushels_per_acre"
    if "yield_unit" in rows.columns:
        unit_value = first_row.get("yield_unit")
        if unit_value is not None and str(unit_value).lower() != "nan":
            unit = str(unit_value)
    yield_value = _finite_float_or_none(first_row["yield"])
    if yield_value is None:
        msg = f"Yield value for FIPS {fips_code} in {target_year} is not numeric"
        raise ValueError(msg)
    return yield_value, unit


def _average_image_features_for_fips(
    image_df: pd.DataFrame,
    *,
    fips_code: str,
    input_year: int,
    image_type: CropNetImageType,
    extractor: ImageFeatureExtractor,
) -> list[float]:
    """Extract and average embeddings for one county/year/image type."""
    _require_columns(image_df, {"fips_code", "year", "image_path"}, "image manifest")
    mask = (image_df["fips_code"].astype(str).str.zfill(5) == fips_code) & (
        image_df["year"].astype(int) == input_year
    )
    if "image_type" in image_df.columns:
        mask = mask & (image_df["image_type"].astype(str).str.upper() == image_type)
    rows = image_df[mask]
    if rows.empty:
        msg = f"No {image_type} image rows found for FIPS {fips_code} in {input_year}"
        raise ValueError(msg)

    feature_vectors: list[list[float]] = []
    for image_path_value in list(rows["image_path"].astype(str)):
        feature_vectors.append(extractor.extract(Path(image_path_value)))
    feature_count = len(feature_vectors[0])
    if any(len(vector) != feature_count for vector in feature_vectors):
        msg = f"Inconsistent MobileNet feature lengths for FIPS {fips_code}"
        raise ValueError(msg)
    return [
        sum(vector[index] for vector in feature_vectors) / len(feature_vectors)
        for index in range(feature_count)
    ]


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> None:
    """Fail when a manifest does not contain required columns."""
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        msg = f"{label} is missing required columns: {missing_columns}"
        raise ValueError(msg)


def _finite_float_or_none(value: object) -> float | None:
    """Convert scalar values to finite floats while skipping blanks/NaN."""
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value):
        return None
    return numeric_value


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a dataframe to row dictionaries without relying on pandas typing."""
    columns = [str(column) for column in dataframe.columns]
    return [
        dict(zip(columns, values, strict=True))
        for values in dataframe.itertuples(index=False, name=None)
    ]


def _directory_size_bytes(path: Path) -> int:
    """Return the total size of files under a directory."""
    if not path.exists():
        return 0
    return sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )


def _raise_if_cache_too_large(cache_dir: Path, max_cache_gb: float) -> None:
    """Protect the feasibility workflow from accidental large downloads."""
    size_bytes = _directory_size_bytes(cache_dir)
    max_bytes = int(max_cache_gb * BYTES_PER_GIB)
    if size_bytes > max_bytes:
        size_gib = size_bytes / BYTES_PER_GIB
        msg = (
            f"Raw CropNet cache at {cache_dir} is {size_gib:.2f} GiB, "
            f"above the configured {max_cache_gb:.2f} GiB limit."
        )
        raise RuntimeError(msg)


def _delete_raw_cache(cache_dir: Path) -> None:
    """Delete disposable raw cache after compact features have been saved."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _write_metadata(
    metadata_path: Path,
    *,
    config: NextYearDatasetConfig,
    output_path: Path,
    row_count: int,
    image_feature_count: int,
) -> None:
    """Write dataset-build metadata for reproducibility."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "crop_type": config.crop_type,
        "image_type": config.image_type,
        "input_year": config.input_year,
        "target_year": config.target_year,
        "fips_codes": list(config.fips_codes),
        "raw_cache_dir": str(config.raw_cache_dir),
        "max_cache_gb": config.max_cache_gb,
        "output_path": str(output_path),
        "row_count": row_count,
        "image_feature_count": image_feature_count,
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Build a compact multistage dataset from manifests and optional downloads."""
    args = parse_args(argv)
    image_type: CropNetImageType = "NDVI" if args.image_type == "NDVI" else "AG"
    config = NextYearDatasetConfig(
        crop_type=args.crop_type,
        image_type=image_type,
        input_year=args.input_year,
        target_year=args.target_year,
        fips_codes=tuple(args.fips),
        raw_cache_dir=args.raw_cache_dir,
        max_cache_gb=args.max_cache_gb,
    )
    if args.download:
        download_bounded_cropnet_inputs(config)
    extractor = MobileNetFeatureExtractor(
        use_pretrained_weights=args.pretrained,
        device=args.device,
    )
    result = build_and_save_next_year_dataset(
        config=config,
        weather_csv=args.weather_csv,
        yield_csv=args.yield_csv,
        image_manifest=args.image_manifest,
        output_path=args.output,
        metadata_path=args.metadata_path,
        extractor=extractor,
        keep_raw_cache=args.keep_raw_cache,
    )
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "metadata_path": str(result.metadata_path),
                "row_count": result.row_count,
                "feature_count": result.feature_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
