"""Synthetic multistage demo for MobileNet + weather + USDA yield labels."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from crop_fusion_ai.models.multistage_yield_model import MultiStageYieldModel

DEFAULT_MODEL_PATH = Path("models/yield_model/multistage_yield_regressor.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics/multistage_yield_metrics.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the multistage demo."""
    parser = argparse.ArgumentParser(
        description="Train a synthetic multistage yield model demo."
    )
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--image-feature-count", type=int, default=16)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args(argv)


def create_synthetic_multistage_dataframe(
    sample_count: int = 48,
    image_feature_count: int = 16,
) -> pd.DataFrame:
    """Create synthetic rows shaped like MobileNet + weather + USDA yield data."""
    if sample_count < 8:
        msg = "Synthetic multistage dataset requires at least 8 samples"
        raise ValueError(msg)
    if image_feature_count < 1:
        msg = "image_feature_count must be at least 1"
        raise ValueError(msg)

    rows: list[dict[str, float | int | str]] = []
    crops = ["corn", "soybean", "winter wheat"]
    regions = ["10003", "22007", "01003", "17019"]
    for index in range(sample_count):
        crop_type = crops[index % len(crops)]
        region = regions[index % len(regions)]
        row: dict[str, float | int | str] = {
            "crop_type": crop_type,
            "region": region,
            "year": 2017 + (index % 6),
            "weather_steps": 120.0,
            "temperature_mean": 18.0 + float((index * 5) % 16),
            "rainfall_sum": 250.0 + float((index * 17) % 300),
            "humidity_mean": 50.0 + float((index * 3) % 35),
        }
        image_signal = 0.0
        for feature_index in range(image_feature_count):
            value = float(((index + 1) * (feature_index + 3)) % 29) / 29.0
            row[f"image_feature_{feature_index:03d}"] = value
            image_signal += value

        crop_bonus = {"corn": 1.2, "soybean": 0.8, "winter wheat": 0.5}[crop_type]
        row["yield"] = (
            2.0
            + crop_bonus
            + 0.004 * float(row["rainfall_sum"])
            - 0.03 * abs(float(row["temperature_mean"]) - 24.0)
            + 0.08 * image_signal
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Train and save a synthetic multistage yield model."""
    args = parse_args(argv)
    dataframe = create_synthetic_multistage_dataframe(
        sample_count=args.samples,
        image_feature_count=args.image_feature_count,
    )
    model = MultiStageYieldModel()
    metrics = model.train_from_dataframe(dataframe, "yield")
    model.save(args.model_path)

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    print(json.dumps(metrics, indent=2))
    print(f"Saved multistage model to {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
