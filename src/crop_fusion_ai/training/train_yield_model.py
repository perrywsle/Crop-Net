"""CLI training entrypoint for the baseline CropNet-style yield regressor."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from crop_fusion_ai.models.yield_regressor import YieldRegressor
from crop_fusion_ai.training.data_splits import build_year_based_split

DEFAULT_MODEL_PATH = Path("models/yield_model/yield_regressor.joblib")
DEFAULT_METRICS_PATH = Path("reports/metrics/yield_metrics.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for yield model training."""
    parser = argparse.ArgumentParser(
        description="Train the baseline CropNet-style yield regressor."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/cropnet_features.csv"),
        help="CSV containing feature columns and target yield column.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="yield",
        help="Target column name in the CSV or synthetic dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path where the trained joblib model will be saved.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path where JSON metrics will be saved.",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=96,
        help="Number of synthetic rows to create when CSV is missing.",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=3,
        help="Required number of preceding years for each target year.",
    )
    return parser.parse_args(argv)


def load_or_create_training_dataframe(
    csv_path: Path,
    *,
    synthetic_samples: int,
) -> tuple[pd.DataFrame, str]:
    """Load a CSV if present, otherwise create a small synthetic demo dataset."""
    if csv_path.exists():
        return pd.read_csv(csv_path), "csv"
    return create_synthetic_yield_dataframe(synthetic_samples), "synthetic"


def create_synthetic_yield_dataframe(sample_count: int = 48) -> pd.DataFrame:
    """Create a deterministic synthetic dataset for demos and tests."""
    if sample_count < 8:
        msg = "Synthetic yield dataset requires at least 8 samples"
        raise ValueError(msg)

    crop_types = ["corn", "soybean", "wheat"]
    regions = ["01003", "17019", "22007", "31079"]
    disease_classes = ["healthy", "mild_disease", "severe_disease"]
    rows: list[dict[str, float | int | str]] = []

    for index in range(sample_count):
        crop_type = crop_types[index % len(crop_types)]
        region = regions[index % len(regions)]
        disease_class = disease_classes[index % len(disease_classes)]
        rainfall_total = 80.0 + float((index * 11) % 140)
        temperature_mean = 18.0 + float((index * 3) % 17)
        health_score = {"healthy": 0.9, "mild_disease": 0.62}.get(
            disease_class,
            0.35,
        )
        image_confidence = 0.55 + float(index % 5) * 0.08
        crop_bonus = {"corn": 1.4, "soybean": 0.8, "wheat": 0.5}[crop_type]
        yield_value = (
            2.0
            + crop_bonus
            + 0.012 * rainfall_total
            - 0.035 * abs(temperature_mean - 25.0)
            + 1.8 * health_score
            + 0.15 * image_confidence
        )
        rows.append(
            {
                "temperature_mean": temperature_mean,
                "rainfall_total": rainfall_total,
                "humidity_mean": 55.0 + float((index * 7) % 35),
                "crop_type": crop_type,
                "region": region,
                "year": 2014 + (index % 8),
                "disease_class": disease_class,
                "health_score": health_score,
                "image_confidence": image_confidence,
                "yield": yield_value,
            }
        )

    return pd.DataFrame(rows)


def save_metrics(
    metrics_path: Path,
    metrics: dict[str, float],
    *,
    data_source: str,
    row_count: int,
    split_row_counts: dict[str, int],
    history_years: int,
) -> None:
    """Save training metrics and data-source metadata as JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_source": data_source,
        "row_count": row_count,
        "split_row_counts": split_row_counts,
        "history_years": history_years,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Train and save the baseline yield regressor."""
    args = parse_args(argv)
    df, data_source = load_or_create_training_dataframe(
        args.csv,
        synthetic_samples=args.synthetic_samples,
    )
    split = build_year_based_split(df, history_years=args.history_years)
    regressor = YieldRegressor()
    regressor.fit_dataframe(split.train, args.target)
    metrics = regressor.evaluate_dataframe(split.validation, args.target)
    regressor.fit_dataframe(
        pd.concat([split.train, split.validation], ignore_index=True),
        args.target,
    )
    regressor.save(args.model_path)
    save_metrics(
        args.metrics_path,
        metrics,
        data_source=data_source,
        row_count=len(df),
        split_row_counts={
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        history_years=args.history_years,
    )

    print(f"Trained yield model using {data_source} data with {len(df)} rows.")
    print(
        "Validation split: "
        f"train={len(split.train)}, validation={len(split.validation)}, "
        f"held-out test={len(split.test)}"
    )
    print(f"Saved model to {args.model_path}")
    print(f"Saved metrics to {args.metrics_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
