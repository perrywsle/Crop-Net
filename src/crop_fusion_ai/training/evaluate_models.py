"""Evaluation utilities for final reports."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from crop_fusion_ai.models.yield_regressor import (
    YieldRegressor,
)
from crop_fusion_ai.training.data_splits import build_year_based_split
from crop_fusion_ai.training.train_yield_model import create_synthetic_yield_dataframe
from crop_fusion_ai.utils.plotting import (
    save_error_distribution_plot,
    save_predicted_vs_actual_plot,
)

DEFAULT_METRICS_PATH = Path("reports/metrics/yield_metrics.json")
DEFAULT_PREDICTED_VS_ACTUAL_PATH = Path("reports/figures/predicted_vs_actual.png")
DEFAULT_ERROR_DISTRIBUTION_PATH = Path("reports/figures/error_distribution.png")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate yield and image model outputs for reports."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/cropnet_features.csv"),
        help="CSV containing yield features and target column.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="yield",
        help="Target yield column in the CSV.",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=3,
        help="Required number of preceding years for each target year.",
    )
    return parser.parse_args(argv)


def evaluate_yield_model_on_csv(
    csv_path: Path,
    target_column: str,
    *,
    history_years: int = 3,
) -> dict[str, float]:
    """Evaluate the baseline yield model on CSV data and write report outputs."""
    if csv_path.exists():
        dataframe = pd.read_csv(csv_path)
    else:
        dataframe = create_synthetic_yield_dataframe(96)

    split = build_year_based_split(dataframe, history_years=history_years)
    actual_values, predicted_values = _fit_and_predict_holdout(
        split.train,
        split.validation,
        split.test,
        target_column,
    )
    errors = [
        predicted_value - actual_value
        for actual_value, predicted_value in zip(
            actual_values,
            predicted_values,
            strict=True,
        )
    ]
    metrics = {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
    }

    _save_yield_metrics(DEFAULT_METRICS_PATH, metrics)
    generate_predicted_vs_actual_plot(
        actual_values,
        predicted_values,
        DEFAULT_PREDICTED_VS_ACTUAL_PATH,
    )
    generate_error_distribution_plot(errors, DEFAULT_ERROR_DISTRIBUTION_PATH)
    return metrics


def generate_predicted_vs_actual_plot(
    actual_values: list[float],
    predicted_values: list[float],
    output_path: Path = DEFAULT_PREDICTED_VS_ACTUAL_PATH,
) -> Path:
    """Generate the yield predicted-vs-actual report figure."""
    if len(actual_values) != len(predicted_values):
        msg = "actual_values and predicted_values must have the same length"
        raise ValueError(msg)
    if not actual_values:
        msg = "At least one value is required to generate a plot"
        raise ValueError(msg)
    return save_predicted_vs_actual_plot(actual_values, predicted_values, output_path)


def generate_error_distribution_plot(
    errors: list[float],
    output_path: Path = DEFAULT_ERROR_DISTRIBUTION_PATH,
) -> Path:
    """Generate the yield prediction error distribution report figure."""
    if not errors:
        msg = "At least one error value is required to generate a plot"
        raise ValueError(msg)
    return save_error_distribution_plot(errors, output_path)


def evaluate_image_folder_dataset(dataset_path: Path) -> dict[str, float]:
    """Placeholder image model evaluation for folder-based datasets.

    TODO: Replace this with real image-model evaluation once the trained
    MobileNet/EfficientNet segmentation or classification model is available.
    TODO: Iterate class folders, run model inference, and compute accuracy,
    macro-F1, confusion matrix, and per-class metrics.
    """
    if not dataset_path.exists():
        msg = f"Image dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)
    if not dataset_path.is_dir():
        msg = f"Image dataset path is not a directory: {dataset_path}"
        raise NotADirectoryError(msg)

    image_count = sum(1 for path in dataset_path.rglob("*") if path.is_file())
    return {
        "image_count": float(image_count),
        "accuracy": 0.0,
        "macro_f1": 0.0,
    }


def _fit_and_predict_holdout(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> tuple[list[float], list[float]]:
    """Train on train+validation years and predict the explicit test year."""
    regressor = YieldRegressor()
    regressor.fit_dataframe(
        pd.concat([train_df, validation_df], ignore_index=True),
        target_column,
    )
    prediction_df = test_df.dropna(subset=[target_column]).copy()
    if prediction_df.empty:
        msg = f"No rows remain after dropping missing target '{target_column}'"
        raise ValueError(msg)
    if regressor.pipeline is None:
        msg = "Yield regressor pipeline is unavailable after fitting"
        raise RuntimeError(msg)
    features = prediction_df.drop(columns=[target_column])
    features = features.reindex(columns=regressor.feature_columns)
    predictions = regressor.pipeline.predict(features)

    return (
        [
            float(value)
            for value in prediction_df[target_column].astype(float).to_list()
        ],
        [float(value) for value in predictions.tolist()],
    )


def _save_yield_metrics(metrics_path: Path, metrics: dict[str, float]) -> None:
    """Save yield metrics JSON for reports."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run yield evaluation and save report artifacts."""
    args = parse_args(argv)
    metrics = evaluate_yield_model_on_csv(
        args.csv,
        args.target,
        history_years=args.history_years,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
