"""Train the real-data multistage MobileNet/weather yield model."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from crop_fusion_ai.models.multistage_yield_model import MultiStageYieldModel
from crop_fusion_ai.training.build_multistage_cropnet_dataset import DEFAULT_OUTPUT_PATH
from crop_fusion_ai.training.train_multistage_demo import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for real multistage training."""
    parser = argparse.ArgumentParser(
        description="Train MobileNet/weather multistage yield model from processed CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target", type=str, default="yield")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.25,
        help="Fraction of counties/rows reserved for final evaluation.",
    )
    return parser.parse_args(argv)


def train_multistage_model_from_csv(
    *,
    csv_path: Path,
    target_column: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    holdout_fraction: float = 0.25,
) -> dict[str, float | int | str | None]:
    """Train from a processed next-year CSV and save model plus metrics."""
    if not csv_path.exists():
        msg = f"Processed multistage CSV does not exist: {csv_path}"
        raise FileNotFoundError(msg)
    if holdout_fraction <= 0.0 or holdout_fraction >= 1.0:
        msg = "holdout_fraction must be between 0 and 1"
        raise ValueError(msg)

    dataframe = pd.read_csv(csv_path)
    if target_column not in dataframe.columns:
        msg = f"Target column '{target_column}' is missing from {csv_path}"
        raise ValueError(msg)
    if len(dataframe) < 2:
        msg = (
            "At least 2 rows/counties are required to train the real "
            "multistage model"
        )
        raise ValueError(msg)

    train_df, holdout_df = _county_holdout_split(
        dataframe,
        holdout_fraction=holdout_fraction,
    )
    model = MultiStageYieldModel()
    model.fit_dataframe(train_df, target_column)
    metrics = _evaluate_model(model, holdout_df, target_column)
    final_model = MultiStageYieldModel()
    final_model.fit_dataframe(dataframe, target_column)
    final_model.save(model_path)

    payload: dict[str, float | int | str | None] = {
        "csv_path": str(csv_path),
        "target_column": target_column,
        "row_count": len(dataframe),
        "train_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "model_path": str(model_path),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    }
    _save_metrics(metrics_path, payload)
    return payload


def _county_holdout_split(
    dataframe: pd.DataFrame,
    *,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by region/county when available to reduce tile-level leakage."""
    split_column = "region" if "region" in dataframe.columns else None
    if split_column is None:
        shuffled = dataframe.sample(frac=1.0, random_state=42).reset_index(drop=True)
        holdout_count = max(1, round(len(shuffled) * holdout_fraction))
        train_df = cast(pd.DataFrame, shuffled.iloc[holdout_count:].copy())
        holdout_df = cast(pd.DataFrame, shuffled.iloc[:holdout_count].copy())
        return train_df, holdout_df

    regions = sorted(str(value) for value in dataframe[split_column].dropna().unique())
    if len(regions) < 2:
        msg = "At least two regions are required for a region-based holdout split"
        raise ValueError(msg)
    holdout_count = max(1, round(len(regions) * holdout_fraction))
    holdout_regions = regions[:holdout_count]
    holdout_mask = dataframe[split_column].astype(str).isin(holdout_regions)
    train_df = cast(pd.DataFrame, dataframe.loc[~holdout_mask].copy())
    holdout_df = cast(pd.DataFrame, dataframe.loc[holdout_mask].copy())
    if train_df.empty or holdout_df.empty:
        msg = "Holdout split produced an empty train or evaluation partition"
        raise ValueError(msg)
    return train_df, holdout_df


def _evaluate_model(
    model: MultiStageYieldModel,
    holdout_df: pd.DataFrame,
    target_column: str,
) -> dict[str, float | None]:
    """Evaluate a trained multistage model on held-out counties/rows."""
    if model.pipeline is None:
        msg = "Multistage model pipeline is unavailable after training"
        raise RuntimeError(msg)
    eval_df = holdout_df.dropna(subset=[target_column]).copy()
    if eval_df.empty:
        msg = f"No rows remain after dropping missing target '{target_column}'"
        raise ValueError(msg)
    features = eval_df.drop(columns=[target_column]).reindex(
        columns=model.feature_columns
    )
    actual = eval_df[target_column].astype(float)
    predictions = model.pipeline.predict(features)
    r2_value = None
    if len(actual) >= 2:
        r2_value = float(r2_score(actual, predictions))
    return {
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(mean_squared_error(actual, predictions) ** 0.5),
        "r2": r2_value,
    }


def _save_metrics(
    metrics_path: Path,
    payload: dict[str, float | int | str | None],
) -> None:
    """Persist training metrics and split metadata."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run real-data multistage training from the processed CSV."""
    args = parse_args(argv)
    metrics = train_multistage_model_from_csv(
        csv_path=args.csv,
        target_column=args.target,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        holdout_fraction=args.holdout_fraction,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
