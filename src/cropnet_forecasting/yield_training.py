from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import selected_feature_columns
from .yield_regression import (
    baseline_prediction_columns,
    build_month_benchmark,
    build_monthly_training_frame,
    build_model_candidate_pipelines,
    build_model_pipelines,
    build_prediction_frame,
    build_residual_frame,
    build_window_benchmark,
    discover_usda_candidates,
    get_feature_importance,
    infer_crop_type_from_path,
    load_usda_yield_tables,
    normalize_crop_type,
    prune_feature_columns,
    resolve_usda_paths,
    save_feature_importance_plot,
    save_json,
)
from .training_dataset import load_prepared_dataset, load_prepared_metadata

TARGET_COLUMN = "yield_bu_acre"
BASELINE_MODELS = ("BaselineTrainMean", "BaselinePreviousYearSameCounty")
DEFAULT_YIELD_DATASET_DIR = Path("data") / "yield_training"
DEFAULT_YIELD_RUNS_DIR = Path("training") / "yield_runs"
DEFAULT_SOURCE_DATASET_DIR = Path("data") / "training"


@dataclass(slots=True)
class PreparedYieldDataset:
    dataset_dir: Path
    feature_group: str
    target_column: str
    feature_columns: list[str]
    feature_columns_before_pruning: list[str]
    frames: dict[str, pd.DataFrame]
    metadata: dict[str, Any]


@dataclass(slots=True)
class YieldModelSummary:
    model: str
    model_type: str
    val_rmse: float
    val_mae: float
    val_mse: float
    val_r2: float
    test_rmse: float
    test_mae: float
    test_mse: float
    test_r2: float
    trainable_parameters: int
    total_parameters: int
    status: str
    model_dir: str
    model_path: str
    val_predictions_path: str
    test_predictions_path: str
    metrics_path: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"rmse": float("nan"), "mae": float("nan"), "mse": float("nan"), "r2": float("nan")}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def _parameter_count(model: Any) -> int:
    estimator = getattr(model, "named_steps", {}).get("model") if hasattr(model, "named_steps") else model
    if estimator is None:
        return 0
    if hasattr(estimator, "coef_"):
        return int(np.asarray(estimator.coef_).size)
    return 0


def _baseline_predictions(train_df: pd.DataFrame, predict_df: pd.DataFrame, target_col: str) -> dict[str, np.ndarray]:
    train_mean = float(np.nanmean(train_df[target_col].to_numpy(dtype=float)))
    lookup = {
        (str(row.county_id), int(row.year)): float(getattr(row, target_col))
        for row in train_df[["county_id", "year", target_col]].itertuples(index=False)
    }
    return {
        "BaselineTrainMean": np.full(len(predict_df), train_mean, dtype=float),
        "BaselinePreviousYearSameCounty": np.asarray(
            [
                lookup.get((str(row.county_id), int(row.year) - 1), train_mean)
                for row in predict_df[["county_id", "year"]].itertuples(index=False)
            ],
            dtype=float,
        ),
    }


def _prediction_frame_from_values(
    predict_df: pd.DataFrame,
    target_col: str,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    columns = ["county_id"]
    if "crop_type" in predict_df.columns:
        columns.append("crop_type")
    columns.extend([col for col in ["year", "month"] if col in predict_df.columns])
    columns.append(target_col)
    frame = predict_df[columns].copy()
    for name, values in predictions.items():
        frame[name] = np.asarray(values, dtype=float)
    return frame


def _resolve_split_years(years: list[int], train_years: list[int] | None, val_years: list[int] | None, test_years: list[int] | None) -> tuple[list[int], list[int], list[int]]:
    available = sorted({int(year) for year in years})
    if not available:
        raise ValueError("No yearly data is available to build a yield dataset.")
    if train_years or val_years or test_years:
        train = sorted({int(year) for year in train_years or []})
        val = sorted({int(year) for year in val_years or []})
        test = sorted({int(year) for year in test_years or []})
        if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
            raise ValueError("train-years, val-years, and test-years must be disjoint.")
        used = set(train) | set(val) | set(test)
        missing = [year for year in available if year not in used]
        if missing:
            raise ValueError(
                "The requested split years do not cover every year in the prepared monthly table. "
                f"Missing years: {missing}"
        )
        if not train or not test:
            raise ValueError("train-years and test-years must both be non-empty.")
        if not val:
            raise ValueError("val-years must be non-empty for explicit yield splits.")
        return train, val, test
    if len(available) < 3:
        raise ValueError(
            "At least three years are required to derive an automatic train/val/test split."
        )
    return available[:-2], [available[-2]], [available[-1]]


def _split_frame_by_years(frame: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    if not years:
        return frame.iloc[0:0].copy()
    return frame[frame["year"].astype(int).isin(years)].copy()


def _discover_source_usda_paths(root: Path, *, crop_type: str, years: list[int]) -> list[Path]:
    candidates = discover_usda_candidates(root)
    selected: list[Path] = []
    for path in candidates:
        path_crop = normalize_crop_type(infer_crop_type_from_path(path))
        if path_crop != crop_type:
            continue
        match = path.name.rsplit("_", 1)[-1].removesuffix(".csv")
        try:
            year = int(match)
        except ValueError:
            continue
        if year in years:
            selected.append(path)
    return sorted(selected)


def prepare_yield_dataset(
    *,
    source_dataset_dir: Path | None = DEFAULT_SOURCE_DATASET_DIR,
    monthly_path: Path | None = None,
    usda_paths: list[Path] | None = None,
    output_dir: Path = DEFAULT_YIELD_DATASET_DIR,
    crop_type: str = "corn",
    feature_group: str | None = None,
    train_years: list[int] | None = None,
    val_years: list[int] | None = None,
    test_years: list[int] | None = None,
    min_overlap_rows: int = 12,
) -> PreparedYieldDataset:
    root = Path(__file__).resolve().parents[2]
    resolved_monthly_path: Path | None = None
    source_metadata: dict[str, Any] = {}
    if source_dataset_dir is not None:
        source_dataset_dir = Path(source_dataset_dir)
        if not source_dataset_dir.exists():
            raise FileNotFoundError(f"Prepared source dataset directory not found: {source_dataset_dir}")
        prepared_source = load_prepared_dataset(source_dataset_dir)
        source_metadata = load_prepared_metadata(source_dataset_dir)
        monthly = prepared_source["all"].copy()
        resolved_monthly_path = source_dataset_dir / "all.parquet"
        if feature_group is None:
            feature_group = str(source_metadata.get("feature_group", "all"))
        if train_years is None:
            train_years = list(source_metadata.get("train_years", [])) or None
        if val_years is None:
            val_years = list(source_metadata.get("val_years", [])) or None
        if test_years is None:
            test_years = list(source_metadata.get("test_years", [])) or None
    else:
        if monthly_path is None:
            raise ValueError("Either source_dataset_dir or monthly_path must be provided.")
        resolved_monthly_path = Path(monthly_path)
        monthly = pd.read_csv(resolved_monthly_path) if resolved_monthly_path.suffix.lower() == ".csv" else pd.read_parquet(resolved_monthly_path)

    crop_type = normalize_crop_type(crop_type) or crop_type
    if "crop_type" in monthly.columns:
        monthly = monthly[monthly["crop_type"].map(normalize_crop_type).eq(crop_type)]
    if monthly.empty:
        raise ValueError(f"No monthly rows remain after applying the crop filter {crop_type!r}.")

    selected_features = selected_feature_columns(feature_group or "all")
    if usda_paths:
        resolved_usda_paths = resolve_usda_paths(root, usda_paths, monthly, crop_type)
    else:
        year_candidates = sorted(pd.to_numeric(monthly["year"], errors="coerce").dropna().astype(int).unique().tolist())
        resolved_usda_paths = _discover_source_usda_paths(root, crop_type=crop_type, years=year_candidates)
        if not resolved_usda_paths:
            resolved_usda_paths = resolve_usda_paths(root, None, monthly, crop_type)
    usda = load_usda_yield_tables(resolved_usda_paths, crop_type=crop_type)
    if "crop_type" in usda.columns:
        usda = usda[usda["crop_type"].map(normalize_crop_type).eq(crop_type)]

    merged, feature_cols, overlap_error = build_monthly_training_frame(monthly, usda, selected_features)
    if merged.empty:
        raise ValueError(overlap_error or "Could not build a merged monthly yield table.")
    if len(merged) < min_overlap_rows:
        raise ValueError(
            f"Matched monthly yield rows are too small for reliable training: {len(merged)} < {min_overlap_rows}."
        )

    feature_cols_before_pruning = list(feature_cols)
    feature_cols, pruning_report = prune_feature_columns(merged, feature_cols)

    year_values = sorted(pd.to_numeric(merged["year"], errors="coerce").dropna().astype(int).unique().tolist())
    resolved_train_years, resolved_val_years, resolved_test_years = _resolve_split_years(
        year_values,
        train_years,
        val_years,
        test_years,
    )

    frames = {
        "all": merged.reset_index(drop=True),
        "train": _split_frame_by_years(merged, resolved_train_years).reset_index(drop=True),
        "val": _split_frame_by_years(merged, resolved_val_years).reset_index(drop=True),
        "test": _split_frame_by_years(merged, resolved_test_years).reset_index(drop=True),
    }
    if frames["train"].empty or frames["test"].empty:
        raise ValueError("The requested split did not yield non-empty train and test sets.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in frames.items():
        _save_frame(frame, output_dir / f"{split_name}.parquet")

    metadata = {
        "monthly_path": resolved_monthly_path,
        "source_dataset_dir": source_dataset_dir,
        "source_dataset_metadata": source_metadata,
        "usda_paths": resolved_usda_paths,
        "crop_type": crop_type,
        "feature_group": feature_group or source_metadata.get("feature_group", "all"),
        "target_column": TARGET_COLUMN,
        "label_strategy": "annual_yield_copied_to_monthly_rows",
        "feature_columns": feature_cols,
        "feature_columns_before_pruning": feature_cols_before_pruning,
        "rows": int(len(merged)),
        "train_rows": int(len(frames["train"])),
        "val_rows": int(len(frames["val"])),
        "test_rows": int(len(frames["test"])),
        "train_years": resolved_train_years,
        "val_years": resolved_val_years,
        "test_years": resolved_test_years,
        "available_years": year_values,
        "pruning": pruning_report,
        "target_units": sorted(str(value) for value in merged["target_unit"].dropna().unique()) if "target_unit" in merged.columns else [],
        "split_strategy": "explicit_years" if (train_years or val_years or test_years) else "auto_latest_years",
    }
    save_json(metadata, output_dir / "metadata.json")
    (output_dir / "yield_dataset_summary.txt").write_text(
        "\n".join(
            [
                f"monthly_path={resolved_monthly_path}",
                f"usda_paths={', '.join(str(path) for path in resolved_usda_paths)}",
                f"crop_type={crop_type}",
                f"feature_group={feature_group}",
                f"target_column={TARGET_COLUMN}",
                f"train_years={' '.join(str(year) for year in resolved_train_years)}",
                f"val_years={' '.join(str(year) for year in resolved_val_years)}",
                f"test_years={' '.join(str(year) for year in resolved_test_years)}",
                f"rows={len(merged)}",
                f"train_rows={len(frames['train'])}",
                f"val_rows={len(frames['val'])}",
                f"test_rows={len(frames['test'])}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedYieldDataset(
        dataset_dir=output_dir,
        feature_group=feature_group,
        target_column=TARGET_COLUMN,
        feature_columns=feature_cols,
        feature_columns_before_pruning=feature_cols_before_pruning,
        frames=frames,
        metadata=metadata,
    )


def load_prepared_yield_dataset(dataset_dir: Path) -> PreparedYieldDataset:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prepared yield metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = {
        split: pd.read_parquet(dataset_dir / f"{split}.parquet")
        for split in ("all", "train", "val", "test")
    }
    feature_cols = list(metadata.get("feature_columns", []))
    return PreparedYieldDataset(
        dataset_dir=dataset_dir,
        feature_group=str(metadata.get("feature_group", "all")),
        target_column=str(metadata.get("target_column", TARGET_COLUMN)),
        feature_columns=feature_cols,
        feature_columns_before_pruning=list(metadata.get("feature_columns_before_pruning", feature_cols)),
        frames=frames,
        metadata=metadata,
    )


def _fit_pipeline(candidate: Any, train_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> Any:
    candidate.fit(train_df[feature_cols], train_df[target_col].to_numpy(dtype=float))
    return candidate


def _select_best_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    random_state: int,
    include_optional_models: bool,
    tune_hyperparameters: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    selected: dict[str, Any] = {}
    val_scores: dict[str, float] = {}
    if tune_hyperparameters:
        for model_name, candidates in build_model_candidate_pipelines(random_state=random_state).items():
            best_score = float("inf")
            best_candidate = candidates[0]
            for candidate in candidates:
                fitted = _fit_pipeline(clone(candidate), train_df, feature_cols, target_col)
                pred = fitted.predict(val_df[feature_cols])
                score = float(np.sqrt(mean_squared_error(val_df[target_col].to_numpy(dtype=float), pred)))
                if score < best_score:
                    best_score = score
                    best_candidate = fitted
            selected[model_name] = best_candidate
            val_scores[model_name] = best_score
    else:
        for model_name, candidate in build_model_pipelines(
            random_state=random_state,
            include_optional_models=False,
        ).items():
            fitted = _fit_pipeline(clone(candidate), train_df, feature_cols, target_col)
            pred = fitted.predict(val_df[feature_cols])
            val_scores[model_name] = float(np.sqrt(mean_squared_error(val_df[target_col].to_numpy(dtype=float), pred)))
            selected[model_name] = fitted

    if include_optional_models:
        optional = build_model_pipelines(
            random_state=random_state,
            include_optional_models=True,
        )
        for model_name, candidate in optional.items():
            if model_name in selected:
                continue
            fitted = _fit_pipeline(clone(candidate), train_df, feature_cols, target_col)
            pred = fitted.predict(val_df[feature_cols])
            val_scores[model_name] = float(np.sqrt(mean_squared_error(val_df[target_col].to_numpy(dtype=float), pred)))
            selected[model_name] = fitted

    return selected, val_scores


def _metric_row(
    *,
    model: str,
    model_type: str,
    trainable_parameters: int,
    total_parameters: int,
    status: str,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model,
        "model_type": model_type,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_mse": val_metrics["mse"],
        "val_r2": val_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_r2": test_metrics["r2"],
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "status": status,
    }


def train_yield_models(
    prepared: PreparedYieldDataset,
    *,
    run_dir: Path,
    models: list[str],
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    include_optional_models: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    train_df = prepared.frames["train"].copy()
    val_df = prepared.frames["val"].copy()
    test_df = prepared.frames["test"].copy()
    if val_df.empty:
        raise ValueError("Prepared yield dataset must include a non-empty validation split.")

    feature_cols = list(prepared.feature_columns)
    target_col = prepared.target_column
    final_train_df = pd.concat([train_df, val_df], ignore_index=True)

    selected_models, val_scores = _select_best_pipeline(
        train_df,
        val_df,
        feature_cols,
        target_col,
        random_state=random_state,
        include_optional_models=include_optional_models,
        tune_hyperparameters=tune_hyperparameters,
    )

    model_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {"models": {}, "best_trainable_model": None, "best_overall_model": None}

    for model_name in models:
        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        if model_name in BASELINE_MODELS:
            val_baselines = _baseline_predictions(train_df, val_df, target_col)
            test_baselines = _baseline_predictions(final_train_df, test_df, target_col)
            val_pred = val_baselines[model_name]
            test_pred = test_baselines[model_name]
            val_metrics = _metric_dict(val_df[target_col].to_numpy(dtype=float), val_pred)
            test_metrics = _metric_dict(test_df[target_col].to_numpy(dtype=float), test_pred)
            val_frame = _prediction_frame_from_values(val_df, target_col, val_baselines)
            test_frame = _prediction_frame_from_values(test_df, target_col, test_baselines)
            val_predictions_path = model_dir / "val_predictions.csv"
            test_predictions_path = model_dir / "test_predictions.csv"
            val_frame.to_csv(val_predictions_path, index=False)
            test_frame.to_csv(test_predictions_path, index=False)
            metrics_path = model_dir / "metrics.json"
            save_json(
                {
                    "model_name": model_name,
                    "model_type": "baseline",
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "status": "evaluated",
                },
                metrics_path,
            )
            row = {
                "model": model_name,
                "model_type": "baseline",
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_r2": val_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "test_r2": test_metrics["r2"],
                "trainable_parameters": 0,
                "total_parameters": 0,
                "status": "evaluated",
                "model_dir": str(model_dir),
                "model_path": "",
                "val_predictions_path": str(val_predictions_path),
                "test_predictions_path": str(test_predictions_path),
                "metrics_path": str(metrics_path),
            }
            model_rows.append(row)
            continue

        if model_name not in selected_models:
            raise ValueError(
                f"Unsupported yield model {model_name!r}. "
                f"Supported models: {', '.join(sorted(list(selected_models) + list(BASELINE_MODELS)))}"
            )

        candidate = selected_models[model_name]
        if candidate is None:
            raise RuntimeError(f"Failed to select a trained pipeline for {model_name}.")
        final_model = clone(candidate)
        final_model.fit(final_train_df[feature_cols], final_train_df[target_col].to_numpy(dtype=float))
        val_pred = candidate.predict(val_df[feature_cols])
        test_pred = final_model.predict(test_df[feature_cols])
        val_metrics = _metric_dict(val_df[target_col].to_numpy(dtype=float), val_pred)
        test_metrics = _metric_dict(test_df[target_col].to_numpy(dtype=float), test_pred)
        val_predictions_path = model_dir / "val_predictions.csv"
        test_predictions_path = model_dir / "test_predictions.csv"
        val_prediction_frame = build_prediction_frame(
            model=candidate,
            train_df=train_df,
            test_df=val_df,
            feature_cols=feature_cols,
            target_col=target_col,
            best_model_name=model_name,
        )
        test_prediction_frame = build_prediction_frame(
            model=final_model,
            train_df=final_train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            best_model_name=model_name,
        )
        val_prediction_frame.to_csv(val_predictions_path, index=False)
        test_prediction_frame.to_csv(test_predictions_path, index=False)
        model_path = model_dir / "model.joblib"
        joblib.dump(final_model, model_path)
        metrics_path = model_dir / "metrics.json"
        trainable = _parameter_count(final_model)
        save_json(
            {
                "model_name": model_name,
                "model_type": "ml",
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "selected_from_train_only": True,
                "candidate_val_rmse": val_scores.get(model_name),
                "status": "trained",
            },
            metrics_path,
        )
        row = {
            "model": model_name,
            "model_type": "ml",
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_r2": val_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_mse": test_metrics["mse"],
            "test_r2": test_metrics["r2"],
            "trainable_parameters": trainable,
            "total_parameters": trainable,
            "status": "trained",
            "model_dir": str(model_dir),
            "model_path": str(model_path),
            "val_predictions_path": str(val_predictions_path),
            "test_predictions_path": str(test_predictions_path),
            "metrics_path": str(metrics_path),
        }
        model_rows.append(row)
        artifacts["models"][model_name] = final_model

    results = pd.DataFrame(model_rows)
    if results.empty:
        raise RuntimeError("No yield models were trained or evaluated.")

    results = results.sort_values(["val_rmse", "test_rmse", "model"]).reset_index(drop=True)
    trainable_results = results[results["model_type"].eq("ml")].copy()
    if trainable_results.empty:
        raise RuntimeError("No trainable yield models were produced.")

    best_trainable_row = trainable_results.iloc[0].to_dict()
    best_trainable_model_name = str(best_trainable_row["model"])
    artifacts["best_trainable_model"] = best_trainable_model_name

    best_trainable_model = artifacts["models"].get(best_trainable_model_name)
    if best_trainable_model is None:
        best_trainable_model = selected_models[best_trainable_model_name]
    best_model_path = run_dir / "best_yield_model.joblib"
    joblib.dump(best_trainable_model, best_model_path)

    if best_trainable_model_name in artifacts["models"]:
        best_model = artifacts["models"][best_trainable_model_name]
    else:
        best_model = best_trainable_model

    best_val_predictions = build_prediction_frame(
        model=selected_models[best_trainable_model_name],
        train_df=train_df,
        test_df=val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        best_model_name=best_trainable_model_name,
    )
    best_test_predictions = build_prediction_frame(
        model=best_model,
        train_df=final_train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        best_model_name=best_trainable_model_name,
    )
    best_val_predictions.to_csv(run_dir / "val_predictions.csv", index=False)
    best_test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    importance = get_feature_importance(
        best_model,
        feature_cols,
        X_reference=final_train_df[feature_cols],
        y_reference=final_train_df[target_col].to_numpy(dtype=float),
    )
    importance_path = save_feature_importance_plot(importance, run_dir / "yield_feature_importance.png")
    importance.to_csv(run_dir / "yield_feature_importance.csv", index=False)
    residuals = build_residual_frame(best_model, test_df, feature_cols, target_col)
    residuals_path = run_dir / "prediction_residuals.csv"
    residuals.to_csv(residuals_path, index=False)
    month_benchmark = build_month_benchmark(
        best_test_predictions,
        target_col=target_col,
        model_columns=[best_trainable_model_name, *BASELINE_MODELS],
    )
    month_benchmark_path = run_dir / "month_benchmark.csv"
    month_benchmark.to_csv(month_benchmark_path, index=False)
    window_benchmark = build_window_benchmark(
        best_test_predictions,
        target_col=target_col,
        model_columns=[best_trainable_model_name, *BASELINE_MODELS],
    )
    window_benchmark_path = run_dir / "window_benchmark.csv"
    window_benchmark.to_csv(window_benchmark_path, index=False)

    results_path = run_dir / "metrics.csv"
    results.to_csv(results_path, index=False)
    model_specs_path = run_dir / "model_specs.csv"
    results[["model", "model_type", "model_dir", "model_path", "val_predictions_path", "test_predictions_path", "metrics_path", "status"]].to_csv(
        model_specs_path, index=False
    )
    report = {
        "run_dir": str(run_dir),
        "dataset_dir": str(prepared.dataset_dir),
        "feature_group": prepared.feature_group,
        "target_column": target_col,
        "models": results.to_dict(orient="records"),
        "best_trainable_model": best_trainable_model_name,
        "best_overall_model": str(results.iloc[0]["model"]),
        "feature_importance_path": str(importance_path),
        "residuals_path": str(residuals_path),
        "month_benchmark_path": str(month_benchmark_path),
        "window_benchmark_path": str(window_benchmark_path),
    }
    save_json(report, run_dir / "report.json")
    (run_dir / "report.md").write_text(
        "# Yield Training Report\n\n"
        f"- Dataset: `{prepared.dataset_dir}`\n"
        f"- Feature group: `{prepared.feature_group}`\n"
        f"- Target: `{target_col}`\n"
        f"- Best trainable model: `{best_trainable_model_name}`\n\n"
        "## Results\n\n"
        + "```text\n"
        + results[["model", "model_type", "val_rmse", "val_r2", "test_rmse", "test_r2"]].to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )

    metadata = dict(prepared.metadata)
    metadata.update(
        {
            "run_dir": str(run_dir),
            "results_csv": str(results_path),
            "model_specs_csv": str(model_specs_path),
            "best_trainable_model": best_trainable_model_name,
            "best_trainable_model_path": str(best_model_path),
            "feature_importance_csv": str(run_dir / "yield_feature_importance.csv"),
            "feature_importance_png": str(importance_path),
            "prediction_residuals_csv": str(residuals_path),
            "month_benchmark_csv": str(month_benchmark_path),
            "window_benchmark_csv": str(window_benchmark_path),
            "val_predictions_csv": str(run_dir / "val_predictions.csv"),
            "test_predictions_csv": str(run_dir / "test_predictions.csv"),
        }
    )
    save_json(metadata, run_dir / "config.json")

    return results, metadata
