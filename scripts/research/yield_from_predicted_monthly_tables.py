from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.blank_fill import rollout_blank_fill
from cropnet_forecasting.data import prepare_monthly_features, read_table
from cropnet_forecasting.features import META_COLS, selected_feature_columns
from cropnet_forecasting.models import CropNetModelFactory
from cropnet_forecasting.predictor import BlankFillPredictor
from cropnet_forecasting.scaling import FeatureScaler
from cropnet_forecasting.yield_regression import (
    build_month_benchmark,
    build_monthly_training_frame,
    build_prediction_frame,
    build_residual_frame,
    build_window_benchmark,
    evaluate_feature_group_benchmarks,
    evaluate_models,
    evaluate_naive_baselines,
    evaluate_year_cv,
    get_feature_importance,
    normalize_county_id,
    normalize_crop_type,
    overlap_summary,
    prune_feature_columns,
    regression_metric_row,
    save_feature_importance_plot,
    split_dataset,
)


DEFAULT_MONTHLY_TABLE = ROOT / "outputs/experiments/corn_ia_monthly_2017_2022/artifacts/official_monthly_feature_table.parquet"
DEFAULT_LABELLED_FRAME = ROOT / "outputs/yield_baseline/corn_ia_2017_2022_monthly_full/merged_monthly_training_frame.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/predicted_yield_experiments"
DEFAULT_SCALER = ROOT / "weights/scaler.csv"
DEFAULT_MODELS = ("gru", "transformer_encoder", "lstm")


@dataclass(frozen=True)
class GeneratedTable:
    name: str
    frame: pd.DataFrame
    path: Path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_official_monthly(path: Path, feature_group: str) -> pd.DataFrame:
    feature_names = selected_feature_columns(feature_group)
    frame = read_table(path)
    missing = [col for col in META_COLS + feature_names if col not in frame.columns]
    if missing:
        raise ValueError(f"Monthly table is missing required columns: {missing}")
    out = frame[META_COLS + feature_names].copy()
    out["county_id"] = out["county_id"].astype(str).str.zfill(5)
    out["crop_type"] = out["crop_type"].astype(str).map(normalize_crop_type)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype(int)
    return out.sort_values(META_COLS).reset_index(drop=True)


def load_usda_labels_from_training_frame(path: Path, crop_type: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda col: col in {"county_id", "crop_type", "year", "yield_bu_acre", "target_unit"})
    required = {"county_id", "year", "yield_bu_acre"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Labelled yield frame is missing required columns: {missing}")
    out = frame.copy()
    out["county_id"] = normalize_county_id(out["county_id"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["yield_bu_acre"] = pd.to_numeric(out["yield_bu_acre"], errors="coerce")
    if "crop_type" in out.columns:
        out["crop_type"] = out["crop_type"].astype(str).map(normalize_crop_type)
    elif crop_type is not None:
        out["crop_type"] = normalize_crop_type(crop_type)
    if "target_unit" not in out.columns:
        out["target_unit"] = "BU / ACRE"
    if crop_type is not None and "crop_type" in out.columns:
        out = out[out["crop_type"].eq(normalize_crop_type(crop_type))]
    return out.dropna(subset=["county_id", "year", "yield_bu_acre"]).drop_duplicates().reset_index(drop=True)


def build_predictor(model_name: str, checkpoint: Path, scaler_path: Path, feature_group: str, seq_len: int, target_mode: str, device: str) -> BlankFillPredictor:
    feature_names = selected_feature_columns(feature_group)
    scaler = FeatureScaler.from_csv(scaler_path).subset(feature_names)
    model = CropNetModelFactory.load_checkpoint(checkpoint, model_name=model_name, device=device)
    return BlankFillPredictor(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_name=model_name,
        target_mode=target_mode,
        seq_len=seq_len,
        device=device,
    )


def deterministic_blank_fill(monthly: pd.DataFrame, feature_names: list[str], year: int, known_months: int, mode: str) -> pd.DataFrame:
    prepared = prepare_monthly_features(monthly, feature_names)
    rows: list[dict[str, Any]] = []
    for (county_id, crop_type), group in prepared.groupby(["county_id", "crop_type"], sort=True):
        group = group.sort_values(["year", "month"]).reset_index(drop=True)
        history = group[(group["year"] < year) | ((group["year"] == year) & (group["month"] <= known_months))].copy()
        if history.empty:
            continue
        for month in range(known_months + 1, 13):
            seasonal = group[(group["year"].eq(year - 1)) & (group["month"].eq(month))]
            if mode == "seasonal_last_year" and not seasonal.empty:
                source = seasonal.iloc[-1]
                source_note = "seasonal_last_year"
            else:
                source = history.iloc[-1]
                source_note = "lag1" if mode == "naive_lag1" else "fallback_last_history"
            row = {
                "county_id": str(county_id).zfill(5),
                "crop_type": str(crop_type),
                "year": int(year),
                "month": int(month),
                "known_months": int(known_months),
                "source_note": source_note,
            }
            for feature in feature_names:
                row[feature] = float(source[feature])
            rows.append(row)
            history = pd.concat([history, pd.DataFrame([{k: row[k] for k in META_COLS + feature_names}])], ignore_index=True)
    return pd.DataFrame(rows)


def merge_known_and_predicted(official: pd.DataFrame, predictions: pd.DataFrame, feature_names: list[str], known_months: int, source_name: str) -> pd.DataFrame:
    observed = official[official["month"].le(known_months)].copy()
    observed["known_months"] = int(known_months)
    observed["source_note"] = "observed_known_month"
    generated = predictions[META_COLS + ["known_months", "source_note"] + feature_names].copy()
    combined = pd.concat([observed[META_COLS + ["known_months", "source_note"] + feature_names], generated], ignore_index=True)
    combined["table_source"] = source_name
    return combined.sort_values(META_COLS).reset_index(drop=True)


def generate_model_table(
    model_name: str,
    monthly: pd.DataFrame,
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    scaler_path: Path,
    feature_group: str,
    seq_len: int,
    target_mode: str,
    known_months: int,
    years: list[int],
    device: str,
) -> GeneratedTable:
    feature_names = selected_feature_columns(feature_group)
    if model_name in {"naive_lag1", "seasonal_last_year"}:
        pieces = [
            deterministic_blank_fill(monthly, feature_names, year=year, known_months=known_months, mode=model_name)
            for year in years
        ]
    else:
        checkpoint = checkpoint_dir / f"{model_name}_best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint}")
        predictor = build_predictor(model_name, checkpoint, scaler_path, feature_group, seq_len, target_mode, device)
        pieces = [
            rollout_blank_fill(predictor, monthly_features=monthly, year=year, known_months=known_months).predictions
            for year in years
        ]
    predictions = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=META_COLS + feature_names)
    full = merge_known_and_predicted(monthly, predictions, feature_names, known_months, model_name)
    path = output_dir / "predicted_monthly_tables" / f"{model_name}_known{known_months}_monthly_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(path, index=False)
    return GeneratedTable(model_name, full, path)


def ensemble_table(name: str, tables: list[GeneratedTable], feature_names: list[str], weights: dict[str, float] | None, output_dir: Path, known_months: int) -> GeneratedTable:
    if not tables:
        raise ValueError("At least one generated table is required for an ensemble.")
    keys = META_COLS + ["known_months"]
    base = tables[0].frame[keys + ["source_note"]].copy()
    base["source_note"] = name
    values = np.zeros((len(base), len(feature_names)), dtype=float)
    weight_sum = 0.0
    for table in tables:
        aligned = base[keys].merge(table.frame[keys + feature_names], on=keys, how="left", validate="one_to_one")
        weight = 1.0 if weights is None else float(weights.get(table.name, 0.0))
        if weight <= 0:
            continue
        values += aligned[feature_names].to_numpy(dtype=float) * weight
        weight_sum += weight
    if weight_sum <= 0:
        raise ValueError(f"No positive weights were available for {name}.")
    base[feature_names] = values / weight_sum
    base["table_source"] = name
    path = output_dir / "predicted_monthly_tables" / f"{name}_known{known_months}_monthly_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(path, index=False)
    return GeneratedTable(name, base.sort_values(META_COLS).reset_index(drop=True), path)


def inverse_rmse_weights(summary_path: Path, model_names: list[str], known_months: int) -> dict[str, float]:
    equal = {name: 1.0 / len(model_names) for name in model_names}
    if not summary_path.exists():
        return equal
    summary = pd.read_csv(summary_path)
    frame = summary[
        summary["model"].isin(model_names)
        & summary["known_months"].eq(known_months)
        & summary["status"].eq("completed")
    ].copy()
    if frame.empty:
        return equal
    scores = frame.groupby("model")["rmse"].mean()
    inv = {name: 1.0 / max(float(scores.get(name, np.nan)), 1e-9) for name in model_names if pd.notna(scores.get(name, np.nan))}
    if set(inv) != set(model_names):
        return equal
    total = sum(inv.values())
    if total <= 0:
        return equal
    return {name: weight / total for name, weight in inv.items()}


def train_yield_from_generated_table(
    table: GeneratedTable,
    usda_labels: pd.DataFrame,
    output_dir: Path,
    *,
    feature_group: str,
    random_state: int,
    include_optional_models: bool,
) -> dict[str, Any]:
    selected_features = selected_feature_columns(feature_group)
    merged, feature_cols, overlap_error = build_monthly_training_frame(table.frame, usda_labels, selected_features)
    if merged.empty:
        raise ValueError(f"{table.name}: {overlap_error}")

    run_dir = output_dir / "yield_models" / table.name
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_cols_before_pruning = list(feature_cols)
    feature_cols, pruning = prune_feature_columns(merged, feature_cols)
    train_df, test_df, split_mode, split_metadata = split_dataset(merged, random_state=random_state)

    ml_results, fitted_models = evaluate_models(
        train_df,
        test_df,
        feature_cols,
        "yield_bu_acre",
        include_optional_models=include_optional_models,
        random_state=random_state,
        tune_hyperparameters=True,
        feature_group=feature_group,
    )
    baseline_results = evaluate_naive_baselines(train_df, test_df, "yield_bu_acre", feature_group=feature_group)
    results = pd.concat([ml_results, baseline_results], ignore_index=True).sort_values(["model_type", "rmse", "mae", "model"]).reset_index(drop=True)
    best_model_name = str(ml_results.iloc[0]["model"])
    holdout_model = fitted_models[best_model_name]
    final_model = clone(holdout_model)
    final_model.fit(merged[feature_cols], merged["yield_bu_acre"].to_numpy(dtype=float))

    prediction_frame = build_prediction_frame(holdout_model, train_df, test_df, feature_cols, "yield_bu_acre", best_model_name)
    model_columns = [best_model_name, "BaselineTrainMean", "BaselinePreviousYearSameCounty"]
    importance = get_feature_importance(final_model, feature_cols, merged[feature_cols], merged["yield_bu_acre"].to_numpy(dtype=float))
    summary = overlap_summary(table.frame, usda_labels)

    merged.to_csv(run_dir / "merged_predicted_monthly_training_frame.csv", index=False)
    results.to_csv(run_dir / "yield_model_benchmark.csv", index=False)
    evaluate_feature_group_benchmarks(
        train_df,
        test_df,
        feature_cols,
        "yield_bu_acre",
        random_state=random_state,
        include_optional_models=include_optional_models,
        tune_hyperparameters=True,
    ).to_csv(run_dir / "feature_group_benchmark.csv", index=False)
    evaluate_year_cv(
        merged,
        feature_cols,
        "yield_bu_acre",
        random_state=random_state,
        include_optional_models=include_optional_models,
        tune_hyperparameters=True,
        feature_group=feature_group,
    ).to_csv(run_dir / "year_cv_benchmark.csv", index=False)
    build_month_benchmark(prediction_frame, target_col="yield_bu_acre", model_columns=model_columns).to_csv(run_dir / "month_benchmark.csv", index=False)
    build_window_benchmark(prediction_frame, target_col="yield_bu_acre", model_columns=model_columns).to_csv(run_dir / "window_benchmark.csv", index=False)
    build_residual_frame(holdout_model, test_df, feature_cols, "yield_bu_acre").to_csv(run_dir / "prediction_residuals.csv", index=False)
    importance.to_csv(run_dir / "yield_feature_importance.csv", index=False)
    save_feature_importance_plot(importance, run_dir / "yield_feature_importance.png", title=f"{table.name} Yield Feature Importance")
    joblib.dump(final_model, run_dir / "best_yield_model.joblib")
    save_json(pruning, run_dir / "feature_pruning_report.json")

    metadata = {
        "monthly_path": table.path,
        "label_source": DEFAULT_LABELLED_FRAME,
        "forecast_table_source": table.name,
        "uses_forecast_generated_features": True,
        "feature_group": feature_group,
        "target_grain": "monthly",
        "label_strategy": "annual_yield_copied_to_months_from_existing_training_frame",
        "feature_columns": feature_cols,
        "feature_columns_before_pruning": feature_cols_before_pruning,
        "split_mode": split_mode,
        "split_strategy": split_metadata["strategy"],
        "test_year": split_metadata["test_year"],
        "rows": len(merged),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "county_overlap": summary["county_overlap"],
        "year_overlap": summary["year_overlap"],
        "county_year_overlap": summary["county_year_overlap"],
        "best_model_name": best_model_name,
        "models_benchmarked": results["model"].tolist(),
        "random_state": random_state,
        "pruning": pruning,
    }
    save_json(metadata, run_dir / "yield_model_metadata.json")
    best_row = results[results["model"].eq(best_model_name)].iloc[0].to_dict()
    previous_baseline = results[results["model"].eq("BaselinePreviousYearSameCounty")].iloc[0].to_dict()
    return {
        "forecast_source": table.name,
        "predicted_monthly_table": str(table.path),
        "yield_model_dir": str(run_dir),
        "best_ml_model": best_model_name,
        "best_ml_rmse": best_row["rmse"],
        "best_ml_mae": best_row["mae"],
        "best_ml_r2": best_row["r2"],
        "previous_year_baseline_rmse": previous_baseline["rmse"],
        "beats_previous_year_baseline": float(best_row["rmse"]) < float(previous_baseline["rmse"]),
        "rows": len(merged),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_year": split_metadata["test_year"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate blank-fill predicted monthly tables and train yield regressors from them.")
    parser.add_argument("--monthly-table", type=Path, default=DEFAULT_MONTHLY_TABLE)
    parser.add_argument("--labelled-yield-frame", type=Path, default=DEFAULT_LABELLED_FRAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--include-baselines", action="store_true", help="Also generate naive_lag1 and seasonal_last_year predicted tables.")
    parser.add_argument("--known-months", type=int, default=1)
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--feature-group", default="all")
    parser.add_argument("--crop-type", default="corn")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--target-mode", default="seasonal_residual")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-optional-yield-models", action="store_true")
    parser.add_argument("--skip-yield-training", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = selected_feature_columns(args.feature_group)
    monthly = load_official_monthly(args.monthly_table, args.feature_group)
    if args.crop_type:
        monthly = monthly[monthly["crop_type"].eq(normalize_crop_type(args.crop_type))].copy()
    years = args.years or sorted(monthly["year"].dropna().astype(int).unique().tolist())

    model_names = list(dict.fromkeys(args.models + (["naive_lag1", "seasonal_last_year"] if args.include_baselines else [])))
    generated: list[GeneratedTable] = []
    for model_name in model_names:
        print(f"Generating predicted monthly table for {model_name}...")
        generated.append(
            generate_model_table(
                model_name,
                monthly,
                output_dir,
                checkpoint_dir=args.checkpoint_dir,
                scaler_path=args.scaler,
                feature_group=args.feature_group,
                seq_len=args.seq_len,
                target_mode=args.target_mode,
                known_months=args.known_months,
                years=years,
                device=args.device,
            )
        )

    learned = [table for table in generated if table.name not in {"naive_lag1", "seasonal_last_year"}]
    if len(learned) >= 2:
        generated.append(ensemble_table("ensemble_mean", learned, feature_names, None, output_dir, args.known_months))
        weights = inverse_rmse_weights(ROOT / "reports/tables/blank_fill_experiment_summary.csv", [table.name for table in learned], args.known_months)
        generated.append(ensemble_table("ensemble_weighted", learned, feature_names, weights, output_dir, args.known_months))
    else:
        weights = {}

    summaries: list[dict[str, Any]] = []
    if not args.skip_yield_training:
        usda_labels = load_usda_labels_from_training_frame(args.labelled_yield_frame, crop_type=args.crop_type)
        for table in generated:
            print(f"Training yield model from {table.name} predicted table...")
            summaries.append(
                train_yield_from_generated_table(
                    table,
                    usda_labels,
                    output_dir,
                    feature_group=args.feature_group,
                    random_state=args.random_state,
                    include_optional_models=args.include_optional_yield_models,
                )
            )

    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["best_ml_rmse", "best_ml_mae", "forecast_source"]).reset_index(drop=True)
        summary_frame.to_csv(output_dir / "predicted_table_yield_summary.csv", index=False)
    save_json(
        {
            "monthly_table": args.monthly_table,
            "labelled_yield_frame": args.labelled_yield_frame,
            "output_dir": output_dir,
            "feature_group": args.feature_group,
            "crop_type": args.crop_type,
            "known_months": args.known_months,
            "years": years,
            "models": model_names,
            "ensemble_weighted_weights": weights,
            "generated_tables": {table.name: table.path for table in generated},
            "yield_training_ran": not args.skip_yield_training,
        },
        output_dir / "run_metadata.json",
    )
    print(f"Done. Wrote predicted yield experiment artifacts to {output_dir}")


if __name__ == "__main__":
    main()
