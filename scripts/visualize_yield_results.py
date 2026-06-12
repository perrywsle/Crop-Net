from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = ("rmse", "mae", "mape", "r2")
BASELINE_PREFIX = "baseline"
DEFAULT_OUTPUT_DIR = Path("outputs/yield_baseline/corn_ia_2017_2022_monthly_full/plots")


def resolve_existing_path(path: Path, alternatives: list[str]) -> Path:
    if path.exists():
        return path
    for name in alternatives:
        candidate = path.parent / name
        if candidate.exists():
            print(f"Using fallback artifact: {candidate}")
            return candidate
    raise FileNotFoundError(f"File not found: {path}. Tried alternatives: {alternatives}")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out


def find_column(frame: pd.DataFrame, candidates: list[str], *, contains_all: list[str] | None = None) -> str | None:
    lower_to_original = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if contains_all:
        for col in frame.columns:
            lowered = str(col).lower()
            if all(part in lowered for part in contains_all):
                return str(col)
    return None


def model_name_column(metrics: pd.DataFrame) -> str:
    col = find_column(metrics, ["model", "model_name", "name"])
    if col is None:
        raise ValueError("Metrics table needs a model/model_name column.")
    return col


def select_best_ml_model(metrics: pd.DataFrame) -> str | None:
    if "rmse" not in {str(col).lower() for col in metrics.columns}:
        return None
    rmse_col = find_column(metrics, ["rmse"])
    model_col = model_name_column(metrics)
    model_type_col = find_column(metrics, ["model_type", "type"])
    ranked = metrics.copy()
    ranked[rmse_col] = pd.to_numeric(ranked[rmse_col], errors="coerce")
    ranked = ranked.dropna(subset=[rmse_col])
    if ranked.empty:
        return None
    if model_type_col is not None:
        ml_rows = ranked[ranked[model_type_col].astype(str).str.lower().eq("ml")]
        if not ml_rows.empty:
            ranked = ml_rows
    else:
        non_baseline = ranked[~ranked[model_col].astype(str).str.lower().str.startswith(BASELINE_PREFIX)]
        if not non_baseline.empty:
            ranked = non_baseline
    return str(ranked.sort_values(rmse_col, ascending=True).iloc[0][model_col])


def detect_actual_column(predictions: pd.DataFrame) -> str | None:
    return find_column(
        predictions,
        ["actual", "y_true", "yield_bu_acre", "yield", "target", "target_value", "observed"],
        contains_all=["yield"],
    )


def detect_prediction_column(predictions: pd.DataFrame, metrics: pd.DataFrame) -> str | None:
    best_model = select_best_ml_model(metrics)
    if best_model is not None:
        for col in predictions.columns:
            if str(col).lower() == best_model.lower():
                return str(col)
    return find_column(predictions, ["best_prediction", "prediction", "predicted", "y_pred", "pred"])


def detect_county_column(predictions: pd.DataFrame) -> str | None:
    return find_column(predictions, ["county_id", "county", "county_name", "fips", "fips_code"], contains_all=["county"])


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_metric(metrics: pd.DataFrame, metric: str, outdir: Path) -> None:
    metric_col = find_column(metrics, [metric])
    if metric_col is None:
        print(f"Skipping {metric.upper()} chart: metric column not found.")
        return
    model_col = model_name_column(metrics)
    frame = metrics[[model_col, metric_col]].copy()
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    frame = frame.dropna(subset=[metric_col])
    if frame.empty:
        print(f"Skipping {metric.upper()} chart: no numeric values.")
        return
    ascending = metric.lower() != "r2"
    frame = frame.sort_values(metric_col, ascending=ascending)
    colors = ["#4C78A8" if not str(model).lower().startswith("baseline") else "#F58518" for model in frame[model_col]]
    plt.figure(figsize=(10, 5))
    plt.bar(frame[model_col].astype(str), frame[metric_col], color=colors)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric.upper())
    better = "lower is better" if ascending else "higher is better"
    plt.title(f"Model Comparison: {metric.upper()} ({better})")
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir / f"model_comparison_{metric.lower()}.png")


def plot_actual_vs_predicted(predictions: pd.DataFrame, actual_col: str, pred_col: str, outdir: Path) -> None:
    actual = pd.to_numeric(predictions[actual_col], errors="coerce")
    predicted = pd.to_numeric(predictions[pred_col], errors="coerce")
    valid = actual.notna() & predicted.notna()
    if not valid.any():
        print("Skipping actual-vs-predicted chart: no valid numeric pairs.")
        return
    actual = actual[valid]
    predicted = predicted[valid]
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.65, s=24)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Actual yield")
    plt.ylabel("Predicted yield")
    plt.title("Actual vs Predicted Yield")
    plt.grid(alpha=0.25)
    save_current_figure(outdir / "actual_vs_predicted.png")


def plot_residuals(predictions: pd.DataFrame, actual_col: str, pred_col: str, outdir: Path) -> None:
    actual = pd.to_numeric(predictions[actual_col], errors="coerce")
    predicted = pd.to_numeric(predictions[pred_col], errors="coerce")
    residual = predicted - actual
    residual = residual[np.isfinite(residual)]
    if residual.empty:
        print("Skipping residual chart: no valid residuals.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=30, color="#54A24B", alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual: predicted - actual")
    plt.ylabel("Count")
    plt.title("Residual Distribution\nPositive = overprediction, negative = underprediction")
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(outdir / "residual_distribution.png")


def plot_worst_counties(predictions: pd.DataFrame, actual_col: str, pred_col: str, outdir: Path) -> None:
    county_col = detect_county_column(predictions)
    if county_col is None:
        print("Skipping county error chart: no county column found.")
        return
    frame = predictions[[county_col, actual_col, pred_col]].copy()
    frame[actual_col] = pd.to_numeric(frame[actual_col], errors="coerce")
    frame[pred_col] = pd.to_numeric(frame[pred_col], errors="coerce")
    frame = frame.dropna(subset=[actual_col, pred_col])
    if frame.empty:
        print("Skipping county error chart: no valid county prediction rows.")
        return
    county = (
        frame.groupby(county_col, dropna=False)
        .agg(actual=(actual_col, "mean"), predicted=(pred_col, "mean"))
        .reset_index()
    )
    county["error"] = county["predicted"] - county["actual"]
    county["abs_error"] = county["error"].abs()
    county = county.sort_values("abs_error", ascending=False)
    table_path = outdir / "county_prediction_errors.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    county.to_csv(table_path, index=False)
    print(f"Saved: {table_path}")
    top = county.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top[county_col].astype(str), top["abs_error"], color="#E45756")
    plt.xlabel("Absolute error")
    plt.ylabel("County")
    plt.title("Top 20 Worst County Yield Errors")
    plt.grid(axis="x", alpha=0.25)
    save_current_figure(outdir / "worst_county_errors.png")


def unwrap_estimator(model: Any) -> Any:
    if hasattr(model, "named_steps"):
        if "model" in model.named_steps:
            return model.named_steps["model"]
        if model.steps:
            return model.steps[-1][1]
    return model


def load_feature_names(features_path: Path | None, model: Any) -> list[str] | None:
    if features_path is not None and features_path.exists():
        names = [line.strip() for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]
    return None


def plot_feature_importance(model_path: Path, features_path: Path | None, outdir: Path) -> None:
    if not model_path.exists():
        print(f"Skipping feature importance: model file not found: {model_path}")
        return
    model = joblib.load(model_path)
    estimator = unwrap_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        print("Skipping feature importance: selected model does not expose feature_importances_.")
        return
    importances = np.asarray(estimator.feature_importances_, dtype=float)
    names = load_feature_names(features_path, model)
    if names is None or len(names) != len(importances):
        if names is not None:
            print("Feature name count does not match importances; using generated feature names.")
        names = [f"feature_{idx}" for idx in range(len(importances))]
    frame = pd.DataFrame({"feature": names, "importance": importances})
    frame = frame.sort_values("importance", ascending=False)
    importance_path = outdir / "feature_importance_top20.csv"
    outdir.mkdir(parents=True, exist_ok=True)
    frame.head(20).to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")
    top = frame.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"], color="#72B7B2")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    plt.grid(axis="x", alpha=0.25)
    save_current_figure(outdir / "feature_importance_top20.png")


def visualize(metrics_path: Path, predictions_path: Path, model_path: Path, features_path: Path | None, outdir: Path) -> None:
    metrics_path = resolve_existing_path(metrics_path, ["yield_model_benchmark.csv", "metrics.csv"])
    predictions_path = resolve_existing_path(predictions_path, ["prediction_residuals.csv", "predictions_2022.csv"])
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = normalize_columns(pd.read_csv(metrics_path))
    predictions = normalize_columns(pd.read_csv(predictions_path))
    for metric in METRIC_COLUMNS:
        plot_metric(metrics, metric, outdir)
    actual_col = detect_actual_column(predictions)
    pred_col = detect_prediction_column(predictions, metrics)
    if actual_col is None or pred_col is None:
        print(f"Skipping prediction charts: actual_col={actual_col}, prediction_col={pred_col}")
    else:
        print(f"Using actual column: {actual_col}")
        print(f"Using prediction column: {pred_col}")
        plot_actual_vs_predicted(predictions, actual_col, pred_col, outdir)
        plot_residuals(predictions, actual_col, pred_col, outdir)
        plot_worst_counties(predictions, actual_col, pred_col, outdir)
    plot_feature_importance(model_path, features_path, outdir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Iowa Corn monthly yield model results.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    visualize(args.metrics, args.predictions, args.model, args.features, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
