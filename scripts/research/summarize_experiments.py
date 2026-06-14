from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def stringify_list(values) -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        return " ".join(str(v) for v in values)
    return str(values)


def summarize_run(run_dir: Path) -> list[dict]:
    config = load_json(run_dir / "config.json")
    status = load_json(run_dir / "run_status.json")
    artifacts = run_dir / "artifacts"
    metrics_path = artifacts / "model_metrics_by_modality.csv"
    feature_metrics_path = artifacts / "model_metrics_by_feature.csv"

    if status.get("status"):
        run_status = status["status"]
    elif metrics_path.exists():
        run_status = "completed"
    elif any(run_dir.rglob("*")):
        run_status = "partial"
    else:
        run_status = "failed"

    beats_by_model: dict[str, int] = {}
    if feature_metrics_path.exists():
        feature_metrics = pd.read_csv(feature_metrics_path)
        if {"model", "rmse_vs_naive"}.issubset(feature_metrics.columns):
            feature_metrics["beats_naive"] = feature_metrics["rmse_vs_naive"] < 1.0
            beats_by_model = (
                feature_metrics.groupby(["split", "model"])["beats_naive"].sum().astype(int).to_dict()
            )

    if not metrics_path.exists():
        return [
            {
                "run_name": config.get("run_name", run_dir.name),
                "states": stringify_list(config.get("state_codes")),
                "max_counties": config.get("max_counties"),
                "years": stringify_list(config.get("years")),
                "train_years": stringify_list(config.get("train_years")),
                "val_years": stringify_list(config.get("val_years")),
                "test_years": stringify_list(config.get("test_years")),
                "seq_len": config.get("seq_len"),
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "learning_rate": config.get("learning_rate"),
                "hidden_size": config.get("hidden_size"),
                "num_layers": config.get("num_layers"),
                "dropout": config.get("dropout"),
                "loss_mode": config.get("loss_mode", "raw_mse"),
                "feature_group": config.get("feature_group", "all"),
                "target_mode": config.get("target_mode", "raw"),
                "split": "",
                "model": "",
                "modality": "",
                "rmse": None,
                "mae": None,
                "r2": None,
                "features_beating_naive": None,
                "runtime_seconds": status.get("runtime_seconds"),
                "status": run_status,
            }
        ]

    metrics = pd.read_csv(metrics_path)
    rows: list[dict] = []
    for _, row in metrics.iterrows():
        model_name = str(row.get("model", ""))
        rows.append(
            {
                "run_name": config.get("run_name", run_dir.name),
                "states": stringify_list(config.get("state_codes")),
                "max_counties": config.get("max_counties"),
                "years": stringify_list(config.get("years")),
                "train_years": stringify_list(config.get("train_years")),
                "val_years": stringify_list(config.get("val_years")),
                "test_years": stringify_list(config.get("test_years")),
                "seq_len": config.get("seq_len"),
                "epochs": config.get("epochs"),
                "batch_size": config.get("batch_size"),
                "learning_rate": config.get("learning_rate"),
                "hidden_size": config.get("hidden_size"),
                "num_layers": config.get("num_layers"),
                "dropout": config.get("dropout"),
                "loss_mode": config.get("loss_mode", "raw_mse"),
                "feature_group": config.get("feature_group", "all"),
                "target_mode": config.get("target_mode", "raw"),
                "split": row.get("split"),
                "model": model_name,
                "modality": row.get("modality"),
                "rmse": row.get("rmse"),
                "mae": row.get("mae"),
                "r2": row.get("r2"),
                "features_beating_naive": beats_by_model.get((row.get("split"), model_name)),
                "runtime_seconds": status.get("runtime_seconds"),
                "status": run_status,
            }
        )
    return rows


def summarize_blank_fill_run(run_dir: Path) -> list[dict]:
    config = load_json(run_dir / "config.json")
    status = load_json(run_dir / "run_status.json")
    artifacts = run_dir / "artifacts"
    summary_path = artifacts / f"{config.get('blank_fill_output_prefix', 'blank_fill')}_metrics_summary.csv"

    if status.get("status"):
        run_status = status["status"]
    elif summary_path.exists():
        run_status = "completed"
    elif any(run_dir.rglob("*")):
        run_status = "partial"
    else:
        run_status = "failed"

    if not summary_path.exists():
        return []

    summary = pd.read_csv(summary_path)
    rows: list[dict] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "run_name": config.get("run_name", run_dir.name),
                "target_mode": config.get("target_mode", "raw"),
                "states": stringify_list(config.get("state_codes")),
                "max_counties": config.get("max_counties"),
                "years": stringify_list(config.get("years")),
                "train_years": stringify_list(config.get("train_years")),
                "val_years": stringify_list(config.get("val_years")),
                "test_years": stringify_list(config.get("test_years")),
                "seq_len": config.get("seq_len"),
                "loss_mode": config.get("loss_mode", "raw_mse"),
                "feature_group": config.get("feature_group", "all"),
                "model": row.get("model"),
                "known_months": row.get("known_months"),
                "modality": row.get("modality"),
                "rmse": row.get("rmse"),
                "mae": row.get("mae"),
                "beats_lag1": row.get("beats_lag1"),
                "beats_seasonal_last_year": row.get("beats_seasonal_last_year"),
                "runtime_seconds": status.get("runtime_seconds"),
                "status": run_status,
            }
        )
    return rows


def build_summary(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        rows.extend(summarize_run(run_dir))
    return pd.DataFrame(rows)


def build_blank_fill_summary(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        rows.extend(summarize_blank_fill_run(run_dir))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CropNet experiment runs under outputs/experiments.")
    parser.add_argument("--experiment-root", default="outputs/experiments")
    parser.add_argument("--output-csv", "--output", dest="output_csv", default=None)
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv).resolve() if args.output_csv else experiment_root / "experiment_summary.csv"
    blank_fill_output_csv = experiment_root / "blank_fill_experiment_summary.csv"

    summary = build_summary(experiment_root)
    blank_fill_summary = build_blank_fill_summary(experiment_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    blank_fill_summary.to_csv(blank_fill_output_csv, index=False)
    print(output_csv)
    print(blank_fill_output_csv)
    print(summary.head(20).to_string(index=False) if not summary.empty else "No runs found.")


if __name__ == "__main__":
    main()
