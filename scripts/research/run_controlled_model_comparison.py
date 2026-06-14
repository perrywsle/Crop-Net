from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


MODELS = ["lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"]
TARGET_MODES = ["raw", "seasonal_residual"]
LOSS_MODES = ["raw_mse", "feature_normalized_mse"]
COMMON_ARGS = [
    "--full-run",
    "--state-codes",
    "IA",
    "--max-counties",
    "30",
    "--years",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "--train-years",
    "2017",
    "2018",
    "2019",
    "--val-years",
    "2020",
    "--test-years",
    "2021",
    "--quarters",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "--seq-len",
    "6",
    "--batch-size",
    "64",
    "--epochs",
    "100",
    "--early-stopping-patience",
    "10",
    "--learning-rate",
    "0.001",
    "--hidden-size",
    "64",
    "--num-layers",
    "1",
    "--dropout",
    "0.0",
    "--feature-groups",
    "all",
    "--run-blank-fill-eval",
    "--strict-blank-fill-no-future-fill",
    "--blank-fill-year",
    "2021",
    "--blank-fill-known-months",
    "0",
    "1",
    "3",
    "6",
    "9",
    "--blank-fill-output-prefix",
    "strict_blank_fill",
    "--resume",
]


def run(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def seed_monthly_artifacts(source_run_dir: Path, target_run_dir: Path) -> None:
    source_artifacts = source_run_dir / "artifacts"
    target_artifacts = target_run_dir / "artifacts"
    target_artifacts.mkdir(parents=True, exist_ok=True)
    for name in ["official_monthly_feature_table.parquet", "feature_contract_diagnostic.json"]:
        src = source_artifacts / name
        dst = target_artifacts / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def build_run_name(model_name: str, target_mode: str, loss_mode: str) -> str:
    target_tag = "raw" if target_mode == "raw" else "seasonal_residual"
    loss_tag = "rawmse" if loss_mode == "raw_mse" else "normloss"
    return f"ia30_seq6_{model_name}_{target_tag}_{loss_tag}"


def collect_model_summary(run_dir: Path) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(run_dir / "artifacts" / "model_metrics_by_modality.csv")
    metrics = metrics[metrics["split"].eq("test")].copy()
    metrics["run_name"] = config.get("run_name", run_dir.name)
    metrics["target_mode"] = config.get("target_mode")
    metrics["loss_mode"] = config.get("loss_mode", "raw_mse")
    metrics["feature_group"] = config.get("feature_group", "all")
    metrics["seq_len"] = config.get("seq_len")
    metrics["max_counties"] = config.get("max_counties")
    metrics["runtime_seconds"] = status.get("runtime_seconds")
    return metrics


def collect_blank_fill_summary(run_dir: Path) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(run_dir / "artifacts" / "strict_blank_fill_metrics_summary.csv")
    norm = pd.read_csv(run_dir / "artifacts" / "strict_blank_fill_normalized_metrics_summary.csv")
    keep_cols = [
        "model",
        "known_months",
        "modality",
        "rmse",
        "mae",
        "nrmse_std",
        "nrmse_range",
        "smape",
        "pearson_corr",
        "r2",
        "beats_lag1",
        "beats_seasonal_last_year",
    ]
    merged = raw.merge(norm[keep_cols], on=["model", "known_months", "modality"], how="left", suffixes=("", "_norm"))
    merged["run_name"] = config.get("run_name", run_dir.name)
    merged["target_mode"] = config.get("target_mode")
    merged["loss_mode"] = config.get("loss_mode", "raw_mse")
    merged["feature_group"] = config.get("feature_group", "all")
    merged["seq_len"] = config.get("seq_len")
    merged["max_counties"] = config.get("max_counties")
    merged["runtime_seconds"] = status.get("runtime_seconds")
    return merged


def collect_model_specs(run_dir: Path) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    specs = pd.read_csv(run_dir / "artifacts" / "model_specs.csv")
    specs["run_name"] = config.get("run_name", run_dir.name)
    return specs


def write_loss_curve_index(run_dirs: list[Path], output_path: Path) -> None:
    lines = ["# Loss Curve Index", ""]
    for run_dir in run_dirs:
        plots_dir = run_dir / "artifacts" / "plots_report"
        history_files = sorted((run_dir / "artifacts").glob("*_history.csv"))
        lines.append(f"## {run_dir.name}")
        if history_files:
            for hist in history_files:
                lines.append(f"- history: `{hist}`")
        pngs = sorted(plots_dir.glob("*loss_curve*.png")) if plots_dir.exists() else []
        if pngs:
            for png in pngs:
                lines.append(f"- plot: `{png}`")
        else:
            lines.append("- no loss plots found")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(root_dir: Path, model_summary: pd.DataFrame, blank_fill_summary: pd.DataFrame) -> None:
    known1_all = blank_fill_summary[(blank_fill_summary["known_months"].eq(1)) & (blank_fill_summary["modality"].eq("all"))].copy()
    known6_all = blank_fill_summary[(blank_fill_summary["known_months"].eq(6)) & (blank_fill_summary["modality"].eq("all"))].copy()
    best_raw_known1 = known1_all.sort_values("rmse").head(1)
    best_norm_known1 = known1_all.sort_values("nrmse_std").head(1)
    best_raw_known6 = known6_all.sort_values("rmse").head(1)
    best_norm_known6 = known6_all.sort_values("nrmse_std").head(1)
    one_step_overall = model_summary.groupby(["run_name", "model"], as_index=False).agg(mean_rmse=("rmse", "mean"))
    best_one_step = one_step_overall.sort_values("mean_rmse").head(1)

    lines = [
        "# Model Comparison V1",
        "",
        "## Scope",
        "",
        "- State: IA",
        "- Max counties: 30",
        "- Years: 2017-2021",
        "- Train/val/test: 2017-2019 / 2020 / 2021",
        "- Sequence length: 6",
        "- Strict blank-fill enabled",
        "- Models compared: lstm, gru, tiny_mamba_ssm, transformer_encoder",
        "- Target modes: raw, seasonal_residual",
        "- Loss modes: raw_mse, feature_normalized_mse",
        "",
        "## Best Results",
        "",
        "### One-Step Mean RMSE",
        "",
        best_one_step.to_markdown(index=False) if not best_one_step.empty else "No one-step results found.",
        "",
        "### Best known_months=1 by Raw RMSE",
        "",
        best_raw_known1.to_markdown(index=False) if not best_raw_known1.empty else "No known_months=1 results found.",
        "",
        "### Best known_months=1 by Normalized RMSE",
        "",
        best_norm_known1.to_markdown(index=False) if not best_norm_known1.empty else "No normalized known_months=1 results found.",
        "",
        "### Best known_months=6 by Raw RMSE",
        "",
        best_raw_known6.to_markdown(index=False) if not best_raw_known6.empty else "No known_months=6 results found.",
        "",
        "### Best known_months=6 by Normalized RMSE",
        "",
        best_norm_known6.to_markdown(index=False) if not best_norm_known6.empty else "No normalized known_months=6 results found.",
        "",
        "## Caveats",
        "",
        "- Raw RMSE remains strongly weather-scale sensitive.",
        "- Seasonal_last_year is still the key comparator under normalized metrics.",
        "- These runs stay within IA 30-county scope and are not yet the final deployment recipe.",
    ]
    (root_dir / "README_model_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled IA-30 model comparison matrix and collect summaries.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-root", default="outputs/experiments/model_comparison_v1")
    parser.add_argument("--monthly-source-run", default="outputs/experiments/ia_more_counties_seq6_bestsmall_blankfill")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    run_root = Path(args.run_root).resolve()
    runs_root = run_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    monthly_source_run = Path(args.monthly_source_run).resolve()
    python_bin = str(Path(args.python_bin).resolve())

    run_dirs: list[Path] = []
    for model_name in MODELS:
        for target_mode in TARGET_MODES:
            for loss_mode in LOSS_MODES:
                run_name = build_run_name(model_name, target_mode, loss_mode)
                run_dir = runs_root / run_name
                run_dirs.append(run_dir)
                seed_monthly_artifacts(monthly_source_run, run_dir)
                cmd = [
                    python_bin,
                    "cropnet_feature_forecasting_v12_server.py",
                    "--run-name",
                    run_name,
                    "--experiment-root",
                    str(runs_root),
                    "--models",
                    "naive_lag1",
                    "seasonal_last_year",
                    model_name,
                    "--target-mode",
                    target_mode,
                    "--loss-mode",
                    loss_mode,
                ] + COMMON_ARGS
                if target_mode == "seasonal_residual":
                    cmd.append("--blank-fill-residual-seasonal")
                run(cmd, project_root)
                run(
                    [
                        python_bin,
                        "analyze_blank_fill_diagnostics.py",
                        "--run-dir",
                        str(run_dir),
                        "--blank-fill-prefix",
                        "strict_blank_fill",
                        "--plot-dir-name",
                        "plots_diagnostics_strict",
                    ],
                    project_root,
                )
                run(
                    [
                        python_bin,
                        "generate_loss_plots.py",
                        "--artifacts-dir",
                        str(run_dir / "artifacts"),
                    ],
                    project_root,
                )

    run(
        [
            python_bin,
            "summarize_experiments.py",
            "--experiment-root",
            str(runs_root),
            "--output",
            str(run_root / "experiment_summary.csv"),
        ],
        project_root,
    )

    model_summary = pd.concat([collect_model_summary(run_dir) for run_dir in run_dirs], ignore_index=True)
    blank_fill_summary = pd.concat([collect_blank_fill_summary(run_dir) for run_dir in run_dirs], ignore_index=True)
    model_specs_summary = pd.concat([collect_model_specs(run_dir) for run_dir in run_dirs], ignore_index=True)

    model_summary.to_csv(run_root / "model_comparison_summary.csv", index=False)
    blank_fill_summary.to_csv(run_root / "blank_fill_comparison_summary.csv", index=False)
    model_specs_summary.to_csv(run_root / "model_specs_summary.csv", index=False)
    write_loss_curve_index(run_dirs, run_root / "loss_curve_index.md")
    write_readme(run_root, model_summary, blank_fill_summary)

    print(run_root / "model_comparison_summary.csv")
    print(run_root / "blank_fill_comparison_summary.csv")
    print(run_root / "model_specs_summary.csv")
    print(run_root / "loss_curve_index.md")
    print(run_root / "README_model_comparison.md")


if __name__ == "__main__":
    main()
