from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


KEY_COLS = [
    "county_id",
    "crop_type",
    "blank_fill_year",
    "known_months",
    "target_month",
    "horizon",
    "feature",
    "modality",
]
SELECTED_FEATURES = [
    "weather_solar_radiation_mean",
    "weather_gdd",
    "weather_total_precipitation",
    "ndvi_mean",
    "ag_mean_brightness",
]
OVERLAY_MODELS = [
    "ensemble_mean",
    "ensemble_weighted",
    "LSTM seasonal_residual",
    "SARIMA",
    "seasonal_last_year",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_to_modality(feature: str) -> str:
    if feature.startswith("ag_"):
        return "ag"
    if feature.startswith("ndvi_"):
        return "ndvi"
    if feature.startswith("weather_"):
        return "weather"
    return "other"


def safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def summarize_group(group: pd.DataFrame) -> pd.Series:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    sq_err = group["squared_error"].to_numpy(dtype=float)
    abs_err = group["abs_error"].to_numpy(dtype=float)
    out = {
        "count": int(len(group)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "mae": float(np.mean(abs_err)),
    }
    return pd.Series(out)


def compute_blank_fill_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = predictions.copy()
    pred["squared_error"] = pred["abs_error"] ** 2

    by_horizon = (
        pred.groupby(["model", "known_months", "horizon", "modality"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    by_month = (
        pred.groupby(["model", "known_months", "target_month", "modality"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    feature_metrics = (
        pred.groupby(["model", "known_months", "feature", "modality"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    summary = (
        pred.groupby(["model", "known_months", "modality"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    horizon_summary = (
        by_horizon.groupby(["model", "known_months", "modality"], as_index=False)
        .agg(avg_horizon_rmse=("rmse", "mean"), worst_horizon_rmse=("rmse", "max"))
    )
    summary = summary.merge(horizon_summary, on=["model", "known_months", "modality"], how="left")

    lag1_ref = summary[summary["model"].eq("naive_lag1")][["known_months", "modality", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_ref = summary[summary["model"].eq("seasonal_last_year")][["known_months", "modality", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    summary = summary.merge(lag1_ref, on=["known_months", "modality"], how="left")
    summary = summary.merge(seasonal_ref, on=["known_months", "modality"], how="left")
    summary["beats_lag1"] = summary["rmse"] < summary["lag1_rmse"]
    summary["beats_seasonal_last_year"] = summary["rmse"] < summary["seasonal_last_year_rmse"]

    lag1_feat = feature_metrics[feature_metrics["model"].eq("naive_lag1")][["known_months", "feature", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_feat = feature_metrics[feature_metrics["model"].eq("seasonal_last_year")][["known_months", "feature", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    feature_metrics = feature_metrics.merge(lag1_feat, on=["known_months", "feature"], how="left")
    feature_metrics = feature_metrics.merge(seasonal_feat, on=["known_months", "feature"], how="left")
    feature_metrics["beats_lag1"] = feature_metrics["rmse"] < feature_metrics["lag1_rmse"]
    feature_metrics["beats_seasonal_last_year"] = feature_metrics["rmse"] < feature_metrics["seasonal_last_year_rmse"]
    return by_horizon, by_month, summary, feature_metrics


def infer_train_feature_stats(monthly: pd.DataFrame, feature_cols: list[str], train_years: list[int]) -> pd.DataFrame:
    train = monthly[monthly["year"].isin(train_years)].copy()
    rows = []
    for feature in feature_cols:
        series = train[feature].dropna().astype(float)
        std = float(series.std(ddof=1)) if len(series) else float("nan")
        mn = float(series.min()) if len(series) else float("nan")
        mx = float(series.max()) if len(series) else float("nan")
        rows.append(
            {
                "feature": feature,
                "modality": feature_to_modality(feature),
                "train_std": std,
                "train_min": mn,
                "train_max": mx,
                "train_range": mx - mn if np.isfinite(mn) and np.isfinite(mx) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def overall_metric_from_summary(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for known_months, group in summary.groupby("known_months"):
        for model, sub in group.groupby("model"):
            weight = sub["count"].to_numpy(dtype=float)
            vals = sub[value_col].to_numpy(dtype=float)
            if value_col == "rmse":
                overall = float(np.sqrt(np.sum((vals ** 2) * weight) / np.sum(weight)))
            else:
                overall = float(np.sum(vals * weight) / np.sum(weight))
            rows.append({"known_months": int(known_months), "model": model, value_col: overall})
    return pd.DataFrame(rows)


def load_component_predictions(run_dir: Path, prefix: str, source_model: str, display_model: str) -> pd.DataFrame:
    pred_path = run_dir / "artifacts" / f"{prefix}_predictions_long.csv"
    predictions = pd.read_csv(pred_path)
    component = predictions[predictions["model"].astype(str).eq(source_model)].copy()
    if component.empty:
        raise RuntimeError(f"No prediction rows found for model={source_model} in {pred_path}")
    component["model"] = display_model
    component["source_model"] = source_model
    component["source_run"] = run_dir.name
    return component


def load_validation_rmse(run_dir: Path, source_model: str, display_model: str) -> pd.DataFrame:
    metrics_path = run_dir / "artifacts" / "model_metrics_by_feature.csv"
    if not metrics_path.exists():
        return pd.DataFrame(columns=["model", "feature", "val_rmse"])
    metrics = pd.read_csv(metrics_path)
    subset = metrics[(metrics["split"].eq("val")) & (metrics["model"].astype(str).eq(source_model))][["feature", "rmse"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["model", "feature", "val_rmse"])
    subset["model"] = display_model
    subset = subset.rename(columns={"rmse": "val_rmse"})
    return subset[["model", "feature", "val_rmse"]]


def config_list(config: dict, *keys: str) -> list[int] | None:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        return [int(v) for v in value]
    return None


def config_int(config: dict, *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if value is None:
            continue
        return int(value)
    return None


def validate_run_config(config: dict, *, target_mode: str | None, feature_group: str, seq_len: int, max_counties: int, years: list[int], train_years: list[int], val_years: list[int], test_years: list[int]) -> None:
    if target_mode is not None and config.get("target_mode") != target_mode:
        raise RuntimeError(f"Unexpected target_mode {config.get('target_mode')} expected {target_mode}")
    feature_value = str(config.get("feature_group", config.get("feature_groups", "all")))
    if feature_value != feature_group:
        raise RuntimeError(f"Unexpected feature_group {feature_value} expected {feature_group}")
    cfg_seq_len = config_int(config, "seq_len", "lookback_months")
    if cfg_seq_len != seq_len:
        raise RuntimeError(f"Unexpected seq_len {cfg_seq_len} expected {seq_len}")
    cfg_max_counties = config_int(config, "max_counties", "max_auto_counties")
    if cfg_max_counties != max_counties:
        raise RuntimeError(f"Unexpected max_counties {cfg_max_counties} expected {max_counties}")
    cfg_years = config_list(config, "years")
    if cfg_years != years:
        raise RuntimeError(f"Unexpected years {cfg_years} expected {years}")
    cfg_train_years = config_list(config, "train_years")
    if cfg_train_years != train_years:
        raise RuntimeError(f"Unexpected train_years {cfg_train_years} expected {train_years}")
    cfg_val_years = config_list(config, "val_years")
    if cfg_val_years != val_years:
        raise RuntimeError(f"Unexpected val_years {cfg_val_years} expected {val_years}")
    cfg_test_years = config_list(config, "test_years")
    if cfg_test_years != test_years:
        raise RuntimeError(f"Unexpected test_years {cfg_test_years} expected {test_years}")


def make_known1_comparison_plots(summary: pd.DataFrame, normalized: pd.DataFrame, feature_metrics: pd.DataFrame, by_horizon: pd.DataFrame, monthly: pd.DataFrame, predictions: pd.DataFrame, plot_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if plt is None:
        return outputs

    known1_all = summary[summary["known_months"].eq(1) & summary["modality"].eq("all")].copy()
    if not known1_all.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        rows = known1_all.sort_values("rmse")
        ax.bar(rows["model"], rows["rmse"])
        ax.set_title("known_months=1 overall raw RMSE")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        path = plot_dir / "known1_raw_rmse_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    norm_known1_all = normalized[normalized["known_months"].eq(1) & normalized["modality"].eq("all")].copy()
    if not norm_known1_all.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        rows = norm_known1_all.sort_values("nrmse_std")
        ax.bar(rows["model"], rows["nrmse_std"])
        ax.set_title("known_months=1 overall normalized RMSE")
        ax.set_ylabel("nRMSE_std")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        path = plot_dir / "known1_normalized_rmse_comparison.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    known1_features = feature_metrics[feature_metrics["known_months"].eq(1)].copy()
    if not known1_features.empty:
        winners = (
            known1_features.sort_values(["feature", "rmse", "model"])
            .groupby("model", as_index=False)
            .size()
            .rename(columns={"size": "feature_win_count"})
            .sort_values("feature_win_count", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(winners["model"], winners["feature_win_count"])
        ax.set_title("known_months=1 feature winner counts")
        ax.set_ylabel("feature wins by raw RMSE")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        path = plot_dir / "known1_feature_winner_counts.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    horizon1 = by_horizon[by_horizon["known_months"].eq(1) & by_horizon["modality"].eq("all")].copy()
    if not horizon1.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for model, group in horizon1.groupby("model"):
            group = group.sort_values("horizon")
            ax.plot(group["horizon"], group["rmse"], marker="o", label=model)
        ax.set_title("known_months=1 overall raw RMSE by horizon")
        ax.set_xlabel("horizon")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = plot_dir / "known1_horizon_rmse.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    year_df = monthly[monthly["year"].eq(2021)].copy()
    pred_known1 = predictions[predictions["known_months"].eq(1)].copy()
    overlay_models = [model for model in OVERLAY_MODELS if model in pred_known1["model"].unique()]
    for feature in SELECTED_FEATURES:
        truth = year_df.groupby("month", as_index=False)[feature].mean()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(truth["month"], truth[feature], marker="o", linewidth=2, label="truth")
        for model in overlay_models:
            pred_avg = (
                pred_known1[(pred_known1["model"].eq(model)) & (pred_known1["feature"].eq(feature))]
                .groupby("target_month", as_index=False)["y_pred"]
                .mean()
            )
            if pred_avg.empty:
                continue
            ax.plot(pred_avg["target_month"], pred_avg["y_pred"], marker="o", linestyle="--", label=model)
        ax.set_title(f"known_months=1 overlay | {feature}")
        ax.set_xlabel("month")
        ax.set_ylabel(feature)
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = plot_dir / f"overlay_known1_{safe_slug(feature)}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    return outputs


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def package_outputs(artifacts: Path, plot_dir: Path, zip_name: str) -> Path:
    zip_path = artifacts / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(plot_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(artifacts))
        for path in sorted(artifacts.glob("broader_ensemble_*")):
            if path.is_file():
                zf.write(path, path.relative_to(artifacts))
        for name in [
            "strict_blank_fill_broader_ensemble_full30_metrics_summary.csv",
            "strict_blank_fill_broader_ensemble_full30_metrics_by_horizon.csv",
            "strict_blank_fill_broader_ensemble_full30_metrics_by_month.csv",
            "strict_blank_fill_broader_ensemble_full30_feature_metrics.csv",
            "strict_blank_fill_broader_ensemble_full30_normalized_metrics_summary.csv",
            "strict_blank_fill_broader_ensemble_full30_normalized_metrics_by_modality.csv",
            "strict_blank_fill_broader_ensemble_full30_normalized_metrics_by_feature.csv",
            "strict_blank_fill_broader_ensemble_full30_known1_model_comparison_raw_and_normalized.md",
            "strict_blank_fill_broader_ensemble_full30_known1_feature_winners.md",
            "strict_blank_fill_broader_ensemble_full30_known_month_sensitivity_summary.md",
            "strict_blank_fill_broader_ensemble_full30_visual_diagnosis_report.md",
        ]:
            path = artifacts / name
            if path.exists():
                zf.write(path, path.relative_to(artifacts))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a broader cross-run blank-fill ensemble benchmark.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--base-run-dir", default="outputs/experiments/ia30_seq6_seasonal_residual_blankfill")
    parser.add_argument("--base-prefix", default="strict_blank_fill_sarima_ensemble_full30")
    parser.add_argument("--gru-run-dir", default="outputs/experiments/model_comparison_v1/runs/ia30_seq6_gru_seasonal_residual_rawmse")
    parser.add_argument("--transformer-run-dir", default="outputs/experiments/model_comparison_v1/runs/ia30_seq6_transformer_encoder_seasonal_residual_rawmse")
    parser.add_argument("--component-prefix", default="strict_blank_fill")
    parser.add_argument("--output-prefix", default="strict_blank_fill_broader_ensemble_full30")
    parser.add_argument("--python-bin", default=sys.executable)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    base_run_dir = (project_root / args.base_run_dir).resolve()
    gru_run_dir = (project_root / args.gru_run_dir).resolve()
    transformer_run_dir = (project_root / args.transformer_run_dir).resolve()
    artifacts = base_run_dir / "artifacts"

    base_config = load_json(base_run_dir / "config.json")
    expected_years = [2017, 2018, 2019, 2020, 2021]
    expected_train = [2017, 2018, 2019]
    expected_val = [2020]
    expected_test = [2021]
    validate_run_config(
        base_config,
        target_mode="seasonal_residual",
        feature_group="all",
        seq_len=6,
        max_counties=30,
        years=expected_years,
        train_years=expected_train,
        val_years=expected_val,
        test_years=expected_test,
    )

    components = [
        {
            "run_dir": base_run_dir,
            "prefix": args.base_prefix,
            "source_model": "naive_lag1",
            "display_model": "naive_lag1",
            "target_mode": "baseline",
            "loss_mode": "n/a",
            "deployable": True,
            "include_mean": False,
            "include_weighted": False,
        },
        {
            "run_dir": base_run_dir,
            "prefix": args.base_prefix,
            "source_model": "seasonal_last_year",
            "display_model": "seasonal_last_year",
            "target_mode": "baseline",
            "loss_mode": "n/a",
            "deployable": True,
            "include_mean": True,
            "include_weighted": True,
        },
        {
            "run_dir": base_run_dir,
            "prefix": args.base_prefix,
            "source_model": "lstm",
            "display_model": "LSTM seasonal_residual",
            "target_mode": "seasonal_residual",
            "loss_mode": base_config.get("loss_mode", "raw_mse"),
            "deployable": True,
            "include_mean": True,
            "include_weighted": True,
        },
        {
            "run_dir": base_run_dir,
            "prefix": args.base_prefix,
            "source_model": "tiny_mamba_ssm",
            "display_model": "tiny_mamba_ssm seasonal_residual",
            "target_mode": "seasonal_residual",
            "loss_mode": base_config.get("loss_mode", "raw_mse"),
            "deployable": True,
            "include_mean": True,
            "include_weighted": True,
        },
        {
            "run_dir": base_run_dir,
            "prefix": args.base_prefix,
            "source_model": "sarima",
            "display_model": "SARIMA",
            "target_mode": "classical_seasonal",
            "loss_mode": "n/a",
            "deployable": True,
            "include_mean": True,
            "include_weighted": False,
        },
    ]

    for run_dir, source_model, display_model in [
        (gru_run_dir, "gru", "GRU seasonal_residual"),
        (transformer_run_dir, "transformer_encoder", "transformer_encoder seasonal_residual"),
    ]:
        config = load_json(run_dir / "config.json")
        validate_run_config(
            config,
            target_mode="seasonal_residual",
            feature_group="all",
            seq_len=6,
            max_counties=30,
            years=expected_years,
            train_years=expected_train,
            val_years=expected_val,
            test_years=expected_test,
        )
        components.append(
            {
                "run_dir": run_dir,
                "prefix": args.component_prefix,
                "source_model": source_model,
                "display_model": display_model,
                "target_mode": "seasonal_residual",
                "loss_mode": config.get("loss_mode", "raw_mse"),
                "deployable": True,
                "include_mean": True,
                "include_weighted": True,
            }
        )

    component_rows = []
    val_weight_rows = []
    variant_rows = []
    for component in components:
        predictions = load_component_predictions(
            component["run_dir"],
            component["prefix"],
            component["source_model"],
            component["display_model"],
        )
        component_rows.append(predictions)
        val_weights = load_validation_rmse(component["run_dir"], component["source_model"], component["display_model"])
        if not val_weights.empty:
            val_weight_rows.append(val_weights)
        variant_rows.append(
            {
                "display_model": component["display_model"],
                "source_run_dir": str(component["run_dir"]),
                "source_prefix": component["prefix"],
                "source_model": component["source_model"],
                "target_mode": component["target_mode"],
                "loss_mode": component["loss_mode"],
                "deployable": component["deployable"],
                "included_in_ensemble_mean": component["include_mean"],
                "included_in_ensemble_weighted": component["include_weighted"],
                "has_validation_weights": not val_weights.empty,
            }
        )

    combined = pd.concat(component_rows, ignore_index=True)
    combined["run_name"] = "ia30_broader_ensemble_full30"

    duplicate_count = int(combined.duplicated(subset=KEY_COLS + ["model"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Duplicate component prediction rows detected: {duplicate_count}")

    val_weights = pd.concat(val_weight_rows, ignore_index=True) if val_weight_rows else pd.DataFrame(columns=["model", "feature", "val_rmse"])

    mean_components = [row["display_model"] for row in variant_rows if row["included_in_ensemble_mean"]]
    weighted_components = [row["display_model"] for row in variant_rows if row["included_in_ensemble_weighted"]]
    mean_frame = combined[combined["model"].isin(mean_components)].copy()
    weighted_frame = combined[combined["model"].isin(weighted_components)].copy()

    if mean_frame.empty or weighted_frame.empty:
        raise RuntimeError("Not enough component predictions to build ensembles.")

    ensemble_mean = (
        mean_frame.groupby(["run_name"] + KEY_COLS, as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    )
    ensemble_mean["model"] = "ensemble_mean"
    ensemble_mean["abs_error"] = (ensemble_mean["y_pred"] - ensemble_mean["y_true"]).abs()
    ensemble_mean["squared_error"] = (ensemble_mean["y_pred"] - ensemble_mean["y_true"]) ** 2
    ensemble_mean["source_note"] = f"ensemble_mean:{','.join(mean_components)}"

    weighted_frame = weighted_frame.merge(val_weights, on=["model", "feature"], how="left")
    missing_weight_components = (
        weighted_frame[weighted_frame["val_rmse"].isna()][["model", "feature"]].drop_duplicates().sort_values(["model", "feature"])
    )
    weighted_usable = weighted_frame[weighted_frame["val_rmse"].notna()].copy()
    weighted_usable["weight"] = 1.0 / weighted_usable["val_rmse"].clip(lower=1e-6)
    weighted_usable["weight"] = weighted_usable.groupby(KEY_COLS)["weight"].transform(lambda s: s / s.sum() if float(s.sum()) > 0 else np.nan)

    ensemble_weighted = (
        weighted_usable.groupby(["run_name"] + KEY_COLS, as_index=False)
        .apply(
            lambda grp: pd.Series(
                {
                    "y_true": float(grp["y_true"].iloc[0]),
                    "y_pred": float(np.average(grp["y_pred"], weights=grp["weight"])),
                }
            )
        )
        .reset_index(drop=True)
    )
    ensemble_weighted["model"] = "ensemble_weighted"
    ensemble_weighted["abs_error"] = (ensemble_weighted["y_pred"] - ensemble_weighted["y_true"]).abs()
    ensemble_weighted["squared_error"] = (ensemble_weighted["y_pred"] - ensemble_weighted["y_true"]) ** 2
    used_weight_models = sorted(weighted_usable["model"].astype(str).unique())
    ensemble_weighted["source_note"] = f"ensemble_weighted:{','.join(used_weight_models)}"

    combined_predictions = pd.concat(
        [
            combined[["run_name", "model", "county_id", "crop_type", "blank_fill_year", "known_months", "target_month", "horizon", "feature", "modality", "y_true", "y_pred", "abs_error", "squared_error", "source_note"]],
            ensemble_mean[["run_name", "model", "county_id", "crop_type", "blank_fill_year", "known_months", "target_month", "horizon", "feature", "modality", "y_true", "y_pred", "abs_error", "squared_error", "source_note"]],
            ensemble_weighted[["run_name", "model", "county_id", "crop_type", "blank_fill_year", "known_months", "target_month", "horizon", "feature", "modality", "y_true", "y_pred", "abs_error", "squared_error", "source_note"]],
        ],
        ignore_index=True,
    )

    prefix = args.output_prefix
    pred_path = artifacts / f"{prefix}_predictions_long.csv"
    combined_predictions.to_csv(pred_path, index=False)

    by_horizon, by_month, summary, feature_metrics = compute_blank_fill_metrics(combined_predictions)
    summary["modality"] = summary["modality"].astype(str)
    by_horizon["modality"] = by_horizon["modality"].astype(str)
    by_month["modality"] = by_month["modality"].astype(str)
    feature_metrics["modality"] = feature_metrics["modality"].astype(str)
    by_horizon.to_csv(artifacts / f"{prefix}_metrics_by_horizon.csv", index=False)
    by_month.to_csv(artifacts / f"{prefix}_metrics_by_month.csv", index=False)
    summary.to_csv(artifacts / f"{prefix}_metrics_summary.csv", index=False)
    feature_metrics.to_csv(artifacts / f"{prefix}_feature_metrics.csv", index=False)

    monthly = pd.read_parquet(artifacts / "official_monthly_feature_table.parquet")
    monthly["crop_type"] = monthly.get("crop_type", "Corn")
    feature_cols = pd.read_csv(artifacts / "scaler.csv")["feature"].tolist()
    feature_stats = infer_train_feature_stats(monthly, feature_cols, expected_train)
    weight_audit = {
        "weighted_components_requested": weighted_components,
        "weighted_components_used": used_weight_models,
        "weighted_components_missing_validation_weights": sorted(missing_weight_components["model"].astype(str).unique()),
        "missing_weight_feature_rows": int(len(missing_weight_components)),
    }

    diagnostics_cmd = [
        args.python_bin,
        "analyze_blank_fill_diagnostics.py",
        "--run-dir",
        str(base_run_dir),
        "--blank-fill-prefix",
        prefix,
        "--compare-prefix",
        args.base_prefix,
    ]
    subprocess.run(diagnostics_cmd, cwd=str(project_root), check=True)

    norm_summary = pd.read_csv(artifacts / f"{prefix}_normalized_metrics_summary.csv")
    norm_feature = pd.read_csv(artifacts / f"{prefix}_normalized_metrics_by_feature.csv")
    norm_modality = pd.read_csv(artifacts / f"{prefix}_normalized_metrics_by_modality.csv")

    raw_overall = overall_metric_from_summary(summary, "rmse").merge(overall_metric_from_summary(summary, "mae"), on=["known_months", "model"], how="left")
    norm_overall = norm_summary[norm_summary["modality"].eq("all")][["known_months", "model", "nrmse_std", "nrmse_range", "pearson_corr", "r2"]].copy()
    broader_summary = raw_overall.merge(norm_overall, on=["known_months", "model"], how="left")
    lag1_wins = feature_metrics.groupby(["known_months", "model"], as_index=False)["beats_lag1"].sum().rename(columns={"beats_lag1": "feature_wins_vs_lag1"})
    seasonal_wins = feature_metrics.groupby(["known_months", "model"], as_index=False)["beats_seasonal_last_year"].sum().rename(columns={"beats_seasonal_last_year": "feature_wins_vs_seasonal_last_year"})
    broader_summary = broader_summary.merge(lag1_wins, on=["known_months", "model"], how="left").merge(seasonal_wins, on=["known_months", "model"], how="left")
    broader_summary.to_csv(artifacts / "broader_ensemble_summary.csv", index=False)

    known1_table = norm_modality[norm_modality["known_months"].eq(1)][["model", "modality", "rmse", "nrmse_std", "mae", "pearson_corr", "r2", "beats_seasonal_last_year"]].sort_values(["modality", "nrmse_std", "rmse"])
    write_markdown(artifacts / "broader_ensemble_known1_model_comparison.md", "# Broader Ensemble Known-Months=1 Comparison\n\n" + known1_table.to_markdown(index=False) + "\n")

    feature_rows = []
    known1_features = norm_feature[norm_feature["known_months"].eq(1)].copy()
    for feature, group in known1_features.groupby("feature"):
        best_raw = group.sort_values(["rmse", "model"]).iloc[0]
        best_norm = group.sort_values(["nrmse_std", "model"]).iloc[0]
        feature_rows.append(
            {
                "feature": feature,
                "modality": feature_to_modality(feature),
                "best_model_raw_rmse": best_raw["model"],
                "best_raw_rmse": best_raw["rmse"],
                "best_model_normalized_rmse": best_norm["model"],
                "best_normalized_rmse": best_norm["nrmse_std"],
            }
        )
    feature_winners = pd.DataFrame(feature_rows).sort_values(["modality", "feature"])
    write_markdown(artifacts / "broader_ensemble_feature_winners.md", "# Broader Ensemble Feature Winners\n\n" + feature_winners.to_markdown(index=False) + "\n")

    model_variant_frame = pd.DataFrame(variant_rows)
    model_variant_frame["weighted_excluded_reason"] = np.where(
        model_variant_frame["included_in_ensemble_weighted"] & ~model_variant_frame["has_validation_weights"],
        "missing validation RMSE; excluded from deployable weighted ensemble",
        "",
    )
    write_markdown(artifacts / "broader_ensemble_model_variant_table.md", "# Broader Ensemble Model Variant Table\n\n" + model_variant_frame.to_markdown(index=False) + "\n")

    known1_overall = broader_summary[broader_summary["known_months"].eq(1)].sort_values("overall_rmse")
    visual_lines = [
        "# Broader Ensemble Visual Diagnosis Report",
        "",
        "## Weighting Audit",
        "",
        f"- Weighted components requested: {', '.join(weighted_components)}",
        f"- Weighted components actually used: {', '.join(used_weight_models)}",
        f"- Components excluded from weighted ensemble due to missing validation weights: {', '.join(weight_audit['weighted_components_missing_validation_weights']) or 'none'}",
        f"- Missing validation weight feature rows: {weight_audit['missing_weight_feature_rows']}",
        "",
        "## known_months=1 Overall Ranking",
        "",
        known1_overall.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Ensemble weighting in this report uses validation metrics only.",
        "- SARIMA is included as a standalone classical comparator and in the simple mean ensemble.",
        "- SARIMA is excluded from the weighted deployable ensemble if validation weights are unavailable.",
    ]
    write_markdown(artifacts / "broader_ensemble_visual_diagnosis_report.md", "\n".join(visual_lines) + "\n")

    plot_dir = artifacts / "plots_broader_ensemble_full30"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = make_known1_comparison_plots(summary, norm_summary, feature_metrics, by_horizon, monthly, combined_predictions, plot_dir)

    zip_path = package_outputs(artifacts, plot_dir, "broader_ensemble_full30_report.zip")

    print(base_run_dir)
    print(pred_path)
    print(artifacts / "broader_ensemble_summary.csv")
    print(artifacts / "broader_ensemble_known1_model_comparison.md")
    print(artifacts / "broader_ensemble_feature_winners.md")
    print(artifacts / "broader_ensemble_visual_diagnosis_report.md")
    print(artifacts / "broader_ensemble_model_variant_table.md")
    print(zip_path)
    print(f"plot_count={len(plot_paths)}")


if __name__ == "__main__":
    main()
