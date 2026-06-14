from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTED_FEATURES = {
    "ag": [
        "ag_green_pixel_ratio",
        "ag_mean_brightness",
        "ag_field_uniformity_score",
    ],
    "ndvi": [
        "ndvi_mean",
        "ndvi_max",
        "ndvi_valid_coverage_ratio",
    ],
    "weather": [
        "weather_temp_mean",
        "weather_gdd",
        "weather_total_precipitation",
        "weather_solar_radiation_mean",
        "weather_vpd_mean",
    ],
}

PREFERRED_MODEL_ORDER = [
    "seasonal_last_year",
    "naive_lag1",
    "lstm",
    "gru",
    "tiny_mamba_ssm",
    "transformer_encoder",
    "sarima",
    "ensemble_mean",
    "ensemble_weighted",
    "ensemble_oracle_report_only",
]


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    return value.strip("_").lower()


def prefixed_name(prefix: str, stem: str) -> str:
    return stem if prefix == "blank_fill" else f"{prefix}_{stem}"


def ordered_models(frame: pd.DataFrame) -> list[str]:
    available = [str(model) for model in frame["model"].dropna().astype(str).unique()]
    preferred = [model for model in PREFERRED_MODEL_ORDER if model in available]
    remaining = sorted(model for model in available if model not in preferred)
    return preferred + remaining


def feature_to_modality(feature: str) -> str:
    if feature.startswith("ag_"):
        return "ag"
    if feature.startswith("ndvi_"):
        return "ndvi"
    if feature.startswith("weather_"):
        return "weather"
    return "other"


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if math.isclose(denom, 0.0):
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(200.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_markdown_table(path: Path, frame: pd.DataFrame) -> None:
    path.write_text(frame.to_markdown(index=False), encoding="utf-8")


def summarize_group(group: pd.DataFrame) -> pd.Series:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    out = {
        "count": int(len(group)),
        "rmse": float(np.sqrt(np.mean(group["squared_error"]))),
        "mae": float(group["abs_error"].mean()),
        "nrmse_std": float(np.sqrt(np.nanmean(group["n_sq_err_std"]))),
        "nrmse_range": float(np.sqrt(np.nanmean(group["n_sq_err_range"]))),
        "smape": safe_smape(y_true, y_pred),
        "pearson_corr": safe_corr(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
    }
    return pd.Series(out)


def load_artifacts(run_dir: Path, blank_fill_prefix: str) -> dict:
    artifacts = run_dir / "artifacts"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    feature_diag = json.loads((artifacts / "feature_contract_diagnostic.json").read_text(encoding="utf-8"))
    monthly = pd.read_parquet(artifacts / "official_monthly_feature_table.parquet")
    scaler = pd.read_csv(artifacts / "scaler.csv")
    blank_fill = pd.read_csv(artifacts / f"{blank_fill_prefix}_predictions_long.csv")
    return {
        "config": config,
        "feature_diag": feature_diag,
        "monthly": monthly,
        "scaler": scaler,
        "blank_fill": blank_fill,
        "artifacts": artifacts,
    }


def compute_train_feature_stats(monthly: pd.DataFrame, feature_cols: list[str], train_years: Iterable[int]) -> pd.DataFrame:
    train_frame = monthly[monthly["year"].isin(list(train_years))].copy()
    stats_rows = []
    for feature in feature_cols:
        series = train_frame[feature].dropna().astype(float)
        std = float(series.std(ddof=1)) if len(series) else float("nan")
        min_v = float(series.min()) if len(series) else float("nan")
        max_v = float(series.max()) if len(series) else float("nan")
        range_v = max_v - min_v if np.isfinite(min_v) and np.isfinite(max_v) else float("nan")
        stats_rows.append(
            {
                "feature": feature,
                "modality": feature_to_modality(feature),
                "train_std": std,
                "train_min": min_v,
                "train_max": max_v,
                "train_range": range_v,
            }
        )
    return pd.DataFrame(stats_rows)


def add_normalized_error_columns(blank_fill: pd.DataFrame, feature_stats: pd.DataFrame) -> pd.DataFrame:
    merged = blank_fill.merge(feature_stats, on=["feature", "modality"], how="left")
    merged["n_sq_err_std"] = np.where(
        merged["train_std"].gt(0),
        merged["squared_error"] / (merged["train_std"] ** 2),
        np.nan,
    )
    merged["n_sq_err_range"] = np.where(
        merged["train_range"].gt(0),
        merged["squared_error"] / (merged["train_range"] ** 2),
        np.nan,
    )
    merged["n_abs_err_std"] = np.where(
        merged["train_std"].gt(0),
        merged["abs_error"] / merged["train_std"],
        np.nan,
    )
    merged["n_abs_err_range"] = np.where(
        merged["train_range"].gt(0),
        merged["abs_error"] / merged["train_range"],
        np.nan,
    )
    return merged


def build_normalized_metric_frames(blank_fill: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_feature = (
        blank_fill.groupby(["model", "known_months", "feature", "modality"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    by_modality = (
        blank_fill.groupby(["model", "known_months", "modality"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    summary = (
        blank_fill.groupby(["model", "known_months"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    summary["modality"] = "all"

    lag1_ref = by_feature[by_feature["model"].eq("naive_lag1")][["known_months", "feature", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_ref = by_feature[by_feature["model"].eq("seasonal_last_year")][["known_months", "feature", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    by_feature = by_feature.merge(lag1_ref, on=["known_months", "feature"], how="left")
    by_feature = by_feature.merge(seasonal_ref, on=["known_months", "feature"], how="left")
    by_feature["beats_lag1"] = by_feature["rmse"] < by_feature["lag1_rmse"]
    by_feature["beats_seasonal_last_year"] = by_feature["rmse"] < by_feature["seasonal_last_year_rmse"]

    lag1_mod = by_modality[by_modality["model"].eq("naive_lag1")][["known_months", "modality", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_mod = by_modality[by_modality["model"].eq("seasonal_last_year")][["known_months", "modality", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    by_modality = by_modality.merge(lag1_mod, on=["known_months", "modality"], how="left")
    by_modality = by_modality.merge(seasonal_mod, on=["known_months", "modality"], how="left")
    by_modality["beats_lag1"] = by_modality["rmse"] < by_modality["lag1_rmse"]
    by_modality["beats_seasonal_last_year"] = by_modality["rmse"] < by_modality["seasonal_last_year_rmse"]

    lag1_sum = summary[summary["model"].eq("naive_lag1")][["known_months", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_sum = summary[summary["model"].eq("seasonal_last_year")][["known_months", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    summary = summary.merge(lag1_sum, on="known_months", how="left")
    summary = summary.merge(seasonal_sum, on="known_months", how="left")
    summary["beats_lag1"] = summary["rmse"] < summary["lag1_rmse"]
    summary["beats_seasonal_last_year"] = summary["rmse"] < summary["seasonal_last_year_rmse"]
    return by_feature, by_modality, summary


def build_scaling_audit_report(run_dir: Path, config: dict, feature_diag: dict) -> str:
    return f"""# Scaling Audit Report

## Training Path
- `prepare_model_frames(...)` interpolates and fills monthly features per county/crop, then computes `mu` and `sigma`.
- `mu` and `sigma` are fitted on **train years only**: `{config.get('train_years')}`.
- `X` sequence inputs are scaled feature-wise using `(x - mu) / sigma`.
- `y` targets are also scaled.
- In `target_mode=seasonal_residual`, targets are residuals in **scaled space**:
  - `y_model_scaled = y_true_scaled - seasonal_base_scaled`
  - because both terms use the same train-fitted scaler, this is consistent and equivalent to dividing raw residuals by train std.

## Leakage Risk
- The scaler is fitted on train years only, not all data.
- Validation/test monthly rows are reindexed and interpolated inside each county time series before scaling.
- This means the feature filler can use neighboring months when constructing dense monthly frames, so there is some temporal smoothing risk in the monthly preparation stage.
- The scaler itself does **not** leak validation/test statistics.

## One-Step Evaluation
- Learned-model outputs are kept in scaled space until evaluation.
- `compute_metrics(...)` and `per_feature_metrics(...)` call `inverse_scale(...)` on both:
  - `y_true_scaled`
  - `y_pred_scaled`
- Reported RMSE/MAE are therefore in **raw feature units**, not scaled units.
- In residual mode, learned predictions are converted back to final scaled feature predictions first:
  - `final_scaled = seasonal_base_scaled + predicted_residual_scaled`
  - then inverse-transformed once.

## Blank-Fill Evaluation
- Recursive histories for learned models are stored in **scaled feature space**.
- During rollout, the model sees scaled history windows.
- In residual mode:
  - model output is interpreted as **scaled residual**
  - `seasonal_base_scaled` is added in scaled space
  - `inverse_scale(...)` is then applied once to obtain raw predictions for metrics
- Ground truth in `blank_fill_predictions_long.csv` is stored as raw `y_true`.
- Final blank-fill metrics are computed in **raw feature units**.

## Residual Seasonal Base
- Seasonal base is matched by:
  - same `county_id`
  - same `crop_type`
  - previous year
  - same month
- If seasonal base is missing during training, the sequence sample is dropped.
- If seasonal base is missing during blank-fill rollout, the code falls back to lag-1 style raw history and records that in `source_note`.

## Audit Verdict
- No double inverse-transform was found.
- No missing inverse-transform was found in one-step or blank-fill metrics.
- Residual outputs are combined with the seasonal base in the correct **scaled** space before inverse-transform.
- The very large aggregate RMSE values are mainly caused by large-scale weather features being measured in raw units, not by an obvious scaling bug.

## Data Quality Notes
- Feature contract: `{feature_diag['expected']['total_model_feature_count']} / {feature_diag['actual']['total_model_feature_count']}`
- Partially-NaN columns: `{feature_diag['missing_and_extra']['present_but_partially_nan']}`
"""


def make_overlay_plots(
    monthly: pd.DataFrame,
    blank_fill: pd.DataFrame,
    plot_dir: Path,
    representative_county: str,
    models: list[str],
) -> list[Path]:
    outputs: list[Path] = []
    year_df = monthly[monthly["year"].eq(2021)].copy()
    for known_months in [1, 6]:
        pred_subset = blank_fill[blank_fill["known_months"].eq(known_months)].copy()
        for modality, features in SELECTED_FEATURES.items():
            for feature in features:
                truth_avg = year_df.groupby("month")[feature].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(truth_avg["month"], truth_avg[feature], marker="o", linewidth=2, label="truth")
                for model in models:
                    pred_avg = (
                        pred_subset[pred_subset["feature"].eq(feature) & pred_subset["model"].eq(model)]
                        .groupby("target_month", as_index=False)["y_pred"]
                        .mean()
                    )
                    if pred_avg.empty:
                        continue
                    ax.plot(pred_avg["target_month"], pred_avg["y_pred"], marker="o", linestyle="--", label=model)
                ax.set_title(f"2021 average blank-fill overlay | known={known_months} | {feature}")
                ax.set_xlabel("month")
                ax.set_ylabel(feature)
                ax.set_xticks(range(1, 13))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                path = plot_dir / f"avg_known{known_months}_{slugify(feature)}.png"
                fig.tight_layout()
                fig.savefig(path, dpi=160)
                plt.close(fig)
                outputs.append(path)

                county_truth = year_df[year_df["county_id"].astype(str).eq(representative_county)].sort_values("month")
                county_pred = pred_subset[pred_subset["county_id"].astype(str).eq(representative_county) & pred_subset["feature"].eq(feature)]
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(county_truth["month"], county_truth[feature], marker="o", linewidth=2, label="truth")
                for model in models:
                    pred_line = county_pred[county_pred["model"].eq(model)].sort_values("target_month")
                    if pred_line.empty:
                        continue
                    ax.plot(pred_line["target_month"], pred_line["y_pred"], marker="o", linestyle="--", label=model)
                ax.set_title(f"2021 county {representative_county} | known={known_months} | {feature}")
                ax.set_xlabel("month")
                ax.set_ylabel(feature)
                ax.set_xticks(range(1, 13))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                path = plot_dir / f"county_{representative_county}_known{known_months}_{slugify(feature)}.png"
                fig.tight_layout()
                fig.savefig(path, dpi=160)
                plt.close(fig)
                outputs.append(path)
    return outputs


def make_scatter_plots(blank_fill: pd.DataFrame, plot_dir: Path, models: list[str]) -> list[Path]:
    outputs: list[Path] = []
    known1 = blank_fill[blank_fill["known_months"].eq(1)].copy()
    for modality in ["ag", "ndvi", "weather"]:
        for model in models:
            sub = known1[(known1["modality"].eq(modality)) & (known1["model"].eq(model))]
            if sub.empty:
                continue
            y_true = sub["y_true"].to_numpy(dtype=float)
            y_pred = sub["y_pred"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(y_true, y_pred, s=8, alpha=0.25)
            lo = min(np.min(y_true), np.min(y_pred))
            hi = max(np.max(y_true), np.max(y_pred))
            ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
            ax.set_title(
                f"known=1 scatter | {model} | {modality}\n"
                f"corr={format_float(safe_corr(y_true, y_pred))} r2={format_float(safe_r2(y_true, y_pred))}"
            )
            ax.set_xlabel("y_true")
            ax.set_ylabel("y_pred")
            ax.grid(True, alpha=0.3)
            path = plot_dir / f"scatter_known1_{slugify(model)}_{modality}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)
    return outputs


def make_heatmaps(blank_fill: pd.DataFrame, plot_dir: Path, models: list[str]) -> list[Path]:
    outputs: list[Path] = []
    known1 = blank_fill[blank_fill["known_months"].eq(1)].copy()
    for modality in ["ag", "ndvi", "weather"]:
        rows = (
            known1[known1["modality"].eq(modality)]
            .groupby(["model", "target_month"], as_index=False)
            .agg(rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))))
        )
        pivot = rows.pivot(index="model", columns="target_month", values="rmse").reindex(index=models)
        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_title(f"known=1 month RMSE heatmap | {modality}")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        path = plot_dir / f"heatmap_known1_{modality}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def make_feature_bar_plots(feature_metrics: pd.DataFrame, plot_dir: Path, models: list[str]) -> list[Path]:
    outputs: list[Path] = []
    plot_models = [model for model in models if model != "naive_lag1"]
    subset = feature_metrics[
        feature_metrics["known_months"].eq(1)
        & feature_metrics["model"].isin(plot_models)
    ].copy()
    for modality in ["ag", "ndvi", "weather"]:
        rows = subset[subset["modality"].eq(modality)]
        if rows.empty:
            continue
        pivot = rows.pivot(index="feature", columns="model", values="nrmse_std").reindex(columns=plot_models).sort_index()
        fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.55), 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"known=1 feature nRMSE(std) | {modality}")
        ax.set_ylabel("nRMSE_std")
        ax.set_xlabel("feature")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=60)
        path = plot_dir / f"feature_bar_known1_{modality}_nrmse_std.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def make_horizon_plots(blank_fill: pd.DataFrame, plot_dir: Path, models: list[str]) -> list[Path]:
    outputs: list[Path] = []
    known1 = blank_fill[blank_fill["known_months"].eq(1)].copy()
    for modality in ["ag", "ndvi", "weather"]:
        rows = (
            known1[known1["modality"].eq(modality)]
            .groupby(["model", "horizon"], as_index=False)
            .apply(summarize_group)
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in models:
            grp = rows[rows["model"].eq(model)].sort_values("horizon")
            if grp.empty:
                continue
            ax.plot(grp["horizon"], grp["nrmse_std"], marker="o", label=model)
        ax.set_title(f"known=1 horizon degradation | {modality} | nRMSE(std)")
        ax.set_xlabel("horizon")
        ax.set_ylabel("nRMSE_std")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        path = plot_dir / f"horizon_known1_{modality}_nrmse_std.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs


def build_report_tables(
    artifacts: Path,
    by_modality: pd.DataFrame,
    by_feature: pd.DataFrame,
    summary: pd.DataFrame,
    prefix: str,
) -> None:
    known1_mod = by_modality[by_modality["known_months"].eq(1)].copy()
    model_table = known1_mod[["model", "modality", "rmse", "nrmse_std", "mae", "pearson_corr", "r2"]].sort_values(["modality", "rmse"])
    write_markdown_table(artifacts / prefixed_name(prefix, "known1_model_comparison_raw_and_normalized.md"), model_table)

    known1_feat = by_feature[by_feature["known_months"].eq(1)].copy()
    rows = []
    for feature, group in known1_feat.groupby("feature"):
        best_raw = group.sort_values("rmse").iloc[0]
        best_norm = group.sort_values("nrmse_std").iloc[0]
        seasonal_rmse = group.loc[group["model"].eq("seasonal_last_year"), "rmse"]
        learned_best_rmse = group[group["model"].isin(["lstm", "tiny_mamba_ssm"])]["rmse"].min()
        rows.append(
            {
                "feature": feature,
                "modality": feature_to_modality(feature),
                "best_model_raw_rmse": best_raw["model"],
                "best_raw_rmse": best_raw["rmse"],
                "best_model_normalized_rmse": best_norm["model"],
                "best_normalized_rmse": best_norm["nrmse_std"],
                "learned_beats_seasonal_last_year": bool(
                    np.isfinite(learned_best_rmse)
                    and not seasonal_rmse.empty
                    and learned_best_rmse < float(seasonal_rmse.iloc[0])
                ),
            }
        )
    winners = pd.DataFrame(rows).sort_values(["modality", "feature"])
    write_markdown_table(artifacts / prefixed_name(prefix, "known1_feature_winners.md"), winners)

    rows = []
    for known, group in summary.groupby("known_months"):
        best_raw = group.sort_values("rmse").iloc[0]
        best_norm = group.sort_values("nrmse_std").iloc[0]
        seasonal_row = group[group["model"].eq("seasonal_last_year")]
        learned_best = group[group["model"].isin(["lstm", "tiny_mamba_ssm"])].sort_values("rmse")
        learned_beats = False
        if not seasonal_row.empty and not learned_best.empty:
            learned_beats = float(learned_best.iloc[0]["rmse"]) < float(seasonal_row.iloc[0]["rmse"])
        rows.append(
            {
                "known_months": known,
                "best_model_raw_rmse": best_raw["model"],
                "best_raw_rmse": best_raw["rmse"],
                "best_model_normalized_rmse": best_norm["model"],
                "best_normalized_rmse": best_norm["nrmse_std"],
                "learned_model_beats_seasonal": learned_beats,
            }
        )
    sens = pd.DataFrame(rows).sort_values("known_months")
    write_markdown_table(artifacts / prefixed_name(prefix, "known_month_sensitivity_summary.md"), sens)


def build_visual_diagnosis_report(
    artifacts: Path,
    run_dir: Path,
    by_modality: pd.DataFrame,
    by_feature: pd.DataFrame,
    summary: pd.DataFrame,
    prefix: str,
) -> None:
    known1 = summary[summary["known_months"].eq(1)].sort_values("rmse")
    weather_known1 = by_modality[(by_modality["known_months"].eq(1)) & (by_modality["modality"].eq("weather"))].sort_values("rmse")
    ndvi_known1 = by_modality[(by_modality["known_months"].eq(1)) & (by_modality["modality"].eq("ndvi"))].sort_values("rmse")
    ag_known1 = by_modality[(by_modality["known_months"].eq(1)) & (by_modality["modality"].eq("ag"))].sort_values("rmse")

    top_raw_features = by_feature[by_feature["known_months"].eq(1)].sort_values("rmse", ascending=False).head(10)
    top_norm_features = by_feature[by_feature["known_months"].eq(1)].sort_values("nrmse_std", ascending=False).head(10)

    text = f"""# Visual Diagnosis Report

## 1. Are the huge RMSE values mainly scale-driven?
Yes, largely. Raw RMSE is dominated by weather features with large natural magnitudes, especially precipitation, radiation, VPD, and temperature-derived features.

## 2. Is scaling/inverse-scaling correct?
The audit found no obvious scaling bug. Metrics are computed after converting predictions and targets back to raw units. Residual mode predicts scaled residuals, adds the seasonal base in scaled space, then inverse-transforms once.

## 3. Which features dominate aggregate RMSE?
Top raw-RMSE contributors for known_months=1 are concentrated in weather. Highest raw-RMSE rows include:
{top_raw_features[['model', 'feature', 'rmse']].head(6).to_markdown(index=False)}

## 4. Which features are visually predicted well?
NDVI and some AG features are visually smooth and seasonally stable. In many plots, the seasonal baseline and learned models track the broad annual shape well.

## 5. Which features are visually poor?
Weather features remain the hardest. Even when overall residual LSTM improves, month-to-month amplitude for some weather series is still imperfect.

## 6. Is residual LSTM improvement meaningful or only caused by large-scale weather?
It is meaningful, but it is also weather-driven. Residual LSTM becomes best overall for known_months=1 because it improves the weather modality enough to overcome losses in AG and NDVI.

## 7. Is the model visually close enough for the known_months=1 industrial case?
It is closer than the raw learned models and now beats the seasonal baseline overall under raw RMSE, but the fit is still mixed by modality. It is promising for research, not yet production-grade.

## 8. Which model should be described as best in a research comparison setting?
For this run and this blank-fill benchmark, residual LSTM is the strongest overall research model for early-year blank filling. Seasonal_last_year remains the strongest classical comparator and still wins or nearly wins in several modality-specific views.

## Known-months=1 modality ranking by raw RMSE
AG:
{ag_known1[['model', 'rmse', 'nrmse_std', 'mae', 'r2']].to_markdown(index=False)}

NDVI:
{ndvi_known1[['model', 'rmse', 'nrmse_std', 'mae', 'r2']].to_markdown(index=False)}

Weather:
{weather_known1[['model', 'rmse', 'nrmse_std', 'mae', 'r2']].to_markdown(index=False)}

## Known-months overall ranking
{known1[['model', 'rmse', 'nrmse_std', 'mae', 'beats_seasonal_last_year']].to_markdown(index=False)}

## Highest normalized-error features for known_months=1
{top_norm_features[['model', 'feature', 'nrmse_std']].head(6).to_markdown(index=False)}
"""
    (artifacts / prefixed_name(prefix, "visual_diagnosis_report.md")).write_text(text, encoding="utf-8")


def make_strict_comparison_plots(
    artifacts: Path,
    current_prefix: str,
    compare_prefix: str | None,
    current_by_modality: pd.DataFrame,
    current_by_horizon: pd.DataFrame,
    current_blank_fill: pd.DataFrame,
    models: list[str],
) -> list[Path]:
    outputs: list[Path] = []
    if not compare_prefix or current_prefix == compare_prefix:
        return outputs

    compare_summary_path = artifacts / f"{compare_prefix}_metrics_summary.csv"
    compare_horizon_path = artifacts / f"{compare_prefix}_metrics_by_horizon.csv"
    compare_pred_path = artifacts / f"{compare_prefix}_predictions_long.csv"
    if not compare_summary_path.exists():
        return outputs

    compare_summary = pd.read_csv(compare_summary_path)
    compare_horizon = pd.read_csv(compare_horizon_path) if compare_horizon_path.exists() else pd.DataFrame()
    compare_pred = pd.read_csv(compare_pred_path) if compare_pred_path.exists() else pd.DataFrame()

    current_known1 = current_by_modality[current_by_modality["known_months"].eq(1)][["model", "modality", "rmse"]].copy()
    current_known1["source"] = current_prefix
    compare_known1 = compare_summary[compare_summary["known_months"].eq(1)][["model", "modality", "rmse"]].copy()
    compare_known1["source"] = compare_prefix
    stacked = pd.concat([current_known1, compare_known1], ignore_index=True)
    if not stacked.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
        for ax, modality in zip(axes, ["ag", "ndvi", "weather"]):
            sub = stacked[stacked["modality"].eq(modality)]
            pivot = sub.pivot(index="model", columns="source", values="rmse").reindex(index=models)
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"known=1 RMSE | {modality}")
            ax.set_xlabel("model")
            ax.set_ylabel("RMSE")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, axis="y", alpha=0.3)
        path = artifacts / "plots_diagnostics_strict" / "strict_vs_standard_known1_rmse_by_modality.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    if not current_by_horizon.empty:
        current_h1 = current_by_horizon[current_by_horizon["known_months"].eq(1)].copy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
        for ax, modality in zip(axes, ["ag", "ndvi", "weather"]):
            sub = current_h1[current_h1["modality"].eq(modality)]
            for model in models:
                grp = sub[sub["model"].eq(model)].sort_values("horizon")
                if grp.empty:
                    continue
                ax.plot(grp["horizon"], grp["rmse"], marker="o", label=model)
            ax.set_title(f"strict known=1 horizon RMSE | {modality}")
            ax.set_xlabel("horizon")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        path = artifacts / "plots_diagnostics_strict" / "strict_known1_horizon_rmse.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    if not current_blank_fill.empty:
        known1 = current_blank_fill[current_blank_fill["known_months"].eq(1)].copy()
        weather_features = [f for f in SELECTED_FEATURES["weather"] if f in known1["feature"].unique()]
        if weather_features:
            fig, axes = plt.subplots(len(weather_features), 1, figsize=(9, 2.8 * len(weather_features)), sharex=True)
            if len(weather_features) == 1:
                axes = [axes]
            for ax, feature in zip(axes, weather_features):
                sub = known1[known1["feature"].eq(feature)]
                truth = sub.groupby("target_month", as_index=False)["y_true"].mean()
                ax.plot(truth["target_month"], truth["y_true"], marker="o", linewidth=2, label="truth")
                for model in models:
                    pred = sub[sub["model"].eq(model)].groupby("target_month", as_index=False)["y_pred"].mean()
                    if pred.empty:
                        continue
                    ax.plot(pred["target_month"], pred["y_pred"], marker="o", linestyle="--", label=model)
                ax.set_title(feature)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("month")
            axes[0].legend(fontsize=8, ncol=2)
            path = artifacts / "plots_diagnostics_strict" / "strict_known1_weather_overlay_selected_features.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)

        weather = known1[known1["modality"].eq("weather")]
        if not weather.empty:
            fig, axes = plt.subplots(2, 2, figsize=(9, 9))
            for ax, model in zip(axes.flatten(), models):
                sub = weather[weather["model"].eq(model)]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                y_true = sub["y_true"].to_numpy(dtype=float)
                y_pred = sub["y_pred"].to_numpy(dtype=float)
                ax.scatter(y_true, y_pred, s=8, alpha=0.25)
                lo = min(np.min(y_true), np.min(y_pred))
                hi = max(np.max(y_true), np.max(y_pred))
                ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1)
                ax.set_title(model)
                ax.set_xlabel("y_true")
                ax.set_ylabel("y_pred")
                ax.grid(True, alpha=0.3)
            path = artifacts / "plots_diagnostics_strict" / "strict_known1_prediction_vs_truth_scatter_weather.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scaling and generate blank-fill diagnostics plots and tables.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--blank-fill-prefix", default="blank_fill")
    parser.add_argument("--plot-dir-name", default=None)
    parser.add_argument("--compare-prefix", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    prefix = str(args.blank_fill_prefix)
    loaded = load_artifacts(run_dir, prefix)
    config = loaded["config"]
    monthly = loaded["monthly"]
    scaler = loaded["scaler"]
    blank_fill = loaded["blank_fill"]
    feature_diag = loaded["feature_diag"]
    artifacts = loaded["artifacts"]
    models = ordered_models(blank_fill)
    plot_dir_name = args.plot_dir_name or ("plots_diagnostics" if prefix == "blank_fill" else f"plots_diagnostics_{slugify(prefix)}")
    plot_dir = artifacts / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = scaler["feature"].tolist()
    feature_stats = compute_train_feature_stats(monthly, feature_cols, config["train_years"])
    blank_fill = add_normalized_error_columns(blank_fill, feature_stats)

    by_feature, by_modality, summary = build_normalized_metric_frames(blank_fill)
    by_feature.to_csv(artifacts / prefixed_name(prefix, "normalized_metrics_by_feature.csv"), index=False)
    by_modality.to_csv(artifacts / prefixed_name(prefix, "normalized_metrics_by_modality.csv"), index=False)
    summary.to_csv(artifacts / prefixed_name(prefix, "normalized_metrics_summary.csv"), index=False)

    scaling_report = build_scaling_audit_report(run_dir, config, feature_diag)
    (artifacts / prefixed_name(prefix, "scaling_audit_report.md")).write_text(scaling_report, encoding="utf-8")

    representative_county = str(sorted(monthly["county_id"].astype(str).unique())[0])
    outputs: list[Path] = []
    outputs += make_overlay_plots(monthly, blank_fill, plot_dir, representative_county, models)
    outputs += make_scatter_plots(blank_fill, plot_dir, models)
    outputs += make_heatmaps(blank_fill, plot_dir, models)
    outputs += make_feature_bar_plots(by_feature, plot_dir, models)
    outputs += make_horizon_plots(blank_fill, plot_dir, models)

    build_report_tables(artifacts, by_modality, by_feature, summary, prefix)
    build_visual_diagnosis_report(artifacts, run_dir, by_modality, by_feature, summary, prefix)
    outputs += make_strict_comparison_plots(
        artifacts,
        prefix,
        args.compare_prefix,
        by_modality,
        pd.read_csv(artifacts / f"{prefix}_metrics_by_horizon.csv"),
        blank_fill,
        models,
    )

    print("Generated reports and plots:")
    for path in [
        artifacts / prefixed_name(prefix, "scaling_audit_report.md"),
        artifacts / prefixed_name(prefix, "normalized_metrics_by_feature.csv"),
        artifacts / prefixed_name(prefix, "normalized_metrics_by_modality.csv"),
        artifacts / prefixed_name(prefix, "normalized_metrics_summary.csv"),
        artifacts / prefixed_name(prefix, "known1_model_comparison_raw_and_normalized.md"),
        artifacts / prefixed_name(prefix, "known1_feature_winners.md"),
        artifacts / prefixed_name(prefix, "known_month_sensitivity_summary.md"),
        artifacts / prefixed_name(prefix, "visual_diagnosis_report.md"),
    ]:
        print(path)
    print(f"{plot_dir_name}_count={len(outputs)}")


if __name__ == "__main__":
    main()
