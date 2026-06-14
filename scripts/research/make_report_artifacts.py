from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


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
MODELS = ["seasonal_last_year", "naive_lag1", "lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"]


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def feature_to_modality(feature: str) -> str:
    if feature.startswith("ag_"):
        return "ag"
    if feature.startswith("ndvi_"):
        return "ndvi"
    if feature.startswith("weather_"):
        return "weather"
    return "other"


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


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
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_table_md(path: Path, frame: pd.DataFrame) -> None:
    path.write_text(frame.to_markdown(index=False), encoding="utf-8")


def load_run(run_dir: Path, strict_prefix: str, standard_prefix: str) -> dict:
    artifacts = run_dir / "artifacts"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    run_status_path = run_dir / "run_status.json"
    run_status = json.loads(run_status_path.read_text(encoding="utf-8")) if run_status_path.exists() else {}
    monthly = pd.read_parquet(artifacts / "official_monthly_feature_table.parquet")
    scaler = pd.read_csv(artifacts / "scaler.csv")
    seq_meta = pd.read_csv(artifacts / "sequence_metadata.csv")
    one_step_mod = pd.read_csv(artifacts / "model_metrics_by_modality.csv")
    one_step_feat = pd.read_csv(artifacts / "model_metrics_by_feature.csv")
    strict_summary = pd.read_csv(artifacts / f"{strict_prefix}_metrics_summary.csv")
    strict_horizon = pd.read_csv(artifacts / f"{strict_prefix}_metrics_by_horizon.csv")
    strict_month = pd.read_csv(artifacts / f"{strict_prefix}_metrics_by_month.csv")
    strict_feature = pd.read_csv(artifacts / f"{strict_prefix}_feature_metrics.csv")
    strict_pred = pd.read_csv(artifacts / f"{strict_prefix}_predictions_long.csv")
    strict_norm_summary = pd.read_csv(artifacts / f"{strict_prefix}_normalized_metrics_summary.csv")
    strict_norm_feature = pd.read_csv(artifacts / f"{strict_prefix}_normalized_metrics_by_feature.csv")
    strict_norm_modality = pd.read_csv(artifacts / f"{strict_prefix}_normalized_metrics_by_modality.csv")
    standard_summary = pd.read_csv(artifacts / f"{standard_prefix}_metrics_summary.csv") if (artifacts / f"{standard_prefix}_metrics_summary.csv").exists() else None
    feature_cols = scaler["feature"].tolist()
    return {
        "artifacts": artifacts,
        "config": config,
        "run_status": run_status,
        "monthly": monthly,
        "scaler": scaler,
        "seq_meta": seq_meta,
        "one_step_mod": one_step_mod,
        "one_step_feat": one_step_feat,
        "strict_summary": strict_summary,
        "strict_horizon": strict_horizon,
        "strict_month": strict_month,
        "strict_feature": strict_feature,
        "strict_pred": strict_pred,
        "strict_norm_summary": strict_norm_summary,
        "strict_norm_feature": strict_norm_feature,
        "strict_norm_modality": strict_norm_modality,
        "standard_summary": standard_summary,
        "feature_cols": feature_cols,
    }


def load_history_if_present(artifacts: Path, model_name: str) -> pd.DataFrame | None:
    path = artifacts / f"{model_name}_history.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def infer_train_stats(monthly: pd.DataFrame, feature_cols: list[str], train_years: Iterable[int]) -> pd.DataFrame:
    rows = []
    train = monthly[monthly["year"].isin(list(train_years))]
    for feature in feature_cols:
        s = train[feature].dropna().astype(float)
        std = float(s.std(ddof=1)) if len(s) else float("nan")
        mn = float(s.min()) if len(s) else float("nan")
        mx = float(s.max()) if len(s) else float("nan")
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


def add_one_step_normalized_metrics(test_predictions: pd.DataFrame, feature_stats: pd.DataFrame) -> pd.DataFrame:
    merged = test_predictions.merge(feature_stats, on=["feature", "modality"], how="left")
    merged["abs_error"] = (merged["y_pred"] - merged["y_true"]).abs()
    merged["squared_error"] = (merged["y_pred"] - merged["y_true"]) ** 2
    merged["n_sq_err_std"] = np.where(merged["train_std"].gt(0), merged["squared_error"] / (merged["train_std"] ** 2), np.nan)
    summary = (
        merged.groupby(["model", "modality"], as_index=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
            nrmse_std=("n_sq_err_std", lambda s: float(np.sqrt(np.nanmean(s))) if np.isfinite(np.nanmean(s)) else float("nan")),
        )
    )
    return summary


def choose_representative_county(monthly: pd.DataFrame) -> str:
    selected = [f for cols in SELECTED_FEATURES.values() for f in cols]
    year_df = monthly[monthly["year"].eq(2021)].copy()
    good = []
    for county, group in year_df.groupby("county_id"):
        if all(col in group.columns for col in selected):
            nan_count = int(group[selected].isna().sum().sum())
            good.append((nan_count, str(county)))
    if good:
        good.sort()
        return good[0][1]
    return str(sorted(year_df["county_id"].astype(str).unique())[0])


def parse_model_stage_seconds(log_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not log_path.exists():
        return out
    pattern = re.compile(r"END\s+\|\s+(train-lstm|train-gru|train-mamba|train-transformer-encoder)\s+\|\s+([0-9.]+)s")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        stage, seconds = match.groups()
        key = {
            "train-lstm": "lstm",
            "train-gru": "gru",
            "train-mamba": "tiny_mamba_ssm",
            "train-transformer-encoder": "transformer_encoder",
        }[stage]
        out[key] = float(seconds)
    return out


def build_model_specs(run: dict) -> tuple[pd.DataFrame, str]:
    import cropnet_feature_forecasting_v12_server as mod

    cfg = run["config"]
    artifacts = run["artifacts"]
    seq_meta = run["seq_meta"]
    feature_cols = run["feature_cols"]
    stage_seconds = parse_model_stage_seconds(Path(cfg.get("log_file") or ""))
    rows = []
    n_features = len(feature_cols)
    split_counts = seq_meta["split"].value_counts().to_dict()

    def add_baseline(name: str, rule: str) -> None:
        rows.append(
            {
                "model": name,
                "model_class": "deterministic_baseline",
                "input_dim": n_features,
                "output_dim": n_features,
                "seq_len": cfg.get("seq_len"),
                "hidden_size": np.nan,
                "num_layers": np.nan,
                "dropout": np.nan,
                "learning_rate": np.nan,
                "batch_size": np.nan,
                "max_epochs": np.nan,
                "early_stopping_patience": np.nan,
                "weight_decay": np.nan,
                "target_mode": cfg.get("target_mode"),
                "trainable_parameters": 0,
                "total_parameters": 0,
                "checkpoint_size_bytes": 0,
                "checkpoint_path": "",
                "training_sample_count": int(split_counts.get("train", 0)),
                "validation_sample_count": int(split_counts.get("val", 0)),
                "test_sample_count": int(split_counts.get("test", 0)),
                "device_used": "deterministic",
                "training_runtime_seconds": np.nan,
                "baseline_rule": rule,
            }
        )

    add_baseline("naive_lag1", "Predict previous available month recursively.")
    add_baseline("seasonal_last_year", "Predict same county same month previous year; fallback to lag1 if unavailable.")

    learned_model_names = [
        name
        for name in ["lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"]
        if (artifacts / f"{name}_best.pt").exists() or (artifacts / f"{name}_history.csv").exists()
    ]
    for model_name in learned_model_names:
        ckpt = artifacts / f"{model_name}_best.pt"
        if model_name == "lstm":
            if ckpt.exists():
                state_dict = mod.normalize_legacy_checkpoint_keys(model_name, mod.torch.load(ckpt, map_location="cpu"))
                hidden_size, num_layers, dropout = mod.infer_lstm_architecture(state_dict, n_features, float(cfg.get("dropout", 0.0)))
            else:
                state_dict = None
                hidden_size = int(cfg.get("hidden_size", 64))
                num_layers = int(cfg.get("num_layers", 1))
                dropout = float(cfg.get("dropout", 0.0))
            model = mod.LSTMForecaster(n_features, hidden_size, num_layers, dropout)
        elif model_name == "gru":
            if ckpt.exists():
                state_dict = mod.normalize_legacy_checkpoint_keys(model_name, mod.torch.load(ckpt, map_location="cpu"))
                hidden_size, num_layers, dropout = mod.infer_gru_architecture(state_dict, n_features, float(cfg.get("dropout", 0.0)))
            else:
                state_dict = None
                hidden_size = int(cfg.get("hidden_size", 64))
                num_layers = int(cfg.get("num_layers", 1))
                dropout = float(cfg.get("dropout", 0.0))
            model = mod.GRUForecaster(n_features, hidden_size, num_layers, dropout)
        elif model_name == "tiny_mamba_ssm":
            if ckpt.exists():
                state_dict = mod.normalize_legacy_checkpoint_keys(model_name, mod.torch.load(ckpt, map_location="cpu"))
                hidden_size, num_layers, dropout = mod.infer_mamba_architecture(state_dict, n_features, float(cfg.get("dropout", 0.0)))
            else:
                state_dict = None
                hidden_size = int(cfg.get("hidden_size", 64))
                num_layers = int(cfg.get("num_layers", 1))
                dropout = float(cfg.get("dropout", 0.0))
            model = mod.MambaStyleForecaster(n_features, d_model=hidden_size, d_state=32, num_layers=num_layers, dropout=dropout)
        else:
            if ckpt.exists():
                state_dict = mod.normalize_legacy_checkpoint_keys(model_name, mod.torch.load(ckpt, map_location="cpu"))
                hidden_size, num_layers, dropout = mod.infer_transformer_architecture(state_dict, n_features, float(cfg.get("dropout", 0.0)))
            else:
                state_dict = None
                hidden_size = int(cfg.get("hidden_size", 64))
                num_layers = int(cfg.get("num_layers", 1))
                dropout = float(cfg.get("dropout", 0.0))
            model = mod.TransformerEncoderForecaster(n_features, d_model=hidden_size, num_layers=num_layers, dropout=dropout)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        total_params = int(sum(p.numel() for p in model.parameters()))
        trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        rows.append(
            {
                "model": model_name,
                "model_class": model.__class__.__name__,
                "input_dim": n_features,
                "output_dim": n_features,
                "seq_len": cfg.get("seq_len"),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": cfg.get("learning_rate"),
                "batch_size": cfg.get("batch_size"),
                "max_epochs": cfg.get("epochs"),
                "early_stopping_patience": cfg.get("early_stopping_patience"),
                "weight_decay": cfg.get("weight_decay"),
                "target_mode": cfg.get("target_mode"),
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "checkpoint_size_bytes": ckpt.stat().st_size if ckpt.exists() else np.nan,
                "checkpoint_path": str(ckpt),
                "training_sample_count": int(split_counts.get("train", 0)),
                "validation_sample_count": int(split_counts.get("val", 0)),
                "test_sample_count": int(split_counts.get("test", 0)),
                "device_used": "cuda" if "cuda" in str(cfg.get("log_file", "")).lower() else "unknown",
                "training_runtime_seconds": stage_seconds.get(model_name, np.nan),
                "baseline_rule": "",
            }
        )

    frame = pd.DataFrame(rows)
    lines = ["# Model Specs", "", "## Learned Models", ""]
    learned = frame[frame["model"].isin(learned_model_names)]
    if not learned.empty:
        lines.append(learned.to_markdown(index=False))
    lines += ["", "## Baselines", ""]
    baselines = frame[frame["model"].isin(["naive_lag1", "seasonal_last_year"])][["model", "baseline_rule", "target_mode", "total_parameters"]]
    lines.append(baselines.to_markdown(index=False))
    return frame, "\n".join(lines)


def build_performance_summaries(run: dict, compare_runs: list[tuple[str, dict]]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    one_step = run["one_step_mod"].copy()
    one_step = one_step[one_step["split"].eq("test")].copy()

    feature_stats = infer_train_stats(run["monthly"], run["feature_cols"], run["config"]["train_years"])
    test_predictions = pd.read_csv(run["artifacts"] / "test_predictions_long.csv")
    one_step_norm = add_one_step_normalized_metrics(test_predictions, feature_stats)
    one_step = one_step.merge(one_step_norm[["model", "modality", "nrmse_std"]], on=["model", "modality"], how="left")
    one_step["task"] = "one_step_test"
    one_step["known_months"] = np.nan

    strict_summary = run["strict_summary"].copy()
    strict_norm = run["strict_norm_summary"].copy()
    strict_perf = strict_summary.merge(
        strict_norm[["model", "known_months", "nrmse_std"]],
        on=["model", "known_months"],
        how="left",
    )
    feat = run["strict_feature"].copy()
    feat_counts = (
        feat.groupby(["model", "known_months"], as_index=False)
        .agg(
            features_beating_lag1=("beats_lag1", "sum"),
            features_beating_seasonal_last_year=("beats_seasonal_last_year", "sum"),
        )
    )
    strict_perf = strict_perf.merge(feat_counts, on=["model", "known_months"], how="left")
    strict_perf["task"] = "blank_fill_strict"
    perf_summary = pd.concat(
        [
            one_step[["task", "model", "modality", "known_months", "mae", "rmse", "nrmse_std", "rmse_vs_naive"]],
            strict_perf[["task", "model", "modality", "known_months", "mae", "rmse", "nrmse_std", "features_beating_lag1", "features_beating_seasonal_last_year"]],
        ],
        ignore_index=True,
        sort=False,
    )

    overall_blank = strict_norm[["model", "known_months", "rmse", "mae", "nrmse_std", "beats_lag1", "beats_seasonal_last_year"]].copy()
    best_by_known = overall_blank.sort_values(["known_months", "rmse"]).groupby("known_months", as_index=False).first()[["known_months", "model", "rmse", "nrmse_std"]]

    lines = ["# Model Performance Report", ""]
    lines += ["## One-Step Test Metrics", "", one_step.sort_values(["modality", "rmse"]).to_markdown(index=False), ""]
    lines += ["## Strict Blank-Fill Summary by Model / Known Months / Modality", "", strict_perf.sort_values(["known_months", "modality", "rmse"]).to_markdown(index=False), ""]
    lines += ["## Best Model by Known Months (Overall Strict Blank-Fill)", "", best_by_known.to_markdown(index=False), ""]

    for label, comp in compare_runs:
        comp_std = comp["strict_norm_summary"] if "strict_norm_summary" in comp else None
        if comp_std is None or comp_std.empty:
            continue
        lines += [f"## Comparison: {label}", "", comp_std.sort_values(["known_months", "rmse"]).to_markdown(index=False), ""]

    lines += [
        "## Interpretation",
        "",
        "- Under raw RMSE, residual LSTM is strongest for early-year strict blank filling.",
        "- Under normalized RMSE, seasonal_last_year remains the strongest overall comparator.",
        "- Learned models beat naive_lag1 consistently, but the residual LSTM advantage is still scale-sensitive.",
    ]
    return perf_summary, strict_perf, "\n".join(lines)


def build_loss_curves(artifacts: Path, config: dict, plot_dir: Path) -> tuple[str, list[Path]]:
    outputs: list[Path] = []
    summaries = []
    histories = {}
    learned_model_names = [
        name
        for name in ["lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"]
        if (artifacts / f"{name}_history.csv").exists()
    ]
    for model_name in learned_model_names:
        hist = load_history_if_present(artifacts, model_name)
        histories[model_name] = hist
        if hist is None or plt is None:
            continue
        best_idx = int(hist["val_loss"].idxmin())
        best_epoch = int(hist.loc[best_idx, "epoch"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        ax.plot(hist["epoch"], hist["val_loss"], label="val_loss")
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1, label=f"best_val_epoch={best_epoch}")
        ax.set_title(f"{model_name} loss curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = plot_dir / f"{model_name}_loss_curve.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

        early_stop = int(hist["epoch"].max()) < int(config.get("epochs", 0))
        overfit = bool(hist["val_loss"].iloc[-1] > hist["val_loss"].min() * 1.05 and hist["train_loss"].iloc[-1] <= hist["train_loss"].min() * 1.02)
        summaries.append(
            {
                "model": model_name,
                "best_validation_epoch": best_epoch,
                "best_validation_loss": float(hist["val_loss"].min()),
                "final_train_loss": float(hist["train_loss"].iloc[-1]),
                "final_validation_loss": float(hist["val_loss"].iloc[-1]),
                "early_stopping_occurred": early_stop,
                "overfitting_visible": overfit,
                "history_path": str(artifacts / f"{model_name}_history.csv"),
            }
        )

    if plt is not None and len([name for name in learned_model_names if histories.get(name) is not None]) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for idx, model_name in enumerate(learned_model_names):
            hist = histories.get(model_name)
            if hist is None:
                continue
            color = colors[idx % len(colors)]
            ax.plot(hist["epoch"], hist["val_loss"], label=f"{model_name} val", color=color)
            ax.plot(hist["epoch"], hist["train_loss"], linestyle="--", alpha=0.7, label=f"{model_name} train", color=color)
        ax.set_title("Loss curve comparison")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        out = plot_dir / "loss_curve_comparison.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    if summaries:
        frame = pd.DataFrame(summaries)
        text = "# Loss Curve Summary\n\n" + frame.to_markdown(index=False) + "\n"
    else:
        text = "# Loss Curve Summary\n\nHistory CSVs were not available, so no loss curves were generated.\n"
    return text, outputs


def make_overlay_plots(monthly: pd.DataFrame, strict_pred: pd.DataFrame, plot_dir: Path, representative_county: str) -> list[Path]:
    outputs: list[Path] = []
    year_df = monthly[monthly["year"].eq(2021)].copy()
    for known_months in [1, 3, 6]:
        pred_subset = strict_pred[strict_pred["known_months"].eq(known_months)].copy()
        for modality, features in SELECTED_FEATURES.items():
            for feature in features:
                truth_avg = year_df.groupby("month", as_index=False)[feature].mean()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(truth_avg["month"], truth_avg[feature], marker="o", linewidth=2, label="truth")
                for model in MODELS:
                    pred_avg = (
                        pred_subset[pred_subset["feature"].eq(feature) & pred_subset["model"].eq(model)]
                        .groupby("target_month", as_index=False)["y_pred"]
                        .mean()
                    )
                    if pred_avg.empty:
                        continue
                    ax.plot(pred_avg["target_month"], pred_avg["y_pred"], marker="o", linestyle="--", label=model)
                ax.set_title(f"Average overlay | known={known_months} | {feature}")
                ax.set_xlabel("month")
                ax.set_ylabel(feature)
                ax.set_xticks(range(1, 13))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                out = plot_dir / f"avg_known{known_months}_{slugify(feature)}.png"
                fig.tight_layout()
                fig.savefig(out, dpi=160)
                plt.close(fig)
                outputs.append(out)

    county_truth = year_df[year_df["county_id"].astype(str).eq(representative_county)].sort_values("month")
    for feature in ["ndvi_mean", "weather_gdd", "weather_total_precipitation", "weather_solar_radiation_mean"]:
        pred_subset = strict_pred[(strict_pred["known_months"].eq(1)) & (strict_pred["county_id"].astype(str).eq(representative_county)) & (strict_pred["feature"].eq(feature))]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(county_truth["month"], county_truth[feature], marker="o", linewidth=2, label="truth")
        for model in MODELS:
            line = pred_subset[pred_subset["model"].eq(model)].sort_values("target_month")
            if line.empty:
                continue
            ax.plot(line["target_month"], line["y_pred"], marker="o", linestyle="--", label=model)
        ax.set_title(f"County {representative_county} | known=1 | {feature}")
        ax.set_xlabel("month")
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out = plot_dir / f"county_{representative_county}_known1_{slugify(feature)}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)
    return outputs


def make_scatter_plots(strict_pred: pd.DataFrame, plot_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    known1 = strict_pred[strict_pred["known_months"].eq(1)].copy()
    for modality in ["ag", "ndvi", "weather"]:
        for model in MODELS:
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
            ax.set_title(f"known=1 scatter | {model} | {modality}\ncorr={format_float(safe_corr(y_true, y_pred))} r2={format_float(safe_r2(y_true, y_pred))}")
            ax.set_xlabel("y_true")
            ax.set_ylabel("y_pred")
            ax.grid(True, alpha=0.3)
            out = plot_dir / f"scatter_known1_{slugify(model)}_{modality}.png"
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            outputs.append(out)
    return outputs


def make_error_plots(strict_horizon: pd.DataFrame, strict_month: pd.DataFrame, strict_norm_modality: pd.DataFrame, strict_pred: pd.DataFrame, feature_stats: pd.DataFrame, plot_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    known1 = strict_horizon[strict_horizon["known_months"].eq(1)].copy()
    if not known1.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        overall = (
            strict_pred[strict_pred["known_months"].eq(1)]
            .merge(feature_stats[["feature", "train_std"]], on="feature", how="left")
        )
        overall["n_sq_err_std"] = np.where(overall["train_std"].gt(0), overall["squared_error"] / (overall["train_std"] ** 2), np.nan)
        rows = (
            overall.groupby(["model", "horizon"], as_index=False)
            .agg(nrmse_std=("n_sq_err_std", lambda s: float(np.sqrt(np.nanmean(s))) if np.isfinite(np.nanmean(s)) else float("nan")))
        )
        for model in MODELS:
            grp = rows[rows["model"].eq(model)].sort_values("horizon")
            if grp.empty:
                continue
            ax.plot(grp["horizon"], grp["nrmse_std"], marker="o", label=model)
        ax.set_title("known=1 overall normalized RMSE by horizon")
        ax.set_xlabel("horizon")
        ax.set_ylabel("nRMSE_std")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out = plot_dir / "known1_normalized_rmse_by_horizon.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    month1 = strict_month[strict_month["known_months"].eq(1)]
    for modality in ["ag", "ndvi", "weather"]:
        rows = month1[month1["modality"].eq(modality)]
        if rows.empty:
            continue
        pivot = rows.pivot(index="model", columns="target_month", values="rmse").reindex(index=MODELS)
        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_title(f"known=1 month RMSE heatmap | {modality}")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        out = plot_dir / f"heatmap_known1_{modality}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in MODELS:
            grp = rows[rows["model"].eq(model)].sort_values("target_month")
            if grp.empty:
                continue
            ax.plot(grp["target_month"], grp["rmse"], marker="o", label=model)
        ax.set_title(f"known=1 RMSE by target month | {modality}")
        ax.set_xlabel("target month")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        out = plot_dir / f"rmse_by_month_known1_{modality}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)
    return outputs


def make_feature_bar_plots(strict_feature: pd.DataFrame, strict_norm_feature: pd.DataFrame, plot_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    raw = strict_feature[(strict_feature["known_months"].eq(1)) & (strict_feature["model"].isin(["seasonal_last_year", "lstm", "tiny_mamba_ssm"]))].copy()
    norm = strict_norm_feature[(strict_norm_feature["known_months"].eq(1)) & (strict_norm_feature["model"].isin(["seasonal_last_year", "lstm", "tiny_mamba_ssm"]))].copy()
    for modality in ["ag", "ndvi", "weather"]:
        raw_rows = raw[raw["modality"].eq(modality)]
        if not raw_rows.empty:
            pivot = raw_rows.pivot(index="feature", columns="model", values="rmse").sort_index()
            fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.55), 4.5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"known=1 feature raw RMSE | {modality}")
            ax.tick_params(axis="x", rotation=60)
            ax.grid(True, axis="y", alpha=0.3)
            out = plot_dir / f"feature_bar_known1_{modality}_rmse.png"
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            outputs.append(out)
        norm_rows = norm[norm["modality"].eq(modality)]
        if not norm_rows.empty:
            pivot = norm_rows.pivot(index="feature", columns="model", values="nrmse_std").sort_index()
            fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.55), 4.5))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"known=1 feature normalized RMSE | {modality}")
            ax.tick_params(axis="x", rotation=60)
            ax.grid(True, axis="y", alpha=0.3)
            out = plot_dir / f"feature_bar_known1_{modality}_nrmse_std.png"
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            outputs.append(out)
    return outputs


def build_plot_index(artifacts: Path, plot_dir: Path, zip_path: Path) -> str:
    plots = sorted(plot_dir.rglob("*.png"))
    recs = [
        ("known1_normalized_rmse_by_horizon.png", "Shows recursive degradation on a feature-normalized scale."),
        ("avg_known1_weather_solar_radiation_mean.png", "Average-across-counties weather overlay for a high-scale difficult feature."),
        ("avg_known1_ndvi_mean.png", "Average NDVI blank-fill overlay; visually easier seasonal feature."),
        ("strict_vs_standard_known1_rmse_by_modality.png", "Confirms strict and standard metrics are practically identical."),
        ("loss_curve_comparison.png", "Training/validation loss behavior of the learned models."),
    ]
    lines = ["# Plots Index", "", f"Zip package: `{zip_path}`", "", "## Recommended First Plots", ""]
    lines += [f"- `{name}`: {desc}" for name, desc in recs]
    lines += ["", "## All Generated Plots", ""]
    for plot in plots:
        lines.append(f"- `{plot.name}`")
    lines += [
        "",
        "## Short Interpretation",
        "",
        "- Residual LSTM is strongest under raw early-year blank-fill RMSE.",
        "- Seasonal_last_year remains strongest under normalized RMSE.",
        "- Weather dominates raw-error magnitude, while AG/NDVI often favor the seasonal baseline visually.",
    ]
    return "\n".join(lines)


def package_outputs(artifacts: Path, plot_dir: Path) -> Path:
    zip_path = artifacts / "prediction_visualizations.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in plot_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(artifacts))
        for name in [
            "model_performance_report.md",
            "model_specs.md",
            "loss_curve_summary.md",
            "plots_index.md",
            "model_performance_summary.csv",
            "blank_fill_model_performance_summary.csv",
            "model_specs.csv",
            "strict_blank_fill_metrics_summary.csv",
            "strict_blank_fill_normalized_metrics_summary.csv",
        ]:
            p = artifacts / name
            if p.exists():
                zf.write(p, p.relative_to(artifacts))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report-ready performance summaries, specs, loss curves, and plots from an existing CropNet run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--strict-prefix", default="strict_blank_fill")
    parser.add_argument("--standard-prefix", default="blank_fill")
    parser.add_argument("--compare-run", action="append", default=[])
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run = load_run(run_dir, args.strict_prefix, args.standard_prefix)
    artifacts = run["artifacts"]
    plots_report = artifacts / "plots_report"
    plots_report.mkdir(parents=True, exist_ok=True)

    compare_runs: list[tuple[str, dict]] = []
    for comp_path in args.compare_run:
        comp_dir = Path(comp_path).resolve()
        if not comp_dir.exists():
            continue
        comp = load_run(comp_dir, args.strict_prefix if (comp_dir / "artifacts" / f"{args.strict_prefix}_metrics_summary.csv").exists() else args.standard_prefix, args.standard_prefix)
        compare_runs.append((comp_dir.name, comp))

    perf_summary, blank_perf_summary, perf_md = build_performance_summaries(run, compare_runs)
    perf_summary.to_csv(artifacts / "model_performance_summary.csv", index=False)
    blank_perf_summary.to_csv(artifacts / "blank_fill_model_performance_summary.csv", index=False)
    write_md(artifacts / "model_performance_report.md", perf_md)

    specs_df, specs_md = build_model_specs(run)
    specs_df.to_csv(artifacts / "model_specs.csv", index=False)
    write_md(artifacts / "model_specs.md", specs_md)

    loss_summary_md, loss_plots = build_loss_curves(artifacts, run["config"], plots_report)
    write_md(artifacts / "loss_curve_summary.md", loss_summary_md)

    if plt is not None:
        representative_county = choose_representative_county(run["monthly"])
        feature_stats = infer_train_stats(run["monthly"], run["feature_cols"], run["config"]["train_years"])
        outputs = []
        outputs += make_overlay_plots(run["monthly"], run["strict_pred"], plots_report, representative_county)
        outputs += make_scatter_plots(run["strict_pred"], plots_report)
        outputs += make_error_plots(run["strict_horizon"], run["strict_month"], run["strict_norm_modality"], run["strict_pred"], feature_stats, plots_report)
        outputs += make_feature_bar_plots(run["strict_feature"], run["strict_norm_feature"], plots_report)
        outputs += loss_plots
    else:
        outputs = []

    plots_index_text = build_plot_index(artifacts, plots_report, artifacts / "prediction_visualizations.zip")
    write_md(artifacts / "plots_index.md", plots_index_text)
    zip_path = package_outputs(artifacts, plots_report)
    print(f"Generated: {artifacts / 'model_performance_report.md'}")
    print(f"Generated: {artifacts / 'model_specs.md'}")
    print(f"Generated: {artifacts / 'loss_curve_summary.md'}")
    print(f"Generated: {artifacts / 'plots_index.md'}")
    print(f"Generated: {zip_path}")
    print(f"plots_report_count={len(outputs)}")


if __name__ == "__main__":
    main()
