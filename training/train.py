from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.scaling import FeatureScaler
from cropnet_forecasting.training_dataset import (
    DEFAULT_OUTPUT_DIR,
    load_prepared_dataset,
    load_prepared_metadata,
)
from cropnet_forecasting.training_engine import (
    ALL_SUPPORTED_MODELS,
    BASELINE_MODELS,
    AugmentationConfig,
    LEARNED_MODELS,
    SequenceSplit,
    build_pinn_model,
    build_sequence_splits,
    ensemble_prediction,
    fit_sarima_predictions,
    predict_final_raw,
    summarize_predictions,
    train_torch_model,
)


@dataclass(slots=True)
class ModelRunSummary:
    model: str
    model_type: str
    split: str
    train_loss: float
    val_loss: float
    physics_loss: float
    rmse: float
    mae: float
    mse: float
    r2: float
    val_rmse: float
    val_mae: float
    val_mse: float
    val_r2: float
    checkpoint_path: str
    history_path: str
    loss_curve_path: str
    physics_curve_path: str
    predictions_path: str
    trainable_parameters: int
    total_parameters: int
    status: str


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CropNet feature-forecasting models from a prepared dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm"],
        help="Models to run. Use 'all' to train the practical training set: learned models, baselines, and ensembles.",
    )
    parser.add_argument("--feature-group", default=None, help="Optional feature-group override. Defaults to the prepared dataset metadata.")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--target-mode", choices=["raw", "seasonal_residual"], default="seasonal_residual")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--feature-mask-prob", type=float, default=0.05)
    parser.add_argument("--time-mask-prob", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--latent-state-dim", type=int, default=3)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--physics-warmup-epochs", type=int, default=5)
    parser.add_argument("--physics-loss", choices=["growth", "phenology", "water", "combined"], default="combined")
    parser.add_argument("--physics-config", type=Path, default=ROOT / "training" / "physics_weights.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_models(requested: list[str]) -> list[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(LEARNED_MODELS + BASELINE_MODELS + ("ensemble_mean", "ensemble_weighted"))
    result: list[str] = []
    for model in requested:
        if model not in ALL_SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. Valid models: {', '.join(ALL_SUPPORTED_MODELS)}")
        if model not in result:
            result.append(model)
    return result


def _make_run_dir(output_dir: Path, run_name: str | None, models: list[str]) -> Path:
    if run_name is None:
        run_name = f"{'_'.join(models)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir / run_name


def _make_loader(split: SequenceSplit, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split.X_scaled, dtype=torch.float32),
        torch.tensor(split.X_raw, dtype=torch.float32),
        torch.tensor(split.y_model_scaled, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _feature_count(frame: pd.DataFrame, feature_cols: list[str]) -> int:
    if not feature_cols:
        raise ValueError("No feature columns available for training.")
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Prepared dataset is missing feature columns: {missing}")
    return len(feature_cols)


def _save_prediction_frame(split: SequenceSplit, pred_raw: np.ndarray, feature_cols: list[str], path: Path) -> None:
    if split.metadata.empty or pred_raw.size == 0:
        pd.DataFrame(columns=["county_id", "crop_type", "target_year", "target_month", "target_date"]).to_csv(path, index=False)
        return
    frame = split.metadata.copy().reset_index(drop=True)
    extra_columns: dict[str, np.ndarray] = {}
    for idx, feature in enumerate(feature_cols):
        extra_columns[f"actual_{feature}"] = split.y_true_raw[:, idx]
        extra_columns[f"pred_{feature}"] = pred_raw[:, idx]
        extra_columns[f"error_{feature}"] = pred_raw[:, idx] - split.y_true_raw[:, idx]
    frame = pd.concat([frame, pd.DataFrame(extra_columns)], axis=1)
    frame.to_csv(path, index=False)


def _save_loss_curve(
    history: list[dict[str, float]],
    path: Path,
    *,
    train_col: str = "train_loss",
    val_col: str = "val_loss",
    title: str = "Training Loss Curve",
    physics_start_epoch: int | None = None,
    hide_before_epoch: int | None = None,
    line_label_prefix: str = "",
) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    if hide_before_epoch is not None:
        df = df[df["epoch"] > hide_before_epoch].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_col in df.columns:
        label = train_col.replace("_", " ").title()
        if line_label_prefix:
            label = f"{line_label_prefix}{label}"
        ax.plot(df["epoch"], df[train_col], label=label, linewidth=2)
    if val_col in df.columns and df[val_col].notna().any():
        label = val_col.replace("_", " ").title()
        if line_label_prefix:
            label = f"{line_label_prefix}{label}"
        ax.plot(df["epoch"], df[val_col], label=label, linewidth=2)
    if physics_start_epoch is not None:
        marker_x = physics_start_epoch + 0.5
        ax.axvline(marker_x, color="#444444", linestyle="--", linewidth=1.5, alpha=0.8)
        ymax = ax.get_ylim()[1]
        ax.text(
            marker_x + 0.15,
            ymax * 0.95,
            "physics loss start here",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color="#444444",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_history_plots(history: list[dict[str, float]], model_dir: Path, *, physics_warmup_epochs: int) -> tuple[Path, Path]:
    loss_curve_path = model_dir / "loss_curve.png"
    physics_curve_path = model_dir / "physics_curve.png"
    _save_loss_curve(
        history,
        loss_curve_path,
        train_col="train_forecast_loss",
        val_col="val_forecast_loss",
        title="Training Forecast Loss",
        physics_start_epoch=physics_warmup_epochs,
    )
    if any("train_physics_loss" in row for row in history):
        _save_loss_curve(
            history,
            physics_curve_path,
            train_col="train_physics_loss",
            val_col="val_physics_loss",
            title="Training Physics Loss",
            hide_before_epoch=physics_warmup_epochs,
            line_label_prefix="Post-warmup ",
        )
    return loss_curve_path, physics_curve_path


def _print_model_report(summary: ModelRunSummary) -> None:
    print(f"Model: {summary.model}")
    print(f"  Type: {summary.model_type}")
    print(f"  Params: {summary.trainable_parameters:,} trainable / {summary.total_parameters:,} total")
    print(f"  Train Loss: {summary.train_loss:.4f} | Val Loss: {summary.val_loss:.4f} | Physics Loss: {summary.physics_loss:.4f}")
    print(f"  Val   RMSE: {summary.val_rmse:.4f} | MAE: {summary.val_mae:.4f} | MSE: {summary.val_mse:.4f} | R2: {summary.val_r2:.4f}")
    print(f"  Test  RMSE: {summary.rmse:.4f} | MAE: {summary.mae:.4f} | MSE: {summary.mse:.4f} | R2: {summary.r2:.4f}")
    if summary.history_path:
        print(f"  Loss curve: {summary.loss_curve_path}")
    if summary.physics_curve_path:
        print(f"  Physics curve: {summary.physics_curve_path}")
    print(f"  Predictions: {summary.predictions_path}")


def _fit_single_model(
    model_name: str,
    *,
    splits: dict[str, SequenceSplit],
    feature_cols: list[str],
    scaler: FeatureScaler,
    run_dir: Path,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    target_mode: str,
    augmentation: AugmentationConfig | None,
    latent_state_dim: int,
    physics_weight: float,
    physics_warmup_epochs: int,
    physics_loss: str,
    physics_config: dict[str, Any],
) -> tuple[ModelRunSummary, dict[str, np.ndarray]]:
    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    if model_name in LEARNED_MODELS:
        model = build_pinn_model(
            model_name,
            len(feature_cols),
            feature_names=feature_cols,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            latent_state_dim=latent_state_dim,
            physics_loss=physics_loss,
            physics_config=physics_config,
        )
        train_loader = _make_loader(train_split, batch_size=batch_size, shuffle=True)
        val_loader = _make_loader(val_split, batch_size=batch_size, shuffle=False) if not val_split.empty() else None
        model, history = train_torch_model(
            model,
            train_loader,
            val_loader,
            device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            augmentation=augmentation,
            physics_weight=physics_weight,
            physics_warmup_epochs=physics_warmup_epochs,
        )
        checkpoint_path = model_dir / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "model_name": model_name, "feature_cols": feature_cols}, checkpoint_path)
        history_path = model_dir / "history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        val_pred = predict_final_raw(
            model_name=model_name,
            model=model,
            split=val_split,
            scaler=scaler,
            target_mode=target_mode,
            device=device,
        )
        test_pred = predict_final_raw(
            model_name=model_name,
            model=model,
            split=test_split,
            scaler=scaler,
            target_mode=target_mode,
            device=device,
        )
        val_metrics = summarize_predictions(val_split.y_true_raw, val_pred)
        test_metrics = summarize_predictions(test_split.y_true_raw, test_pred)
        predictions_path = model_dir / "test_predictions.csv"
        _save_prediction_frame(test_split, test_pred, feature_cols, predictions_path)
        loss_curve_path, physics_curve_path = _save_history_plots(history, model_dir, physics_warmup_epochs=physics_warmup_epochs)
        final_row = history[-1] if history else {}
        train_loss = float(final_row.get("train_loss", float("nan")))
        val_loss = float(final_row.get("val_loss", float("nan")))
        physics_loss_value = float(final_row.get("val_physics_loss", float("nan")))
        save_json(
            {
                "model_name": model_name,
                "feature_cols": feature_cols,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "physics_weight": physics_weight,
                "physics_warmup_epochs": physics_warmup_epochs,
                "physics_loss": physics_loss,
                "physics_config": physics_config,
                "augmentation": None
                if augmentation is None
                else {
                    "input_noise_std": augmentation.input_noise_std,
                    "feature_mask_prob": augmentation.feature_mask_prob,
                    "time_mask_prob": augmentation.time_mask_prob,
                },
                "seq_len": int(train_split.X_scaled.shape[1]) if not train_split.empty() else 0,
            },
            model_dir / "config.json",
        )
        summary = ModelRunSummary(
            model=model_name,
            model_type="learned",
            split="test",
            train_loss=train_loss,
            val_loss=val_loss,
            physics_loss=physics_loss_value,
            rmse=test_metrics["rmse"],
            mae=test_metrics["mae"],
            mse=test_metrics["mse"],
            r2=test_metrics["r2"],
            val_rmse=val_metrics["rmse"],
            val_mae=val_metrics["mae"],
            val_mse=val_metrics["mse"],
            val_r2=val_metrics["r2"],
            checkpoint_path=str(checkpoint_path),
            history_path=str(history_path),
            loss_curve_path=str(loss_curve_path),
            physics_curve_path=str(physics_curve_path),
            predictions_path=str(predictions_path),
            trainable_parameters=int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
            total_parameters=int(sum(param.numel() for param in model.parameters())),
            status="trained",
        )
        save_json(
            {
                "model_name": model_name,
                "model_type": "learned",
                "train_metrics": val_metrics,
                "test_metrics": test_metrics,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "final_physics_loss": physics_loss_value,
                "trainable_parameters": summary.trainable_parameters,
                "total_parameters": summary.total_parameters,
                "history_path": str(history_path),
                "loss_curve_path": str(loss_curve_path),
                "physics_curve_path": str(physics_curve_path),
                "checkpoint_path": str(checkpoint_path),
                "predictions_path": str(predictions_path),
            },
            model_dir / "metrics.json",
        )
        _print_model_report(summary)
        return summary, {"val": val_pred, "test": test_pred, "val_rmse": np.asarray([val_metrics["rmse"]])}

    if model_name in BASELINE_MODELS:
        if model_name == "naive_lag1":
            val_pred = scaler.inverse_transform_array(val_split.X_scaled[:, -1, :], feature_cols).astype(np.float32)
            test_pred = scaler.inverse_transform_array(test_split.X_scaled[:, -1, :], feature_cols).astype(np.float32)
        else:
            val_pred = val_split.seasonal_base_raw.copy()
            test_pred = test_split.seasonal_base_raw.copy()
        val_metrics = summarize_predictions(val_split.y_true_raw, val_pred)
        test_metrics = summarize_predictions(test_split.y_true_raw, test_pred)
        predictions_path = model_dir / "test_predictions.csv"
        _save_prediction_frame(test_split, test_pred, feature_cols, predictions_path)
        summary = ModelRunSummary(
            model=model_name,
            model_type="baseline",
            split="test",
            train_loss=float("nan"),
            val_loss=float("nan"),
            physics_loss=float("nan"),
            rmse=test_metrics["rmse"],
            mae=test_metrics["mae"],
            mse=test_metrics["mse"],
            r2=test_metrics["r2"],
            val_rmse=val_metrics["rmse"],
            val_mae=val_metrics["mae"],
            val_mse=val_metrics["mse"],
            val_r2=val_metrics["r2"],
            checkpoint_path="",
            history_path="",
            loss_curve_path="",
            physics_curve_path="",
            predictions_path=str(predictions_path),
            trainable_parameters=0,
            total_parameters=0,
            status="evaluated",
        )
        save_json(
            {
                "model_name": model_name,
                "model_type": "baseline",
                "train_metrics": val_metrics,
                "test_metrics": test_metrics,
                "final_train_loss": None,
                "final_val_loss": None,
                "final_physics_loss": None,
                "trainable_parameters": 0,
                "total_parameters": 0,
                "predictions_path": str(predictions_path),
            },
            model_dir / "metrics.json",
        )
        _print_model_report(summary)
        return summary, {"val": val_pred, "test": test_pred, "val_rmse": np.asarray([val_metrics["rmse"]])}

    raise ValueError(f"Unsupported model in training pipeline: {model_name}")


def _evaluate_sarima(
    *,
    frame: pd.DataFrame,
    feature_cols: list[str],
    splits: dict[str, SequenceSplit],
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    run_dir: Path,
) -> ModelRunSummary:
    forecast_frames = fit_sarima_predictions(frame, feature_cols, train_years, val_years, test_years)
    model_dir = run_dir / "sarima"
    model_dir.mkdir(parents=True, exist_ok=True)

    def _predict_for_split(split_name: str) -> np.ndarray:
        split = splits[split_name]
        if split.metadata.empty:
            return np.empty((0, len(feature_cols)), dtype=np.float32)
        forecast_frame = forecast_frames.get(split_name, pd.DataFrame())
        if forecast_frame.empty:
            return np.zeros((len(split.metadata), len(feature_cols)), dtype=np.float32)
        merged = split.metadata.merge(
            forecast_frame,
            left_on=["county_id", "crop_type", "target_date"],
            right_on=["county_id", "crop_type", "date"],
            how="left",
            sort=False,
        )
        pred = np.zeros((len(merged), len(feature_cols)), dtype=np.float32)
        for idx, feature in enumerate(feature_cols):
            pred[:, idx] = merged[feature].to_numpy(dtype=float)
        return pred

    val_pred = _predict_for_split("val")
    test_pred = _predict_for_split("test")
    val_metrics = summarize_predictions(splits["val"].y_true_raw, val_pred)
    test_metrics = summarize_predictions(splits["test"].y_true_raw, test_pred)
    predictions_path = model_dir / "test_predictions.csv"
    _save_prediction_frame(splits["test"], test_pred, feature_cols, predictions_path)
    return ModelRunSummary(
        model="sarima",
        model_type="baseline",
        split="test",
        train_loss=float("nan"),
        val_loss=float("nan"),
        physics_loss=float("nan"),
        rmse=test_metrics["rmse"],
        mae=test_metrics["mae"],
        mse=test_metrics["mse"],
        r2=test_metrics["r2"],
        val_rmse=val_metrics["rmse"],
        val_mae=val_metrics["mae"],
        val_mse=val_metrics["mse"],
        val_r2=val_metrics["r2"],
        checkpoint_path="",
        history_path="",
        loss_curve_path="",
        physics_curve_path="",
        predictions_path=str(predictions_path),
        trainable_parameters=0,
        total_parameters=0,
        status="evaluated",
    ), {"val": val_pred, "test": test_pred, "val_rmse": np.asarray([val_metrics["rmse"]])}


def main() -> int:
    args = build_parser().parse_args()
    model_names = _resolve_models(args.models)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metadata = load_prepared_metadata(args.dataset_dir)
    frames = load_prepared_dataset(args.dataset_dir)
    feature_cols = list(metadata["feature_columns"])
    if args.feature_group is not None and args.feature_group != metadata.get("feature_group"):
        raise ValueError(
            f"Prepared dataset feature group is {metadata.get('feature_group')!r}, "
            f"but --feature-group requested {args.feature_group!r}."
        )
    scaler = FeatureScaler.from_csv(args.dataset_dir / "scaler.csv").subset(feature_cols)
    splits = build_sequence_splits(
        frames["all"],
        feature_cols,
        scaler,
        seq_len=args.seq_len,
        target_mode=args.target_mode,
    )
    if splits["train"].empty():
        raise ValueError("Prepared training dataset did not yield any train sequences.")

    augmentation = AugmentationConfig(
        input_noise_std=args.input_noise_std,
        feature_mask_prob=args.feature_mask_prob,
        time_mask_prob=args.time_mask_prob,
    )

    run_dir = _make_run_dir(args.output_dir, args.run_name, model_names)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Re-run with --overwrite or choose a new --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "dataset_dir": args.dataset_dir,
            "models": model_names,
            "feature_cols": feature_cols,
            "target_mode": args.target_mode,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "latent_state_dim": args.latent_state_dim,
            "physics_weight": args.physics_weight,
            "physics_warmup_epochs": args.physics_warmup_epochs,
            "physics_loss": args.physics_loss,
            "physics_config_path": str(args.physics_config),
            "augmentation": {
                "input_noise_std": augmentation.input_noise_std,
                "feature_mask_prob": augmentation.feature_mask_prob,
                "time_mask_prob": augmentation.time_mask_prob,
            },
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "device": str(device),
            "metadata": metadata,
        },
        run_dir / "config.json",
    )

    # Keep the scaler alongside the run for downstream inference.
    pd.read_csv(args.dataset_dir / "scaler.csv").to_csv(run_dir / "scaler.csv", index=False)
    physics_config = json.loads(args.physics_config.read_text(encoding="utf-8")) if args.physics_config.exists() else {}

    results: list[ModelRunSummary] = []
    learned_predictions: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    val_rmses: dict[str, float] = {}

    for model_name in model_names:
        if model_name in LEARNED_MODELS or model_name in BASELINE_MODELS:
            summary, preds = _fit_single_model(
                model_name,
                splits=splits,
                feature_cols=feature_cols,
                scaler=scaler,
                run_dir=run_dir,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_epochs=args.max_epochs,
                patience=args.patience,
                device=device,
                target_mode=args.target_mode,
                augmentation=augmentation if model_name in LEARNED_MODELS else None,
                latent_state_dim=args.latent_state_dim,
                physics_weight=args.physics_weight,
                physics_warmup_epochs=args.physics_warmup_epochs,
                physics_loss=args.physics_loss,
                physics_config=physics_config,
            )
            results.append(summary)
            learned_predictions["val"][model_name] = preds["val"]
            learned_predictions["test"][model_name] = preds["test"]
            val_rmses[model_name] = float(preds["val_rmse"][0]) if preds.get("val_rmse") is not None else float("nan")
            continue

        if model_name == "sarima":
            summary, preds = _evaluate_sarima(
                frame=frames["all"],
                feature_cols=feature_cols,
                splits=splits,
                train_years=list(metadata["train_years"]),
                val_years=list(metadata["val_years"]),
                test_years=list(metadata["test_years"]),
                run_dir=run_dir,
            )
            results.append(summary)
            learned_predictions["val"][model_name] = preds["val"]
            learned_predictions["test"][model_name] = preds["test"]
            val_rmses[model_name] = float(preds["val_rmse"][0]) if preds.get("val_rmse") is not None else float("nan")
            continue

    if "ensemble_mean" in model_names or "ensemble_weighted" in model_names:
        learned_names = [name for name in learned_predictions["test"] if name in LEARNED_MODELS]
        if not learned_names:
            raise ValueError("Ensemble models require at least one learned model to be trained in the same run.")
        ensemble_dir = run_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        weights = None
        if "ensemble_weighted" in model_names:
            weights = {}
            inverse_scores = []
            for name in learned_names:
                score = val_rmses.get(name)
                if score is None or not np.isfinite(score) or score <= 0:
                    continue
                inverse_scores.append((name, 1.0 / score))
            total = sum(weight for _, weight in inverse_scores)
            if total > 0:
                weights = {name: weight / total for name, weight in inverse_scores}
        for ensemble_name in [name for name in ("ensemble_mean", "ensemble_weighted") if name in model_names]:
            pred_val = ensemble_prediction({name: learned_predictions["val"][name] for name in learned_names}, weights=weights if ensemble_name == "ensemble_weighted" else None)
            pred_test = ensemble_prediction({name: learned_predictions["test"][name] for name in learned_names}, weights=weights if ensemble_name == "ensemble_weighted" else None)
            val_metrics = summarize_predictions(splits["val"].y_true_raw, pred_val)
            metrics = summarize_predictions(splits["test"].y_true_raw, pred_test)
            predictions_path = ensemble_dir / f"{ensemble_name}_test_predictions.csv"
            _save_prediction_frame(splits["test"], pred_test, feature_cols, predictions_path)
            results.append(
                ModelRunSummary(
                    model=ensemble_name,
                    model_type="ensemble",
                    split="test",
                    train_loss=float("nan"),
                    val_loss=float("nan"),
                    physics_loss=float("nan"),
                    rmse=metrics["rmse"],
                    mae=metrics["mae"],
                    mse=metrics["mse"],
                    r2=metrics["r2"],
                    val_rmse=val_metrics["rmse"],
                    val_mae=val_metrics["mae"],
                    val_mse=val_metrics["mse"],
                    val_r2=val_metrics["r2"],
                    checkpoint_path="",
                    history_path="",
                    loss_curve_path="",
                    physics_curve_path="",
                    predictions_path=str(predictions_path),
                    trainable_parameters=0,
                    total_parameters=0,
                    status="evaluated",
                )
            )

    results_frame = pd.DataFrame([asdict(summary) for summary in results]).sort_values(["rmse", "mae", "model"]).reset_index(drop=True)
    results_frame.to_csv(run_dir / "metrics.csv", index=False)
    if not results_frame.empty:
        print("\nRun summary:")
        print(results_frame.to_string(index=False))
        print(f"Best model: {results_frame.iloc[0]['model']}")
    save_json(
        {
            "run_dir": str(run_dir),
            "best_model": None if results_frame.empty else str(results_frame.iloc[0]["model"]),
            "results": results_frame.to_dict(orient="records"),
        },
        run_dir / "report.json",
    )
    (run_dir / "report.md").write_text(
        "# Training Report\n\n"
        f"- Dataset: `{args.dataset_dir}`\n"
        f"- Models: `{', '.join(model_names)}`\n"
        f"- Target mode: `{args.target_mode}`\n"
        f"- Seq len: `{args.seq_len}`\n"
        f"- Device: `{device}`\n\n"
        "## Results\n\n"
        + ("```text\n" + (results_frame.to_string(index=False) if not results_frame.empty else "No results") + "\n```\n"),
        encoding="utf-8",
    )

    model_specs_rows = []
    for summary in results:
        model_specs_rows.append(
            {
                "model": summary.model,
                "model_type": summary.model_type,
                "checkpoint_path": summary.checkpoint_path,
                "history_path": summary.history_path,
                "predictions_path": summary.predictions_path,
                "trainable_parameters": summary.trainable_parameters,
                "total_parameters": summary.total_parameters,
                "status": summary.status,
            }
        )
    pd.DataFrame(model_specs_rows).to_csv(run_dir / "model_specs.csv", index=False)

    print(f"Training complete. Results written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
