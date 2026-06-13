from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .features import META_COLS
from .models import CropNetModelFactory
from .scaling import FeatureScaler
from .pinn import PINNForecaster

LEARNED_MODELS = ("lstm", "gru", "transformer_encoder", "tiny_mamba_ssm")
BASELINE_MODELS = ("naive_lag1", "seasonal_last_year")
EVAL_ONLY_MODELS = ("sarima", "ensemble_mean", "ensemble_weighted")
ALL_SUPPORTED_MODELS = LEARNED_MODELS + BASELINE_MODELS + EVAL_ONLY_MODELS


@dataclass(slots=True)
class SequenceSplit:
    X_scaled: np.ndarray
    X_raw: np.ndarray
    y_model_scaled: np.ndarray
    y_true_raw: np.ndarray
    seasonal_base_scaled: np.ndarray
    seasonal_base_raw: np.ndarray
    metadata: pd.DataFrame

    def empty(self) -> bool:
        return self.X_scaled.size == 0


@dataclass(slots=True)
class TrainedModelArtifacts:
    model_name: str
    output_dir: Path
    checkpoint_path: Path | None
    history_path: Path | None
    metrics_path: Path
    predictions_path: Path


@dataclass(slots=True)
class AugmentationConfig:
    input_noise_std: float = 0.01
    feature_mask_prob: float = 0.05
    time_mask_prob: float = 0.0


def _date_index(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(dict(year=frame["year"].astype(int), month=frame["month"].astype(int), day=1))


def _build_seasonal_lookup(frame: pd.DataFrame, feature_cols: list[str]) -> dict[tuple[str, str, int, int], np.ndarray]:
    lookup: dict[tuple[str, str, int, int], np.ndarray] = {}
    cols = ["county_id", "crop_type", "year", "month"] + feature_cols
    for row in frame[cols].itertuples(index=False, name=None):
        county_id, crop_type, year, month, *values = row
        lookup[(str(county_id), str(crop_type), int(year), int(month))] = np.asarray(values, dtype=np.float32)
    return lookup


def build_sequence_splits(
    frame: pd.DataFrame,
    feature_cols: list[str],
    scaler: FeatureScaler,
    *,
    seq_len: int = 6,
    target_mode: str = "seasonal_residual",
) -> dict[str, SequenceSplit]:
    if "split" not in frame.columns:
        raise ValueError("Prepared dataset must contain a split column.")

    working = frame.copy()
    working["date"] = _date_index(working)
    working = working.sort_values(["county_id", "crop_type", "date"]).reset_index(drop=True)
    working_raw = working[META_COLS + feature_cols + ["split", "date"]].copy()
    working_scaled = scaler.transform_frame(working_raw, feature_cols)
    working_scaled["date"] = working_raw["date"].to_numpy()
    working_raw[feature_cols] = (
        working_raw[feature_cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    )
    working_scaled[feature_cols] = (
        working_scaled[feature_cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
    )
    seasonal_lookup_raw = _build_seasonal_lookup(working_raw, feature_cols)
    seasonal_lookup_scaled = _build_seasonal_lookup(working_scaled, feature_cols)

    buckets_scaled: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "val", "test")}
    buckets_raw: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "val", "test")}
    targets: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "val", "test")}
    truths: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "val", "test")}
    seasonal_scaled: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "val", "test")}
    seasonal_raw: dict[str, list[np.ndarray]] = {key: [] for key in ("train", "val", "test")}
    metadata_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "val", "test")}

    for (_, _), group_idx in working.groupby(["county_id", "crop_type"], sort=True).groups.items():
        group_raw = working_raw.loc[group_idx].sort_values("date").reset_index(drop=True)
        group_scaled = working_scaled.loc[group_idx].sort_values("date").reset_index(drop=True)
        group_raw = group_raw.replace([np.inf, -np.inf], np.nan)
        group_scaled = group_scaled.replace([np.inf, -np.inf], np.nan)
        valid_mask = group_raw[feature_cols].notna().all(axis=1) & group_scaled[feature_cols].notna().all(axis=1)
        group_raw = group_raw.loc[valid_mask].reset_index(drop=True)
        group_scaled = group_scaled.loc[valid_mask].reset_index(drop=True)
        if group_raw.empty:
            continue
        raw_values = group_raw[feature_cols].to_numpy(dtype=np.float32)
        scaled_values = group_scaled[feature_cols].to_numpy(dtype=np.float32)
        dates = group_raw["date"].to_list()

        for i in range(seq_len, len(group_raw)):
            target_row = group_raw.iloc[i]
            split = str(target_row["split"])
            if split not in buckets_scaled:
                continue
            if not np.isfinite(raw_values[i]).all() or not np.isfinite(scaled_values[i]).all():
                continue

            window_dates = dates[i - seq_len : i]
            expected_dates = list(pd.date_range(end=target_row["date"] - pd.offsets.MonthBegin(1), periods=seq_len, freq="MS"))
            if window_dates != expected_dates:
                continue

            raw_target = raw_values[i]
            scaled_target = scaled_values[i]
            county_id = str(target_row["county_id"])
            crop_type = str(target_row["crop_type"])
            target_year = int(target_row["year"])
            target_month = int(target_row["month"])
            seasonal_key = (county_id, crop_type, target_year - 1, target_month)
            seasonal_base_raw = seasonal_lookup_raw.get(seasonal_key)
            seasonal_base_scaled = seasonal_lookup_scaled.get(seasonal_key)

            if target_mode == "seasonal_residual":
                if seasonal_base_raw is None or seasonal_base_scaled is None:
                    continue
                if not np.isfinite(seasonal_base_raw).all() or not np.isfinite(seasonal_base_scaled).all():
                    continue
            else:
                seasonal_base_raw = np.zeros(len(feature_cols), dtype=np.float32)
                seasonal_base_scaled = np.zeros(len(feature_cols), dtype=np.float32)

            buckets_scaled[split].append(scaled_values[i - seq_len : i])
            buckets_raw[split].append(raw_values[i - seq_len : i])
            targets[split].append(
                scaled_target if target_mode == "raw" else (scaled_target - seasonal_base_scaled)
            )
            truths[split].append(raw_target)
            seasonal_scaled[split].append(
                seasonal_base_scaled if seasonal_base_scaled is not None else np.zeros(len(feature_cols), dtype=np.float32)
            )
            seasonal_raw[split].append(
                seasonal_base_raw if seasonal_base_raw is not None else np.zeros(len(feature_cols), dtype=np.float32)
            )
            metadata_rows[split].append(
                {
                    "county_id": county_id,
                    "crop_type": crop_type,
                    "target_year": target_year,
                    "target_month": target_month,
                    "target_date": target_row["date"],
                    "split": split,
                }
            )

    result: dict[str, SequenceSplit] = {}
    for split in ("train", "val", "test"):
        if not buckets_scaled[split]:
            result[split] = SequenceSplit(
                X_scaled=np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                X_raw=np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                y_model_scaled=np.empty((0, len(feature_cols)), dtype=np.float32),
                y_true_raw=np.empty((0, len(feature_cols)), dtype=np.float32),
                seasonal_base_scaled=np.empty((0, len(feature_cols)), dtype=np.float32),
                seasonal_base_raw=np.empty((0, len(feature_cols)), dtype=np.float32),
                metadata=pd.DataFrame(columns=["county_id", "crop_type", "target_year", "target_month", "target_date", "split"]),
            )
            continue
        result[split] = SequenceSplit(
            X_scaled=np.stack(buckets_scaled[split]).astype(np.float32),
            X_raw=np.stack(buckets_raw[split]).astype(np.float32),
            y_model_scaled=np.stack(targets[split]).astype(np.float32),
            y_true_raw=np.stack(truths[split]).astype(np.float32),
            seasonal_base_scaled=np.stack(seasonal_scaled[split]).astype(np.float32),
            seasonal_base_raw=np.stack(seasonal_raw[split]).astype(np.float32),
            metadata=pd.DataFrame(metadata_rows[split]),
        )
    return result


def make_loss_fn(loss_weights: torch.Tensor | None = None):
    if loss_weights is None:
        return nn.MSELoss()

    def weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean((diff * diff) * loss_weights)

    return weighted_loss


def _augment_batch(xb: torch.Tensor, config: AugmentationConfig | None) -> torch.Tensor:
    if config is None:
        return xb
    out = xb
    if config.input_noise_std > 0:
        out = out + torch.randn_like(out) * config.input_noise_std
    if config.feature_mask_prob > 0:
        feature_mask = torch.rand(out.shape[0], 1, out.shape[2], device=out.device) < config.feature_mask_prob
        out = out.masked_fill(feature_mask, 0.0)
    if config.time_mask_prob > 0:
        time_mask = torch.rand(out.shape[0], out.shape[1], 1, device=out.device) < config.time_mask_prob
        out = out.masked_fill(time_mask, 0.0)
    return out


def _build_batch_features(split: SequenceSplit) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(split.X_scaled, dtype=torch.float32),
        torch.tensor(split.X_raw, dtype=torch.float32),
        torch.tensor(split.y_model_scaled, dtype=torch.float32),
    )


def _normalize_batch(batch):
    if len(batch) == 3:
        return batch
    if len(batch) == 2:
        xb, yb = batch
        return xb, xb, yb
    raise ValueError("Unexpected batch structure for CropNet training.")


def _forward_model(model: nn.Module, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hasattr(model, "forward_with_latents"):
        pred, latent = model.forward_with_latents(xb)
        return pred, latent
    return model(xb), None


def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn,
    *,
    physics_weight: float = 0.0,
    physics_active: bool = False,
) -> dict[str, float]:
    model.eval()
    if loader is None or len(loader.dataset) == 0:
        return {"forecast_loss": float("nan"), "physics_loss": float("nan"), "total_loss": float("nan")}
    forecast_losses = []
    physics_losses = []
    total_losses = []
    with torch.no_grad():
        for batch in loader:
            xb, xraw, yb = _normalize_batch(batch)
            xb, xraw, yb = xb.to(device), xraw.to(device), yb.to(device)
            pred, latent = _forward_model(model, xb)
            forecast_loss = loss_fn(pred, yb)
            physics_loss = torch.tensor(0.0, device=device)
            if physics_active:
                physics_module = getattr(model, "physics", None)
                if physics_module is None or latent is None:
                    raise ValueError("Physics mode requires a model with forward_with_latents() and physics module.")
                physics_loss = physics_module(xraw, latent)["total"]
            total_loss = forecast_loss + physics_weight * physics_loss
            forecast_losses.append(forecast_loss.item() * len(xb))
            physics_losses.append(physics_loss.item() * len(xb))
            total_losses.append(total_loss.item() * len(xb))
    denom = len(loader.dataset)
    return {
        "forecast_loss": float(np.sum(forecast_losses) / denom),
        "physics_loss": float(np.sum(physics_losses) / denom),
        "total_loss": float(np.sum(total_losses) / denom),
    }


def predict_torch_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            xb, _, _ = _normalize_batch(batch)
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    if not preds:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    loss_weights: torch.Tensor | None = None,
    augmentation: AugmentationConfig | None = None,
    physics_weight: float = 0.0,
    physics_warmup_epochs: int = 0,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, patience // 3),
        min_lr=1e-5,
    )
    loss_fn = make_loss_fn(loss_weights)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = patience
    physics_active = hasattr(model, "physics") and hasattr(model, "forward_with_latents")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_forecast_total = 0.0
        train_physics_total = 0.0
        train_total = 0.0
        train_count = 0
        effective_physics_weight = physics_weight if epoch > physics_warmup_epochs else 0.0
        for batch in train_loader:
            xb, xraw, yb = _normalize_batch(batch)
            xb, xraw, yb = xb.to(device), xraw.to(device), yb.to(device)
            xb = _augment_batch(xb, augmentation)
            optimizer.zero_grad(set_to_none=True)
            pred, latent = _forward_model(model, xb)
            forecast_loss = loss_fn(pred, yb)
            physics_loss = torch.tensor(0.0, device=device)
            if physics_active:
                physics_module = getattr(model, "physics", None)
                if physics_module is None or latent is None:
                    raise ValueError("Physics mode requires a model with forward_with_latents() and physics module.")
                physics_loss = physics_module(xraw, latent)["total"]
            loss = forecast_loss + effective_physics_weight * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_forecast_total += forecast_loss.item() * len(xb)
            train_physics_total += physics_loss.item() * len(xb)
            train_total += loss.item() * len(xb)
            train_count += len(xb)

        train_forecast_loss = float(train_forecast_total / max(train_count, 1))
        train_physics_loss = float(train_physics_total / max(train_count, 1))
        train_loss = float(train_total / max(train_count, 1))
        val_losses = (
            evaluate_torch_model(
                model,
                val_loader,
                device,
                loss_fn,
                physics_weight=effective_physics_weight,
                physics_active=physics_active,
            )
            if val_loader is not None
            else {"forecast_loss": train_forecast_loss, "physics_loss": train_physics_loss, "total_loss": train_loss}
        )
        val_loss = val_losses["total_loss"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_forecast_loss": train_forecast_loss,
                "train_physics_loss": train_physics_loss,
                "val_loss": val_loss,
                "val_forecast_loss": val_losses["forecast_loss"],
                "val_physics_loss": val_losses["physics_loss"],
                "lr": current_lr,
            }
        )
        scheduler.step(val_loss if np.isfinite(val_loss) else train_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_final_raw(
    *,
    model_name: str,
    model: nn.Module | None,
    split: SequenceSplit,
    scaler: FeatureScaler,
    target_mode: str,
    device: torch.device,
) -> np.ndarray:
    if split.empty():
        return np.empty((0, 0), dtype=np.float32)
    if model_name == "naive_lag1":
        return scaler.inverse_transform_array(split.X_scaled[:, -1, :], scaler.feature_names).astype(np.float32)
    if model_name == "seasonal_last_year":
        return split.seasonal_base_raw.copy()
    if model is None:
        raise ValueError(f"Model object is required for model_name={model_name!r}.")
    loader = DataLoader(
        TensorDataset(
            torch.tensor(split.X_scaled, dtype=torch.float32),
            torch.tensor(split.X_raw, dtype=torch.float32),
            torch.tensor(split.y_model_scaled, dtype=torch.float32),
        ),
        batch_size=128,
        shuffle=False,
    )
    pred_scaled = predict_torch_model(model, loader, device)
    if target_mode == "seasonal_residual":
        pred_scaled = pred_scaled + split.seasonal_base_scaled
    return scaler.inverse_transform_array(pred_scaled, scaler.feature_names).astype(np.float32)


def summarize_predictions(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    if y_true_raw.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mse": float("nan"), "r2": float("nan"), "n_samples": 0.0}
    mask = np.isfinite(y_true_raw.reshape(-1)) & np.isfinite(y_pred_raw.reshape(-1))
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "mse": float("nan"), "r2": float("nan"), "n_samples": 0.0}
    y_true = y_true_raw.reshape(-1)[mask]
    y_pred = y_pred_raw.reshape(-1)[mask]
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_samples": float(len(y_true)),
    }


def build_model(model_name: str, feature_count: int, *, hidden_size: int, num_layers: int, dropout: float) -> nn.Module:
    return CropNetModelFactory.create(
        model_name=model_name,
        input_dim=feature_count,
        output_dim=feature_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_pinn_model(
    model_name: str,
    feature_count: int,
    *,
    feature_names: list[str],
    hidden_size: int,
    num_layers: int,
    dropout: float,
    latent_state_dim: int = 3,
    physics_loss: str = "combined",
    physics_config: dict[str, Any] | str | Path | None = None,
) -> nn.Module:
    backbone = CropNetModelFactory.create(
        model_name=model_name,
        input_dim=feature_count,
        output_dim=feature_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    return PINNForecaster(
        backbone,
        feature_names,
        input_dim=feature_count,
        latent_state_dim=latent_state_dim,
        dropout=dropout,
        physics_loss=physics_loss,
        physics_config=physics_config,
    )


def fit_sarima_predictions(
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {"val": pd.DataFrame(), "test": pd.DataFrame()}

    working = frame.copy()
    working["date"] = _date_index(working)
    working = working.sort_values(["county_id", "crop_type", "date"]).reset_index(drop=True)
    train_end = pd.Timestamp(f"{max(train_years)}-12-01")
    forecast_end = pd.Timestamp(f"{max(test_years)}-12-01")
    val_dates = set(pd.to_datetime([f"{year}-{month:02d}-01" for year in val_years for month in range(1, 13)]))
    test_dates = set(pd.to_datetime([f"{year}-{month:02d}-01" for year in test_years for month in range(1, 13)]))
    prediction_rows: list[dict[str, Any]] = []

    for (county_id, crop_type), group in working.groupby(["county_id", "crop_type"], sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        group = group.set_index("date")
        selected_dates = pd.date_range(start=train_end + pd.offsets.MonthBegin(1), end=forecast_end, freq="MS")
        if len(selected_dates) == 0:
            continue

        feature_forecasts: dict[str, pd.Series] = {}
        for feature in feature_cols:
            series = group[feature].astype(float).asfreq("MS")
            train_series = series.loc[:train_end].dropna()
            if len(train_series) < 12:
                continue
            seasonal_order = (1, 0, 0, 12) if len(train_series) >= 36 else (0, 0, 0, 0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    model = sm.tsa.statespace.SARIMAX(
                        train_series,
                        order=(1, 0, 0),
                        seasonal_order=seasonal_order,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    result = model.fit(disp=False, maxiter=25)
                forecast = result.forecast(steps=len(selected_dates))
                forecast.index = selected_dates
            except Exception:
                fallback_value = float(train_series.iloc[-1])
                forecast = pd.Series(index=selected_dates, data=fallback_value)
            feature_forecasts[feature] = forecast

        if not feature_forecasts:
            continue

        for dt in selected_dates:
            row = {
                "county_id": str(county_id),
                "crop_type": str(crop_type),
                "date": dt,
            }
            for feature in feature_cols:
                row[feature] = float(feature_forecasts[feature].loc[dt]) if feature in feature_forecasts else np.nan
            prediction_rows.append(row)

    forecast_df = pd.DataFrame(prediction_rows)
    outputs: dict[str, pd.DataFrame] = {}
    for split_name, date_set in {"val": val_dates, "test": test_dates}.items():
        if forecast_df.empty:
            outputs[split_name] = pd.DataFrame()
            continue
        subset = forecast_df[forecast_df["date"].isin(date_set)].copy()
        outputs[split_name] = subset.reset_index(drop=True)
    return outputs


def ensemble_prediction(
    predictions: dict[str, np.ndarray],
    *,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    if not predictions:
        return np.empty((0, 0), dtype=np.float32)
    model_names = list(predictions.keys())
    first = predictions[model_names[0]]
    combined = np.zeros_like(first, dtype=np.float32)
    total = 0.0
    for name, pred in predictions.items():
        weight = 1.0 if weights is None else float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        combined += pred.astype(np.float32) * weight
        total += weight
    if total <= 0:
        return np.mean(np.stack([pred for pred in predictions.values()], axis=0), axis=0).astype(np.float32)
    return (combined / total).astype(np.float32)
