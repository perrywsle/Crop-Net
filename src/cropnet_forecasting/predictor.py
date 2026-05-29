from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .blank_fill import rollout_autoregressive, rollout_blank_fill
from .config import load_config
from .features import selected_feature_columns
from .models import CropNetModelFactory
from .scaling import FeatureScaler

@dataclass
class BlankFillPredictor:
    model: torch.nn.Module
    scaler: FeatureScaler
    feature_names: list[str]
    model_name: str
    target_mode: str
    seq_len: int
    device: str = "cpu"
    last_predictions: pd.DataFrame | None = None

    @classmethod
    def from_artifacts(cls, checkpoint_path: str | Path, scaler_path: str | Path, config_path: str | Path, device: str | None = None) -> "BlankFillPredictor":
        config = load_config(config_path)
        model_name = str(config.get("model_name") or config.get("model") or Path(checkpoint_path).name.replace("_best.pt", ""))
        feature_group = str(config.get("feature_group", "all"))
        feature_names = selected_feature_columns(feature_group)
        scaler = FeatureScaler.from_csv(scaler_path).subset(feature_names)
        runtime_device = device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        model = CropNetModelFactory.load_checkpoint(checkpoint_path, model_name=model_name, device=runtime_device)
        return cls(
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            model_name=model_name,
            target_mode=str(config.get("target_mode", "raw")),
            seq_len=int(config.get("seq_len", config.get("lookback_months", 6))),
            device=str(runtime_device),
        )

    def predict_next(self, input_window: np.ndarray | pd.DataFrame, seasonal_base: np.ndarray | pd.Series | None = None) -> np.ndarray:
        if isinstance(input_window, pd.DataFrame):
            window = input_window[self.feature_names].to_numpy(dtype=float)
        else:
            window = np.asarray(input_window, dtype=float)
        if window.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected window width {len(self.feature_names)}, got {window.shape[1]}")
        if window.shape[0] != self.seq_len:
            raise ValueError(f"Expected window length {self.seq_len}, got {window.shape[0]}")
        scaled_window = self.scaler.transform_array(window, self.feature_names)
        tensor = torch.tensor(scaled_window[None, :, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predicted_scaled = self.model(tensor).detach().cpu().numpy()[0]
        if self.target_mode == "seasonal_residual":
            seasonal_raw = window[-1] if seasonal_base is None else np.asarray(seasonal_base, dtype=float)
            seasonal_scaled = self.scaler.transform_array(seasonal_raw[None, :], self.feature_names)[0]
            predicted_scaled = seasonal_scaled + predicted_scaled
        return self.scaler.inverse_transform_array(predicted_scaled[None, :], self.feature_names)[0]

    def predict_remaining_year(self, monthly_features: pd.DataFrame, year: int, known_months: int) -> pd.DataFrame:
        result = rollout_blank_fill(self, monthly_features=monthly_features, year=year, known_months=known_months)
        self.last_predictions = result.predictions
        return result.predictions

    def predict_future_months(
        self,
        monthly_features: pd.DataFrame,
        horizon: int = 12,
        progress=None,
    ) -> pd.DataFrame:
        result = rollout_autoregressive(
            self,
            monthly_features=monthly_features,
            horizon=horizon,
            progress=progress,
        )
        self.last_predictions = result.predictions
        return result.predictions

    def save_predictions(self, path: str | Path) -> None:
        if self.last_predictions is None:
            raise ValueError("No predictions available. Run predict_remaining_year first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".parquet":
            self.last_predictions.to_parquet(path, index=False)
        else:
            self.last_predictions.to_csv(path, index=False)
