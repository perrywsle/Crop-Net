from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])

class TransformerEncoderForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0, max_seq_len: int = 512) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.positional = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        hidden = self.input_proj(x) + self.positional[:, :seq_len, :]
        encoded = self.encoder(hidden)
        return self.head(encoded[:, -1, :])

def _legacy_script_path(custom_path: str | Path | None = None) -> Path:
    if custom_path is not None:
        return Path(custom_path)
    return Path(__file__).resolve().parents[2] / "scripts" / "research" / "cropnet_feature_forecasting_v12_server.py"

def load_legacy_module(script_path: str | Path | None = None):
    path = _legacy_script_path(script_path)
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("cropnet_research_legacy", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def infer_architecture_from_state_dict(model_name: str, state_dict: dict[str, Any]) -> dict[str, int | float]:
    if model_name == "lstm":
        return {
            "input_dim": int(state_dict["lstm.weight_ih_l0"].shape[1]),
            "output_dim": int(state_dict["head.2.bias"].shape[0]),
            "hidden_size": int(state_dict["lstm.weight_hh_l0"].shape[1]),
            "num_layers": len([key for key in state_dict if key.startswith("lstm.weight_ih_l")]),
            "dropout": 0.0,
        }
    if model_name == "gru":
        return {
            "input_dim": int(state_dict["gru.weight_ih_l0"].shape[1]),
            "output_dim": int(state_dict["head.2.bias"].shape[0]),
            "hidden_size": int(state_dict["gru.weight_hh_l0"].shape[1]),
            "num_layers": len([key for key in state_dict if key.startswith("gru.weight_ih_l")]),
            "dropout": 0.0,
        }
    if model_name == "transformer_encoder":
        return {
            "input_dim": int(state_dict["input_proj.weight"].shape[1]),
            "output_dim": int(state_dict["head.2.bias"].shape[0]),
            "hidden_size": int(state_dict["input_proj.weight"].shape[0]),
            "num_layers": len({key.split(".")[2] for key in state_dict if key.startswith("encoder.layers.")}) or 1,
            "dropout": 0.0,
        }
    raise ValueError(f"Architecture inference not implemented for model '{model_name}'")

class CropNetModelFactory:
    @staticmethod
    def create(model_name: str, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0, seq_len: int = 6, legacy_script_path: str | Path | None = None) -> nn.Module:
        if model_name == "lstm":
            return LSTMForecaster(input_dim, output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        if model_name == "gru":
            return GRUForecaster(input_dim, output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        if model_name == "transformer_encoder":
            return TransformerEncoderForecaster(input_dim, output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        legacy = load_legacy_module(legacy_script_path)
        if legacy is None:
            raise ValueError(f"Model '{model_name}' requires the legacy research script to be present.")
        return legacy.build_learned_model(model_name, input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    @staticmethod
    def load_checkpoint(checkpoint_path: str | Path, model_name: str | None = None, device: str = "cpu", legacy_script_path: str | Path | None = None) -> nn.Module:
        checkpoint_path = Path(checkpoint_path)
        inferred_name = model_name or checkpoint_path.name.replace("_best.pt", "")
        legacy = load_legacy_module(legacy_script_path)
        if legacy is not None:
            try:
                model = legacy.load_trained_model_for_eval(checkpoint_path, inferred_name, torch.device(device))
                model.eval()
                return model
            except Exception:
                pass
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        if inferred_name == "tiny_mamba_ssm":
            raise ValueError("tiny_mamba_ssm checkpoint loading requires the legacy research script. It is present in this repo under scripts/research/.")
        params = infer_architecture_from_state_dict(inferred_name, state_dict)
        model = CropNetModelFactory.create(inferred_name, **params)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
