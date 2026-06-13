from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

try:  # Torch is only required when constructing or loading learned forecasters.
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = types.SimpleNamespace(load=None, device=object)
    nn = types.SimpleNamespace(Module=object)


def _require_torch() -> None:
    if getattr(torch, "load", None) is None:
        raise ModuleNotFoundError(
            "Torch is required for CropNet learned forecasting models. "
            "Install project requirements before loading checkpoints."
        )


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
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        _require_torch()
        super().__init__()
        self.hidden_size = hidden_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)[:, -1, :])

class GRUForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        _require_torch()
        super().__init__()
        self.hidden_size = hidden_size
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=gru_dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)[:, -1, :])

class TransformerEncoderForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0, max_seq_len: int = 512) -> None:
        _require_torch()
        super().__init__()
        self.hidden_size = hidden_size
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        hidden = self.input_proj(x) + self.positional[:, :seq_len, :]
        return self.encoder(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)[:, -1, :])

class TinyMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, conv_kernel: int = 3) -> None:
        _require_torch()
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        uc = self.conv(u.transpose(1, 2))[..., : u.shape[1]].transpose(1, 2)
        uc = torch.nn.functional.silu(uc)
        params = self.x_proj(uc)
        Bp, Cp, delta = torch.split(params, [self.d_state, self.d_state, self.d_model], dim=-1)
        delta = torch.nn.functional.softplus(delta)
        A = -torch.exp(self.A_log)
        state = torch.zeros(x.shape[0], self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):
            dt = delta[:, t, :].unsqueeze(-1)
            At = torch.exp(dt * A.unsqueeze(0))
            Bt = Bp[:, t, :].unsqueeze(1)
            ut = uc[:, t, :].unsqueeze(-1)
            state = At * state + dt * Bt * ut
            Ct = Cp[:, t, :].unsqueeze(1)
            yt = (state * Ct).sum(dim=-1) + self.D * uc[:, t, :]
            ys.append(yt)
        y = torch.stack(ys, dim=1)
        y = y * torch.sigmoid(z)
        return self.norm(residual + self.out_proj(y))


class MambaStyleForecaster(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = 64, d_state: int = 32, num_layers: int = 1, dropout: float = 0.0) -> None:
        _require_torch()
        super().__init__()
        self.hidden_size = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TinyMambaBlock(d_model=d_model, d_state=d_state) for _ in range(max(1, num_layers))])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, output_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.dropout(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)[:, -1, :])

def infer_architecture_from_state_dict(model_name: str, state_dict: dict[str, Any]) -> dict[str, int | float]:
    normalized_name = "tiny_mamba_ssm" if model_name == "gamma_ssm" else model_name
    if normalized_name == "lstm":
        return {
            "input_dim": int(state_dict["lstm.weight_ih_l0"].shape[1]),
            "output_dim": int(state_dict["head.2.bias"].shape[0]),
            "hidden_size": int(state_dict["lstm.weight_hh_l0"].shape[1]),
            "num_layers": len([key for key in state_dict if key.startswith("lstm.weight_ih_l")]),
            "dropout": 0.0,
        }
    if normalized_name == "gru":
        return {
            "input_dim": int(state_dict["gru.weight_ih_l0"].shape[1]),
            "output_dim": int(state_dict["head.2.bias"].shape[0]),
            "hidden_size": int(state_dict["gru.weight_hh_l0"].shape[1]),
            "num_layers": len([key for key in state_dict if key.startswith("gru.weight_ih_l")]),
            "dropout": 0.0,
        }
    if normalized_name == "transformer_encoder":
        return {
            "input_dim": int(state_dict["input_proj.weight"].shape[1]),
            "output_dim": int(state_dict["head.2.bias"].shape[0]),
            "hidden_size": int(state_dict["input_proj.weight"].shape[0]),
            "num_layers": len({key.split(".")[2] for key in state_dict if key.startswith("encoder.layers.")}) or 1,
            "dropout": 0.0,
        }
    if normalized_name == "tiny_mamba_ssm":
        block_keys = [key for key in state_dict if key.startswith("blocks.")]
        if not block_keys and any(key.startswith("block.") for key in state_dict):
            # Legacy checkpoints sometimes use the singular prefix.
            block_keys = [key for key in state_dict if key.startswith("block.")]
        d_model_key = next((key for key in ("input_proj.weight", "out_proj.weight", "in_proj.weight", "proj.weight") if key in state_dict), None)
        if d_model_key is None:
            d_model_key = next(
                (key for key in state_dict if key.startswith("blocks.") and key.endswith(".weight") and len(getattr(state_dict[key], "shape", ())) == 2),
                None,
            )
        if d_model_key is None:
            raise KeyError("Could not infer gamma_ssm / tiny_mamba_ssm input dimension from checkpoint")
        head_key = "head.2.bias" if "head.2.bias" in state_dict else "head.bias"
        if head_key not in state_dict:
            raise KeyError("Could not infer gamma_ssm / tiny_mamba_ssm output dimension from checkpoint")
        model_width_key = next((key for key in ("input_proj.weight", "blocks.0.out_proj.weight", "block.out_proj.weight") if key in state_dict), None)
        if model_width_key is None:
            model_width_key = d_model_key
        return {
            "input_dim": int(state_dict[d_model_key].shape[1]),
            "output_dim": int(state_dict[head_key].shape[0]),
            "hidden_size": int(state_dict[model_width_key].shape[0]),
            "num_layers": len({key.split(".")[1] for key in block_keys}) or 1,
            "dropout": 0.0,
        }
    raise ValueError(f"Architecture inference not implemented for model '{model_name}'")

class CropNetModelFactory:
    @staticmethod
    def create(model_name: str, input_dim: int, output_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0, seq_len: int = 6, legacy_script_path: str | Path | None = None) -> nn.Module:
        normalized_name = "tiny_mamba_ssm" if model_name == "gamma_ssm" else model_name
        if normalized_name == "lstm":
            return LSTMForecaster(input_dim, output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        if normalized_name == "gru":
            return GRUForecaster(input_dim, output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        if normalized_name == "transformer_encoder":
            return TransformerEncoderForecaster(input_dim, output_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        if normalized_name == "tiny_mamba_ssm":
            return MambaStyleForecaster(input_dim, output_dim, d_model=hidden_size, d_state=32, num_layers=num_layers, dropout=dropout)
        raise ValueError(f"Model '{model_name}' is not supported by the factory.")

    @staticmethod
    def load_checkpoint(checkpoint_path: str | Path, model_name: str | None = None, device: str = "cpu", legacy_script_path: str | Path | None = None) -> nn.Module:
        _require_torch()
        checkpoint_path = Path(checkpoint_path)
        inferred_name = model_name or checkpoint_path.name.replace("_best.pt", "")
        normalized_name = "tiny_mamba_ssm" if inferred_name == "gamma_ssm" else inferred_name
        state = torch.load(checkpoint_path, map_location=device)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        if normalized_name == "tiny_mamba_ssm":
            normalized_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                if key.startswith("block."):
                    new_key = "blocks.0." + key.split("block.", 1)[1]
                if new_key.startswith("head.1."):
                    new_key = "head.2." + new_key.split("head.1.", 1)[1]
                normalized_state_dict[new_key] = value
            state_dict = normalized_state_dict
        params = infer_architecture_from_state_dict(normalized_name, state_dict)
        model = CropNetModelFactory.create(normalized_name, **params)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
