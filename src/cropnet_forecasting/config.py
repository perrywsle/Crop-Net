from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

@dataclass(slots=True)
class ForecastingConfig:
    model_name: str = "lstm"
    target_mode: str = "seasonal_residual"
    loss_mode: str = "raw_mse"
    feature_group: str = "all"
    seq_len: int = 6
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 10
    known_months: list[int] = field(default_factory=lambda: [0, 1, 3, 6, 9])
    blank_fill_year: int = 2021
    checkpoint_path: str | None = None
    scaler_path: str | None = None
    run_dir: str | None = None
    output_dir: str | None = None
    python_bin: str = "python"
    legacy_script_path: str = "scripts/research/cropnet_feature_forecasting_v12_server.py"
    device: str = "cpu"
    state_codes: list[str] = field(default_factory=lambda: ["IA"])
    years: list[int] = field(default_factory=lambda: [2017, 2018, 2019, 2020, 2021])
    train_years: list[int] = field(default_factory=lambda: [2017, 2018, 2019])
    val_years: list[int] = field(default_factory=lambda: [2020])
    test_years: list[int] = field(default_factory=lambda: [2021])
    max_counties: int = 30
    extra_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "ForecastingConfig":
        return cls(**load_config(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)

def save_config(config: ForecastingConfig | dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict() if isinstance(config, ForecastingConfig) else config
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
