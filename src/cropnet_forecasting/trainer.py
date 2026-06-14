from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ForecastingConfig

@dataclass
class CropNetTrainer:
    config: ForecastingConfig
    repo_root: Path

    @classmethod
    def from_config(cls, config_path: str | Path) -> "CropNetTrainer":
        config_path = Path(config_path)
        config = ForecastingConfig.from_path(config_path)
        return cls(config=config, repo_root=config_path.resolve().parents[1])

    def _script_path(self) -> Path:
        return (self.repo_root / self.config.legacy_script_path).resolve()

    def _base_command(self) -> list[str]:
        cfg = self.config
        output_dir = cfg.output_dir or cfg.run_dir or "outputs/experiments/handover_run"
        command = [
            cfg.python_bin or sys.executable,
            str(self._script_path()),
            "--full-run",
            "--run-name", Path(output_dir).name,
            "--experiment-root", str(Path(output_dir).parent),
            "--state-codes", *cfg.state_codes,
            "--years", *[str(year) for year in cfg.years],
            "--train-years", *[str(year) for year in cfg.train_years],
            "--val-years", *[str(year) for year in cfg.val_years],
            "--test-years", *[str(year) for year in cfg.test_years],
            "--quarters", "Q1", "Q2", "Q3", "Q4",
            "--seq-len", str(cfg.seq_len),
            "--models", cfg.model_name,
            "--target-mode", cfg.target_mode,
            "--loss-mode", cfg.loss_mode,
            "--feature-groups", cfg.feature_group,
            "--max-counties", str(cfg.max_counties),
            "--batch-size", str(cfg.batch_size),
            "--learning-rate", str(cfg.learning_rate),
            "--hidden-size", str(cfg.hidden_size),
            "--num-layers", str(cfg.num_layers),
            "--dropout", str(cfg.dropout),
            "--weight-decay", str(cfg.weight_decay),
            "--epochs", str(cfg.max_epochs),
            "--patience", str(cfg.patience),
            "--blank-fill-year", str(cfg.blank_fill_year),
            "--blank-fill-known-months", *[str(month) for month in cfg.known_months],
        ]
        if cfg.target_mode == "seasonal_residual":
            command.append("--blank-fill-residual-seasonal")
        return command

    def fit(self) -> subprocess.CompletedProcess:
        return subprocess.run(self._base_command(), check=False, cwd=self.repo_root)

    def evaluate(self) -> subprocess.CompletedProcess:
        return subprocess.run(self._base_command() + ["--resume"], check=False, cwd=self.repo_root)

    def evaluate_blank_fill(self) -> subprocess.CompletedProcess:
        command = self._base_command() + ["--run-blank-fill-eval", "--strict-blank-fill-no-future-fill"]
        return subprocess.run(command, check=False, cwd=self.repo_root)

    def save_artifacts(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = output_dir / "trainer_command.txt"
        snapshot.write_text(" ".join(shlex.quote(part) for part in self._base_command()), encoding="utf-8")
        return snapshot
