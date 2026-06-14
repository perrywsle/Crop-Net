from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def plot_prediction_overlay(frame: pd.DataFrame, feature: str, output_path: str | Path, title: str | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    for column in frame.columns:
        if column == "month":
            continue
        plt.plot(frame["month"], frame[column], marker="o", label=column)
    plt.xlabel("Month")
    plt.ylabel(feature)
    plt.title(title or feature)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_loss_curves(
    history_csv: str | Path,
    output_path: str | Path,
    train_col: str = "train_loss",
    val_col: str = "val_loss",
    *,
    physics_start_epoch: int | None = None,
    hide_before_epoch: int | None = None,
    title: str = "Training History",
) -> None:
    history = pd.read_csv(history_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hide_before_epoch is not None:
        history = history[history["epoch"] > hide_before_epoch].copy()
    if history.empty:
        return
    plt.figure(figsize=(8, 4))
    x_values = history["epoch"] if "epoch" in history.columns else history.index + 1
    if train_col in history.columns:
        plt.plot(x_values, history[train_col], label=train_col)
    if val_col in history.columns:
        plt.plot(x_values, history[val_col], label=val_col)
    if physics_start_epoch is not None:
        plt.axvline(physics_start_epoch + 0.5, color="#444444", linestyle="--", linewidth=1.5, alpha=0.8)
        plt.text(
            physics_start_epoch + 0.65,
            plt.ylim()[1] * 0.95,
            "physics loss start here",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color="#444444",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
