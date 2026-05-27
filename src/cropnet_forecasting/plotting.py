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

def plot_loss_curves(history_csv: str | Path, output_path: str | Path, train_col: str = "train_loss", val_col: str = "val_loss") -> None:
    history = pd.read_csv(history_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if train_col in history.columns:
        plt.plot(history.index + 1, history[train_col], label=train_col)
    if val_col in history.columns:
        plt.plot(history.index + 1, history[val_col], label=val_col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
