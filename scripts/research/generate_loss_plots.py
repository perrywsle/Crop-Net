from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PREFERRED_ORDER = ["lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True)
    args = parser.parse_args()

    artifacts = Path(args.artifacts_dir)
    plot_dir = artifacts / "plots_report"
    plot_dir.mkdir(parents=True, exist_ok=True)

    history_paths = sorted(artifacts.glob("*_history.csv"), key=lambda path: (PREFERRED_ORDER.index(path.stem.replace("_history", "")) if path.stem.replace("_history", "") in PREFERRED_ORDER else 999, path.name))
    histories = {}
    for path in history_paths:
        model_name = path.stem.replace("_history", "")
        hist = pd.read_csv(path)
        histories[model_name] = hist

        best_idx = int(hist["val_loss"].idxmin())
        best_epoch = int(hist.loc[best_idx, "epoch"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        ax.plot(hist["epoch"], hist["val_loss"], label="val_loss")
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1, label=f"best={best_epoch}")
        ax.set_title(f"{model_name} loss curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = plot_dir / f"{model_name}_loss_curve.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)

    if len(histories) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for idx, model_name in enumerate(histories):
            hist = histories[model_name]
            color = color_cycle[idx % len(color_cycle)]
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


if __name__ == "__main__":
    main()
