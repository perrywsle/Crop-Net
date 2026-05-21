"""Matplotlib plotting utilities for model evaluation reports."""

import os
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(gettempdir()) / "crop_fusion_ai_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def save_predicted_vs_actual_plot(
    actual_values: list[float],
    predicted_values: list[float],
    output_path: Path,
) -> Path:
    """Save a predicted-vs-actual scatter plot as a PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lower_bound = min([*actual_values, *predicted_values])
    upper_bound = max([*actual_values, *predicted_values])

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(actual_values, predicted_values, alpha=0.75, edgecolors="black")
    axis.plot([lower_bound, upper_bound], [lower_bound, upper_bound], "r--")
    axis.set_title("Predicted vs Actual Yield")
    axis.set_xlabel("Actual yield")
    axis.set_ylabel("Predicted yield")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_error_distribution_plot(
    errors: list[float],
    output_path: Path,
) -> Path:
    """Save a prediction error histogram as a PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(errors, bins=min(12, max(3, len(errors))), edgecolor="black", alpha=0.8)
    axis.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
    axis.set_title("Yield Prediction Error Distribution")
    axis.set_xlabel("Prediction error")
    axis.set_ylabel("Frequency")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def create_flowchart_figure(
    width: float = 13.0,
    height: float = 8.0,
) -> tuple[Figure, Axes]:
    """Create a blank matplotlib figure configured for flowchart-style diagrams."""
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return figure, axis


def draw_flowchart_box(
    axis: Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#1f2937",
    fontsize: int = 11,
) -> None:
    """Draw a rounded flowchart box with centered text."""
    x_pos, y_pos = xy
    patch = FancyBboxPatch(
        (x_pos, y_pos),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        x_pos + width / 2,
        y_pos + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
        color="#111827",
    )


def draw_flowchart_arrow(
    axis: Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
) -> None:
    """Draw a directional arrow between two flowchart nodes."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#374151",
    )
    axis.add_patch(arrow)
    if text is not None:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.025,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color="#374151",
        )
