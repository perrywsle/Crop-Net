"""Generate architecture flowcharts as PNG figures."""

from pathlib import Path

from matplotlib.figure import Figure

from crop_fusion_ai.utils.plotting import (
    create_flowchart_figure,
    draw_flowchart_arrow,
    draw_flowchart_box,
)

DATA_SOURCING_FLOWCHART_PATH = Path("reports/figures/data_sourcing_pipeline.png")
INFERENCE_FLOWCHART_PATH = Path("reports/figures/inference_architecture_pipeline.png")


def generate_data_sourcing_flowchart(
    output_path: Path = DATA_SOURCING_FLOWCHART_PATH,
) -> Path:
    """Generate a PNG flowchart for the CropNet data sourcing pipeline."""
    figure, axis = create_flowchart_figure()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    draw_flowchart_box(
        axis,
        xy=(0.05, 0.74),
        width=0.18,
        height=0.12,
        text="CropNet Dataset\n(Hugging Face + official API)",
        facecolor="#dbeafe",
    )
    draw_flowchart_box(
        axis,
        xy=(0.30, 0.74),
        width=0.16,
        height=0.12,
        text="Bounded Query\ncrop, year, FIPS,\nmodality",
        facecolor="#e0f2fe",
    )
    draw_flowchart_box(
        axis,
        xy=(0.53, 0.74),
        width=0.18,
        height=0.12,
        text="Selective Download\nSentinel-2 / HRRR /\nUSDA yield labels",
        facecolor="#e9d5ff",
    )
    draw_flowchart_box(
        axis,
        xy=(0.77, 0.74),
        width=0.18,
        height=0.12,
        text="Local Cache\nH5 tiles, weather tables,\nyield records",
        facecolor="#fce7f3",
    )

    draw_flowchart_box(
        axis,
        xy=(0.12, 0.42),
        width=0.22,
        height=0.12,
        text="Satellite Image Extraction\n224x224 field tiles\nfor MobileNet",
        facecolor="#dcfce7",
    )
    draw_flowchart_box(
        axis,
        xy=(0.39, 0.42),
        width=0.22,
        height=0.12,
        text="Weather Time-Series\nprevious HRRR records\nsummarized or sequenced",
        facecolor="#fef3c7",
    )
    draw_flowchart_box(
        axis,
        xy=(0.66, 0.42),
        width=0.22,
        height=0.12,
        text="USDA Ground Truth\ncounty/year crop yield\nsupervision target",
        facecolor="#fee2e2",
    )

    draw_flowchart_box(
        axis,
        xy=(0.24, 0.12),
        width=0.22,
        height=0.12,
        text="Training Table / Samples\nimage path + weather +\nyield label",
        facecolor="#ddd6fe",
    )
    draw_flowchart_box(
        axis,
        xy=(0.55, 0.12),
        width=0.24,
        height=0.12,
        text=(
            "Model Training Stage\n"
            "MobileNet features + weather\n"
            "model + yield predictor"
        ),
        facecolor="#fde68a",
    )

    draw_flowchart_arrow(axis, start=(0.23, 0.80), end=(0.30, 0.80))
    draw_flowchart_arrow(axis, start=(0.46, 0.80), end=(0.53, 0.80))
    draw_flowchart_arrow(axis, start=(0.71, 0.80), end=(0.77, 0.80))
    draw_flowchart_arrow(axis, start=(0.83, 0.74), end=(0.77, 0.54))
    draw_flowchart_arrow(axis, start=(0.83, 0.74), end=(0.50, 0.54))
    draw_flowchart_arrow(axis, start=(0.83, 0.74), end=(0.23, 0.54))
    draw_flowchart_arrow(axis, start=(0.23, 0.42), end=(0.35, 0.24))
    draw_flowchart_arrow(axis, start=(0.50, 0.42), end=(0.35, 0.24))
    draw_flowchart_arrow(axis, start=(0.77, 0.42), end=(0.35, 0.24))
    draw_flowchart_arrow(axis, start=(0.46, 0.18), end=(0.55, 0.18))

    axis.set_title("CropNet Data Sourcing Pipeline", fontsize=17, pad=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    return _close_and_return(figure, output_path)


def generate_inference_architecture_flowchart(
    output_path: Path = INFERENCE_FLOWCHART_PATH,
) -> Path:
    """Generate a PNG flowchart for the model inference architecture."""
    figure, axis = create_flowchart_figure(width=13.5, height=7.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    draw_flowchart_box(
        axis,
        xy=(0.04, 0.68),
        width=0.18,
        height=0.14,
        text="Input A\nClose-up plant image\n(optional leaf/plant view)",
        facecolor="#dbeafe",
    )
    draw_flowchart_box(
        axis,
        xy=(0.04, 0.30),
        width=0.18,
        height=0.14,
        text="Input B\nField-scale CropNet\nsatellite tiles",
        facecolor="#bfdbfe",
    )
    draw_flowchart_box(
        axis,
        xy=(0.29, 0.68),
        width=0.18,
        height=0.14,
        text="Plant Image Classifier\ncurrent placeholder,\nlater disease model",
        facecolor="#dcfce7",
    )
    draw_flowchart_box(
        axis,
        xy=(0.29, 0.30),
        width=0.18,
        height=0.14,
        text="MobileNet Feature\nExtractor\nfield-condition embedding",
        facecolor="#bbf7d0",
    )
    draw_flowchart_box(
        axis,
        xy=(0.54, 0.49),
        width=0.18,
        height=0.14,
        text="Weather Module\nprevious HRRR records\nsummary / time series",
        facecolor="#fef3c7",
    )
    draw_flowchart_box(
        axis,
        xy=(0.77, 0.49),
        width=0.18,
        height=0.14,
        text="Yield Predictor\nfused image + weather\n+ crop metadata",
        facecolor="#fde68a",
    )
    draw_flowchart_box(
        axis,
        xy=(0.77, 0.14),
        width=0.18,
        height=0.14,
        text="Output\npredicted yield\n+ warnings",
        facecolor="#fecaca",
    )
    draw_flowchart_box(
        axis,
        xy=(0.54, 0.79),
        width=0.18,
        height=0.12,
        text="Crop Metadata\ncrop type, region,\nyear, age",
        facecolor="#e9d5ff",
    )

    draw_flowchart_arrow(axis, start=(0.22, 0.75), end=(0.29, 0.75))
    draw_flowchart_arrow(axis, start=(0.22, 0.37), end=(0.29, 0.37))
    draw_flowchart_arrow(
        axis,
        start=(0.47, 0.75),
        end=(0.77, 0.56),
        text="health_score",
    )
    draw_flowchart_arrow(axis, start=(0.47, 0.37), end=(0.77, 0.56), text="embedding")
    draw_flowchart_arrow(axis, start=(0.72, 0.85), end=(0.77, 0.60))
    draw_flowchart_arrow(axis, start=(0.72, 0.56), end=(0.77, 0.56))
    draw_flowchart_arrow(axis, start=(0.86, 0.49), end=(0.86, 0.28))

    axis.text(
        0.48,
        0.12,
        (
            "Segmentation note:\n"
            "not currently implemented.\n"
            "Useful if you want tile-level crop masking,\n"
            "weed/background removal, or field-part analysis before MobileNet."
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f9fafb",
            "edgecolor": "#9ca3af",
        },
    )

    axis.set_title("Inference Architecture Pipeline", fontsize=17, pad=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    return _close_and_return(figure, output_path)


def _close_and_return(figure: Figure, output_path: Path) -> Path:
    """Close a matplotlib figure and return the saved output path."""
    import matplotlib.pyplot as plt

    plt.close(figure)
    return output_path


def main() -> int:
    """Generate both architecture flowcharts."""
    data_path = generate_data_sourcing_flowchart()
    inference_path = generate_inference_architecture_flowchart()
    print(data_path)
    print(inference_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
