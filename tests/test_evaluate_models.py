"""Tests for evaluation and report utilities."""

from pathlib import Path

import pytest

from crop_fusion_ai.training.evaluate_models import (
    evaluate_image_folder_dataset,
    generate_error_distribution_plot,
    generate_predicted_vs_actual_plot,
)


def test_generate_predicted_vs_actual_plot_writes_png(tmp_path: Path) -> None:
    """Predicted-vs-actual plotting should create a PNG file."""
    output_path = tmp_path / "predicted_vs_actual.png"

    result_path = generate_predicted_vs_actual_plot(
        [1.0, 2.0, 3.0],
        [1.1, 1.9, 3.2],
        output_path,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_error_distribution_plot_writes_png(tmp_path: Path) -> None:
    """Error distribution plotting should create a PNG file."""
    output_path = tmp_path / "error_distribution.png"

    result_path = generate_error_distribution_plot([0.1, -0.2, 0.3], output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_predicted_vs_actual_plot_rejects_mismatched_lengths(
    tmp_path: Path,
) -> None:
    """Plot helpers should reject invalid paired inputs."""
    with pytest.raises(ValueError, match="same length"):
        generate_predicted_vs_actual_plot([1.0], [1.0, 2.0], tmp_path / "plot.png")


def test_evaluate_image_folder_dataset_placeholder_counts_files(
    tmp_path: Path,
) -> None:
    """Image folder placeholder should count files and return metric keys."""
    class_dir = tmp_path / "healthy"
    class_dir.mkdir()
    (class_dir / "sample.jpg").write_bytes(b"fake image bytes")

    metrics = evaluate_image_folder_dataset(tmp_path)

    assert metrics["image_count"] == 1.0
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
