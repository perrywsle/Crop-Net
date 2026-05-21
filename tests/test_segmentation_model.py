"""Tests for the segmentation PoC model."""

from pathlib import Path

from PIL import Image

from crop_fusion_ai.models import PlantSegmentationModel, SegmentationResult
from crop_fusion_ai.training.train_segmentation_model import train_segmentation_model


def test_segmenter_fallback_saves_mask_and_overlay(tmp_path: Path) -> None:
    """Fallback segmentation should create visible artifacts for the UI."""
    image_path = tmp_path / "leaf.png"
    output_dir = tmp_path / "segmentation"
    Image.new("RGB", (24, 24), color=(40, 180, 50)).save(image_path)

    result = PlantSegmentationModel(model_path=None).segment(
        image_path,
        output_dir=output_dir,
    )

    assert isinstance(result, SegmentationResult)
    assert result.mask_path.exists()
    assert result.overlay_path.exists()
    assert 0.0 <= result.coverage_ratio <= 1.0
    assert result.used_trained_model is False


def test_segmenter_rejects_missing_image(tmp_path: Path) -> None:
    """Missing image paths should fail clearly."""
    segmenter = PlantSegmentationModel(model_path=None)

    try:
        segmenter.segment(tmp_path / "missing.png")
    except FileNotFoundError as exc:
        assert "Image file does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_train_segmentation_model_saves_loadable_checkpoint(tmp_path: Path) -> None:
    """Tiny training should produce a checkpoint loadable by the segmenter."""
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(3):
        Image.new("RGB", (24, 24), color=(40 + index * 20, 150, 60)).save(
            image_root / f"leaf_{index}.png"
        )
    model_path = tmp_path / "segmentation_model.pt"
    metadata_path = tmp_path / "segmentation_metadata.json"

    metadata = train_segmentation_model(
        image_root=image_root,
        model_path=model_path,
        metadata_path=metadata_path,
        epochs=1,
        batch_size=1,
        max_samples=3,
        input_size=16,
        device="cpu",
    )
    result = PlantSegmentationModel(model_path=model_path).segment(
        image_root / "leaf_0.png",
        output_dir=tmp_path / "outputs",
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert metadata["sample_count"] == 3
    assert result.used_trained_model is True
    assert result.overlay_path.exists()
