"""Training skeleton for the plant health image classifier."""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_CLASS_FOLDERS = ("healthy", "mild_disease", "severe_disease")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for image model training."""
    parser = argparse.ArgumentParser(
        description="Train the plant health image classifier skeleton."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/images"),
        help="Folder containing class subfolders such as healthy/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/image_model"),
        help="Directory where image model metadata will be saved.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for demo training samples; real training TODO.",
    )
    return parser.parse_args(argv)


def count_images_by_class(dataset_path: Path) -> dict[str, int]:
    """Count supported image files in each expected class folder."""
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
        raise FileNotFoundError(msg)
    if not dataset_path.is_dir():
        msg = f"Dataset path is not a directory: {dataset_path}"
        raise NotADirectoryError(msg)

    class_counts: dict[str, int] = {}
    missing_classes: list[str] = []

    for class_name in EXPECTED_CLASS_FOLDERS:
        class_dir = dataset_path / class_name
        if not class_dir.is_dir():
            missing_classes.append(class_name)
            continue
        class_counts[class_name] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if missing_classes:
        msg = f"Missing expected class folders: {', '.join(missing_classes)}"
        raise ValueError(msg)

    return class_counts


def save_training_metadata(
    output_dir: Path,
    dataset_path: Path,
    class_counts: dict[str, int],
    max_samples: int | None,
) -> Path:
    """Save metadata describing the placeholder training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "model_type": "placeholder_training_skeleton",
        "architecture_todo": "EfficientNet-B0",
        "dataset_path": str(dataset_path),
        "class_counts": class_counts,
        "max_samples": max_samples,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "notes": [
            "TODO: add PyTorch Dataset and transforms.",
            "TODO: add EfficientNet-B0 classifier head.",
            "TODO: add train/validation split and metrics.",
            "TODO: save trained weights separately from metadata.",
        ],
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return metadata_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the image training skeleton."""
    args = parse_args(argv)
    dataset_path = args.dataset
    class_counts = count_images_by_class(dataset_path)

    print("Plant health image dataset summary")
    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count} images")

    print("\nTODO: implement EfficientNet-B0 training here.")
    metadata_path = save_training_metadata(
        output_dir=args.output_dir,
        dataset_path=dataset_path,
        class_counts=class_counts,
        max_samples=args.max_samples,
    )
    print(f"Saved training metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
