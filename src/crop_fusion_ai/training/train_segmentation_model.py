"""Train a tiny segmentation model for the UI proof of concept."""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from crop_fusion_ai.models.segmentation_model import (
    DEFAULT_SEGMENTATION_MODEL_PATH,
    build_tiny_segmentation_network,
    image_to_tensor,
    pseudo_mask_tensor,
)
from crop_fusion_ai.training.train_image_model import IMAGE_EXTENSIONS

DEFAULT_IMAGE_ROOT = Path("data/processed/cropnet_images")
DEFAULT_METADATA_PATH = Path("models/image_model/segmentation_metadata.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse tiny segmentation training arguments."""
    parser = argparse.ArgumentParser(
        description="Train a low-memory segmentation PoC from local crop images."
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_SEGMENTATION_MODEL_PATH,
    )
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def train_segmentation_model(
    *,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    model_path: Path = DEFAULT_SEGMENTATION_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    epochs: int = 2,
    batch_size: int = 2,
    max_samples: int = 16,
    input_size: int = 64,
    device: str = "cuda",
) -> dict[str, float | int | str]:
    """Train a tiny CNN against deterministic pseudo masks and save weights."""
    if epochs < 1:
        msg = "epochs must be at least 1"
        raise ValueError(msg)
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)
    image_paths = collect_training_images(image_root, max_samples=max_samples)
    torch, nn, optim = _import_training_stack()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    size = (input_size, input_size)
    model = build_tiny_segmentation_network(nn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        for batch_paths in _batched(image_paths, batch_size):
            images, masks = _load_batch(batch_paths, size=size, device=device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": size,
            "model_type": "tiny_segmentation_cnn",
        },
        model_path,
    )
    final_loss = losses[-1] if losses else 0.0
    metadata: dict[str, float | int | str] = {
        "model_type": "tiny_segmentation_cnn",
        "model_path": str(model_path),
        "image_root": str(image_root),
        "sample_count": len(image_paths),
        "epochs": epochs,
        "batch_size": batch_size,
        "input_size": input_size,
        "device": device,
        "final_loss": final_loss,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "mask_source": "deterministic pseudo masks from image color/brightness",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return metadata


def collect_training_images(image_root: Path, *, max_samples: int) -> list[Path]:
    """Collect a bounded set of local images for tiny segmentation training."""
    if not image_root.exists():
        msg = f"Image root does not exist: {image_root}"
        raise FileNotFoundError(msg)
    if max_samples < 1:
        msg = "max_samples must be at least 1"
        raise ValueError(msg)
    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        msg = f"No supported images found under {image_root}"
        raise ValueError(msg)
    return image_paths[:max_samples]


def _load_batch(
    image_paths: Sequence[Path],
    *,
    size: tuple[int, int],
    device: str,
) -> tuple[Any, Any]:
    """Load one tiny batch as tensors."""
    torch, _, _ = _import_training_stack()
    image_tensors: list[Any] = []
    mask_tensors: list[Any] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        image_tensors.append(image_to_tensor(rgb_image, size))
        mask_tensors.append(pseudo_mask_tensor(rgb_image, size))
    return torch.stack(image_tensors).to(device), torch.stack(mask_tensors).to(device)


def _batched(values: Sequence[Path], batch_size: int) -> list[list[Path]]:
    """Split paths into small batches."""
    return [
        list(values[index : index + batch_size])
        for index in range(0, len(values), batch_size)
    ]


def _import_training_stack() -> tuple[Any, Any, Any]:
    """Import torch training modules lazily."""
    try:
        import torch
        from torch import nn, optim
    except ImportError as exc:
        msg = "Segmentation training requires torch; install the vision extra."
        raise ImportError(msg) from exc
    return torch, nn, optim


def main(argv: Sequence[str] | None = None) -> int:
    """Train and save the tiny segmentation model."""
    args = parse_args(argv)
    metadata = train_segmentation_model(
        image_root=args.image_root,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        input_size=args.input_size,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
