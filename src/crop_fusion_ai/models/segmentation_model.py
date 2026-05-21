"""Lightweight plant/crop segmentation model for the desktop PoC."""

import hashlib
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, Field

DEFAULT_SEGMENTATION_MODEL_PATH = Path("models/image_model/segmentation_model.pt")
DEFAULT_SEGMENTATION_OUTPUT_DIR = Path("reports/segmentation")


class SegmentationDependencyError(ImportError):
    """Raised when optional PyTorch dependencies are unavailable."""


class SegmentationResult(BaseModel):
    """Serializable segmentation output for UI and report saving."""

    mask_path: Path
    overlay_path: Path
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model_path: Path | None = None
    used_trained_model: bool


class PlantSegmentationModel:
    """Tiny CNN segmentation wrapper with a threshold fallback for demos."""

    def __init__(
        self,
        model_path: Path | None = DEFAULT_SEGMENTATION_MODEL_PATH,
        *,
        device: str = "cpu",
    ) -> None:
        """Create the segmenter and load weights when available."""
        self.model_path = model_path
        self.device = device
        self.input_size = (64, 64)
        self.model: Any | None = None
        self.used_trained_model = False
        if model_path is not None and model_path.exists():
            self._load_torch_model(model_path)

    def segment(
        self,
        image_path: Path,
        *,
        output_dir: Path = DEFAULT_SEGMENTATION_OUTPUT_DIR,
    ) -> SegmentationResult:
        """Segment a crop image, save mask/overlay, and return output paths."""
        if not image_path.exists():
            msg = f"Image file does not exist: {image_path}"
            raise FileNotFoundError(msg)

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        if self.model is None:
            mask = create_pseudo_segmentation_mask(rgb_image)
            confidence = 0.5
        else:
            mask = self._predict_mask(rgb_image)
            confidence = 0.85

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = _stable_output_stem(image_path)
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"
        _save_mask(mask, mask_path)
        _save_overlay(rgb_image, mask, overlay_path)
        coverage_ratio = float(mask.mean())
        return SegmentationResult(
            mask_path=mask_path,
            overlay_path=overlay_path,
            coverage_ratio=coverage_ratio,
            confidence=confidence,
            model_path=self.model_path if self.used_trained_model else None,
            used_trained_model=self.used_trained_model,
        )

    def _load_torch_model(self, model_path: Path) -> None:
        """Load the tiny CNN checkpoint."""
        torch, nn = _import_torch_stack()
        checkpoint: dict[str, Any] = torch.load(
            model_path,
            map_location=torch.device(self.device),
            weights_only=False,
        )
        input_size_value = checkpoint.get("input_size", self.input_size)
        self.input_size = (int(input_size_value[0]), int(input_size_value[1]))
        model = build_tiny_segmentation_network(nn)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.model = model.to(self.device)
        self.used_trained_model = True

    def _predict_mask(self, image: Image.Image) -> NDArray[np.float32]:
        """Run the trained tiny CNN and return a binary mask at image size."""
        if self.model is None:
            msg = "Segmentation model is not loaded"
            raise RuntimeError(msg)
        torch, _ = _import_torch_stack()
        original_size = image.size
        tensor = image_to_tensor(image, self.input_size).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        probability_image = Image.fromarray((probabilities * 255.0).astype(np.uint8))
        resized = probability_image.resize(original_size, Image.Resampling.BILINEAR)
        return cast(
            NDArray[np.float32],
            (np.asarray(resized, dtype=np.float32) / 255.0 >= 0.5).astype(np.float32),
        )


def build_tiny_segmentation_network(nn: Any) -> Any:  # noqa: ANN401
    """Build a tiny low-memory segmentation CNN."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 1, kernel_size=1),
    )


def image_to_tensor(  # noqa: ANN401
    image: Image.Image,
    input_size: tuple[int, int],
) -> Any:  # noqa: ANN401
    """Convert a PIL image to a normalized CHW torch tensor."""
    torch, _ = _import_torch_stack()
    resized = image.resize(input_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def create_pseudo_segmentation_mask(image: Image.Image) -> NDArray[np.float32]:
    """Create a deterministic vegetation-like pseudo mask for PoC training."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    brightness = array.mean(axis=2)
    vegetation_signal = (green >= red * 0.9) & (green >= blue * 0.9)
    bright_signal = brightness >= float(np.quantile(brightness, 0.55))
    mask = vegetation_signal | bright_signal
    return cast(NDArray[np.float32], mask.astype(np.float32))


def pseudo_mask_tensor(  # noqa: ANN401
    image: Image.Image,
    input_size: tuple[int, int],
) -> Any:  # noqa: ANN401
    """Create a torch tensor pseudo mask for a resized image."""
    torch, _ = _import_torch_stack()
    resized = image.resize(input_size, Image.Resampling.BILINEAR)
    mask = create_pseudo_segmentation_mask(resized)
    return torch.from_numpy(mask).unsqueeze(0).float()


def _save_mask(mask: NDArray[np.float32], output_path: Path) -> None:
    """Save a binary mask as an 8-bit image."""
    Image.fromarray((mask * 255.0).astype(np.uint8)).save(output_path)


def _save_overlay(
    image: Image.Image,
    mask: NDArray[np.float32],
    output_path: Path,
) -> None:
    """Save a red transparent overlay for UI display."""
    overlay_base = image.convert("RGBA")
    mask_image = Image.fromarray((mask * 160.0).astype(np.uint8)).resize(
        overlay_base.size,
        Image.Resampling.NEAREST,
    )
    red_overlay = Image.new("RGBA", overlay_base.size, (255, 48, 48, 0))
    red_overlay.putalpha(mask_image)
    Image.alpha_composite(overlay_base, red_overlay).save(output_path)


def _stable_output_stem(image_path: Path) -> str:
    """Create a stable short output stem from an image path."""
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}"


def _import_torch_stack() -> tuple[Any, Any]:
    """Import torch lazily so non-vision paths still work without it."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        msg = (
            "PlantSegmentationModel requires optional dependency 'torch'. "
            "Install the project vision extra before training or inference."
        )
        raise SegmentationDependencyError(msg) from exc
    return torch, nn
