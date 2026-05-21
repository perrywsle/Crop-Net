"""Plant health image classifier wrapper.

The current implementation intentionally uses deterministic placeholder
inference. It validates that an image can be opened and returns a safe
``ImagePrediction`` until a real PyTorch/EfficientNet model is trained.
"""

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from crop_fusion_ai.config.schemas import ImagePrediction

LOGGER = logging.getLogger(__name__)


class ImageModelMetadata(BaseModel):
    """Serializable metadata for the plant health image model."""

    model_type: str = "placeholder"
    classes: list[str] = Field(
        default_factory=lambda: ["healthy", "mild_disease", "severe_disease"]
    )
    input_size: tuple[int, int] = (224, 224)
    notes: str = "Deterministic placeholder model; replace with trained model later."


class PlantHealthClassifier:
    """Wrapper for plant health image prediction.

    The public interface is intentionally stable so a real Torch model can
    replace the placeholder implementation without changing the UI or fusion
    pipeline.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """Create the classifier and optionally load metadata from disk."""
        self.model_path: Path | None = None
        self.metadata = ImageModelMetadata()
        self.is_placeholder = True

        if model_path is not None:
            self.load(model_path)

    def predict(self, image_path: Path) -> ImagePrediction:
        """Predict plant health from an image path.

        Placeholder behavior:
        - Opens and verifies the image with Pillow.
        - Computes a deterministic brightness-based dummy severity.
        - Logs a warning so callers know no trained model is being used.
        """
        image = self._load_rgb_image(image_path)
        brightness = self._mean_brightness(image)

        LOGGER.warning(
            "Using placeholder plant health inference for %s; train/load a real "
            "model before using predictions for decisions.",
            image_path,
        )

        if brightness >= 0.66:
            return ImagePrediction(
                disease_class="healthy",
                health_score=0.85,
                confidence=0.50,
            )
        if brightness >= 0.33:
            return ImagePrediction(
                disease_class="mild_disease",
                health_score=0.60,
                confidence=0.45,
            )
        return ImagePrediction(
            disease_class="severe_disease",
            health_score=0.30,
            confidence=0.40,
        )

    def load(self, model_path: Path) -> None:
        """Load model metadata from a JSON file.

        Real model weight loading is intentionally deferred. For now, this
        method records the path and reads metadata if the path exists.
        """
        self.model_path = model_path
        if not model_path.exists():
            LOGGER.warning(
                "Image model file %s does not exist; continuing with placeholder "
                "inference.",
                model_path,
            )
            self.is_placeholder = True
            return

        with model_path.open("r", encoding="utf-8") as file:
            metadata_json: dict[str, Any] = json.load(file)
        self.metadata = ImageModelMetadata.model_validate(metadata_json)
        self.is_placeholder = self.metadata.model_type == "placeholder"

    def save(self, model_path: Path) -> None:
        """Save current image model metadata to disk."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w", encoding="utf-8") as file:
            file.write(self.metadata.model_dump_json(indent=2))
            file.write("\n")
        self.model_path = model_path

    def _load_rgb_image(self, image_path: Path) -> Image.Image:
        """Open an image and convert it to RGB."""
        if not image_path.exists():
            msg = f"Image file does not exist: {image_path}"
            raise FileNotFoundError(msg)

        try:
            with Image.open(image_path) as image:
                return image.convert("RGB")
        except UnidentifiedImageError as exc:
            msg = f"File is not a readable image: {image_path}"
            raise ValueError(msg) from exc

    @staticmethod
    def _mean_brightness(image: Image.Image) -> float:
        """Return mean RGB brightness normalized to the [0, 1] interval."""
        pixels = image.resize((1, 1)).getpixel((0, 0))
        if not isinstance(pixels, tuple):
            if pixels is None:
                return 0.0
            return float(pixels) / 255.0
        return sum(float(channel) for channel in pixels[:3]) / (3.0 * 255.0)
