"""Optional MobileNet feature extractor for crop-condition imagery."""

from pathlib import Path
from typing import Any

from PIL import Image


class MobileNetDependencyError(ImportError):
    """Raised when optional PyTorch/Torchvision dependencies are missing."""


class MobileNetFeatureExtractor:
    """Extract image embeddings with Torchvision MobileNet.

    Heavy ML dependencies are imported lazily so the rest of the project remains
    usable without PyTorch installed.
    """

    def __init__(
        self,
        *,
        use_pretrained_weights: bool = False,
        device: str = "cpu",
    ) -> None:
        """Create a MobileNet feature extractor."""
        torch, models, transforms = self._import_torch_stack()
        self._torch = torch
        self.device = device

        weights = None
        if use_pretrained_weights:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT

        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract(self, image_path: Path) -> list[float]:
        """Extract a MobileNet feature vector from one image."""
        if not image_path.exists():
            msg = f"Image file does not exist: {image_path}"
            raise FileNotFoundError(msg)

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            features = self.model(tensor).squeeze(0).detach().cpu().tolist()
        return [float(value) for value in features]

    @staticmethod
    def _import_torch_stack() -> tuple[Any, Any, Any]:
        """Import torch and torchvision or raise an actionable setup error."""
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            msg = (
                "MobileNetFeatureExtractor requires optional dependencies "
                "'torch' and 'torchvision'. Install the project vision extra or "
                "install compatible PyTorch wheels for your machine."
            )
            raise MobileNetDependencyError(msg) from exc
        return torch, models, transforms
