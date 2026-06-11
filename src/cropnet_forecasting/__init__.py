"""Developer-friendly CropNet forecasting package."""

from .config import ForecastingConfig
from .features import FEATURE_COLS, FEATURE_GROUP_SELECTIONS, selected_feature_columns


def __getattr__(name: str):  # noqa: ANN202
    """Load Torch-backed helpers only when callers explicitly request them."""
    if name == "BlankFillPredictor":
        from .predictor import BlankFillPredictor

        return BlankFillPredictor
    if name == "CropNetModelFactory":
        from .models import CropNetModelFactory

        return CropNetModelFactory
    if name == "CropNetTrainer":
        from .trainer import CropNetTrainer

        return CropNetTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BlankFillPredictor",
    "CropNetModelFactory",
    "CropNetTrainer",
    "FEATURE_COLS",
    "FEATURE_GROUP_SELECTIONS",
    "ForecastingConfig",
    "selected_feature_columns",
]
