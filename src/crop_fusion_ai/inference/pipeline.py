"""End-to-end late-fusion inference pipeline."""

from pathlib import Path

from crop_fusion_ai.config.schemas import (
    CropFeatures,
    ImagePrediction,
    WeatherFeatures,
    YieldInput,
    YieldPrediction,
)
from crop_fusion_ai.models import (
    PlantHealthClassifier,
    PlantSegmentationModel,
    SegmentationResult,
    YieldRegressor,
)

LOW_HEALTH_SCORE_THRESHOLD = 0.5
LOW_IMAGE_CONFIDENCE_THRESHOLD = 0.6


class CropFusionPipeline:
    """Connect plant health image inference with yield prediction."""

    def __init__(
        self,
        image_model_path: Path | None = None,
        yield_model_path: Path | None = None,
        segmentation_model_path: Path | None = None,
    ) -> None:
        """Create an end-to-end inference pipeline."""
        self.image_classifier = PlantHealthClassifier(model_path=image_model_path)
        self.image_segmenter = PlantSegmentationModel(
            model_path=segmentation_model_path
        )
        self.yield_regressor = YieldRegressor(model_path=yield_model_path)

    def predict_from_image_and_features(
        self,
        image_path: Path,
        weather: WeatherFeatures,
        crop: CropFeatures,
    ) -> tuple[ImagePrediction, SegmentationResult, YieldPrediction]:
        """Run image inference, build fusion features, and predict yield."""
        image_prediction = self.image_classifier.predict(image_path)
        segmentation_result = self.image_segmenter.segment(image_path)
        warnings = self._collect_pre_yield_warnings(image_prediction)

        yield_input = YieldInput(
            weather=weather,
            crop=crop,
            image_prediction=image_prediction,
        )
        try:
            yield_prediction = self.yield_regressor.predict(yield_input)
        except RuntimeError as exc:
            msg = (
                "Yield model is untrained or not loaded. Train it with "
                "`python -m crop_fusion_ai.training.train_yield_model` or pass "
                "a valid yield_model_path."
            )
            raise RuntimeError(msg) from exc

        return image_prediction, segmentation_result, yield_prediction.model_copy(
            update={"warnings": [*warnings, *yield_prediction.warnings]}
        )

    def _collect_pre_yield_warnings(
        self,
        image_prediction: ImagePrediction,
    ) -> list[str]:
        """Collect warnings available before yield prediction runs."""
        warnings: list[str] = []

        if self.image_classifier.is_placeholder:
            warnings.append(
                "Image classifier is using placeholder inference; health score is "
                "not from a trained vision model."
            )
        if self.yield_regressor.pipeline is None:
            warnings.append(
                "Yield regressor is untrained or not loaded; prediction cannot run."
            )
        if image_prediction.health_score < LOW_HEALTH_SCORE_THRESHOLD:
            warnings.append(
                "Image health score is low; crop stress may reduce predicted yield."
            )
        if image_prediction.confidence < LOW_IMAGE_CONFIDENCE_THRESHOLD:
            warnings.append(
                "Image prediction confidence is low; interpret fusion output with "
                "caution."
            )

        return warnings
