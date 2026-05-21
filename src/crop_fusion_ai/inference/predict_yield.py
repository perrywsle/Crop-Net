"""Command-line yield prediction for late-fusion CropNet-style features."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from crop_fusion_ai.config.schemas import (
    CropFeatures,
    ImagePrediction,
    WeatherFeatures,
    YieldInput,
)
from crop_fusion_ai.models.yield_regressor import YieldRegressor

DEFAULT_MODEL_PATH = Path("models/yield_model/yield_regressor.joblib")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line yield prediction arguments."""
    parser = argparse.ArgumentParser(description="Predict crop yield.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--crop-type", type=str, required=True)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--planting-age-days", type=int, default=None)
    parser.add_argument("--temperature-mean", type=float, required=True)
    parser.add_argument("--temperature-min", type=float, default=None)
    parser.add_argument("--temperature-max", type=float, default=None)
    parser.add_argument("--rainfall-total", type=float, required=True)
    parser.add_argument("--humidity-mean", type=float, default=None)
    parser.add_argument("--solar-radiation-mean", type=float, default=None)
    parser.add_argument("--disease-class", type=str, default="healthy")
    parser.add_argument("--health-score", type=float, default=0.8)
    parser.add_argument("--image-confidence", type=float, default=0.5)
    return parser.parse_args(argv)


def build_yield_input(args: argparse.Namespace) -> YieldInput:
    """Build a validated YieldInput from CLI arguments."""
    return YieldInput(
        weather=WeatherFeatures(
            temperature_mean=args.temperature_mean,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            rainfall_total=args.rainfall_total,
            humidity_mean=args.humidity_mean,
            solar_radiation_mean=args.solar_radiation_mean,
        ),
        crop=CropFeatures(
            crop_type=args.crop_type,
            region=args.region,
            year=args.year,
            planting_age_days=args.planting_age_days,
        ),
        image_prediction=ImagePrediction(
            disease_class=args.disease_class,
            health_score=args.health_score,
            confidence=args.image_confidence,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Load the saved yield regressor and print a JSON prediction."""
    args = parse_args(argv)
    regressor = YieldRegressor(model_path=args.model_path)
    prediction = regressor.predict(build_yield_input(args))
    print(prediction.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
