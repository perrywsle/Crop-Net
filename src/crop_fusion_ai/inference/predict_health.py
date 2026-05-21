"""Command-line plant health image prediction."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from crop_fusion_ai.models import PlantHealthClassifier


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for plant health prediction."""
    parser = argparse.ArgumentParser(description="Predict plant health from an image.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional path to image model metadata or future weights.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run plant health image prediction and print JSON output."""
    args = parse_args(argv)
    classifier = PlantHealthClassifier(model_path=args.model)
    prediction = classifier.predict(args.image)
    print(prediction.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
