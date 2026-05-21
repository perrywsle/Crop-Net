"""Command-line plant/crop image segmentation."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from crop_fusion_ai.models import PlantSegmentationModel


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for segmentation."""
    parser = argparse.ArgumentParser(description="Segment crop area from an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/segmentation"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run segmentation and print JSON output."""
    args = parse_args(argv)
    segmenter = PlantSegmentationModel(model_path=args.model, device=args.device)
    result = segmenter.segment(args.image, output_dir=args.output_dir)
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
