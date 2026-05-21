"""CLI for MobileNet crop-condition feature extraction smoke tests."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from crop_fusion_ai.models.mobilenet_feature_extractor import MobileNetFeatureExtractor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse MobileNet feature extraction arguments."""
    parser = argparse.ArgumentParser(
        description="Extract MobileNet features from one crop-condition image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--feature-preview-count", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run MobileNet feature extraction and print a compact JSON summary."""
    args = parse_args(argv)
    extractor = MobileNetFeatureExtractor(use_pretrained_weights=args.pretrained)
    features = extractor.extract(args.image)
    preview_count = max(0, args.feature_preview_count)
    payload = {
        "image": str(args.image),
        "feature_count": len(features),
        "feature_preview": features[:preview_count],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
