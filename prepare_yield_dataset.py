from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.yield_training import (
    DEFAULT_SOURCE_DATASET_DIR,
    DEFAULT_YIELD_DATASET_DIR,
    prepare_yield_dataset,
)


def _parse_years(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    return [int(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the official CropNet monthly yield dataset.")
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        default=DEFAULT_SOURCE_DATASET_DIR,
        help="Prepared monthly feature dataset directory produced by prepare_dataset.py.",
    )
    parser.add_argument(
        "--monthly-path",
        type=Path,
        default=None,
        help="Legacy fallback: direct monthly feature table path. Prefer --source-dataset-dir.",
    )
    parser.add_argument("--usda-path", type=Path, nargs="+", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_YIELD_DATASET_DIR)
    parser.add_argument("--crop-type", type=str, default="corn")
    parser.add_argument(
        "--feature-group",
        type=str,
        default="all",
        choices=sorted({"all", "ag", "ndvi", "weather", "ag_ndvi", "ag_weather", "ndvi_weather"}),
    )
    parser.add_argument("--train-years", nargs="+", default=None)
    parser.add_argument("--val-years", nargs="+", default=None)
    parser.add_argument("--test-years", nargs="+", default=None)
    parser.add_argument("--min-overlap-rows", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def flatten_usda_path_groups(usda_path_groups: list[list[Path]] | None) -> list[Path] | None:
    if not usda_path_groups:
        return None
    return [path for group in usda_path_groups for path in group]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {args.output_dir}. "
            "Re-run with --overwrite or choose a new --output-dir."
        )
    prepared = prepare_yield_dataset(
        source_dataset_dir=args.source_dataset_dir,
        monthly_path=args.monthly_path,
        usda_paths=flatten_usda_path_groups(args.usda_path),
        output_dir=args.output_dir,
        crop_type=args.crop_type,
        feature_group=args.feature_group,
        train_years=_parse_years(args.train_years),
        val_years=_parse_years(args.val_years),
        test_years=_parse_years(args.test_years),
        min_overlap_rows=args.min_overlap_rows,
    )
    print(f"Prepared yield dataset at {prepared.dataset_dir}")
    print(f"  train: {prepared.dataset_dir / 'train.parquet'}")
    print(f"  val:   {prepared.dataset_dir / 'val.parquet'}")
    print(f"  test:  {prepared.dataset_dir / 'test.parquet'}")
    print(f"  meta:  {prepared.dataset_dir / 'metadata.json'}")
    print(f"  source: {args.source_dataset_dir or args.monthly_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
