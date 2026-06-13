from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.training_dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CROP_TYPE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_REPO_ID,
    DEFAULT_STATE_CODES,
    DEFAULT_YEARS,
    prepare_training_dataset_from_download,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a canonical CropNet training dataset.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT, help="Directory where downloaded CropNet raw data will be stored.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where train/val/test files will be written.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory used for Hugging Face download cache.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--crop-type", default=DEFAULT_CROP_TYPE, help="Crop filter applied to the downloaded CropNet data.")
    parser.add_argument("--state-codes", nargs="+", default=list(DEFAULT_STATE_CODES), help="State abbreviations or FIPS codes to download.")
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS), help="Years to download and split.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker threads to use for feature extraction.")
    parser.add_argument("--train-years", nargs="+", type=int, default=None)
    parser.add_argument("--val-years", nargs="+", type=int, default=None)
    parser.add_argument("--test-years", nargs="+", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting a non-empty output directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prepared = prepare_training_dataset_from_download(
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        crop_type=args.crop_type,
        state_codes=args.state_codes,
        years=args.years,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(f"Wrote prepared training dataset to {prepared.output_dir}")
    print(f"  train: {prepared.train_path}")
    print(f"  val:   {prepared.val_path}")
    print(f"  test:  {prepared.test_path}")
    print(f"  meta:  {prepared.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
