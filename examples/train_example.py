from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the new dataset-prep and training commands.")
    parser.add_argument("--dataset-dir", default="data/training")
    parser.add_argument("--run-name", default="lstm_demo")
    parser.add_argument("--model", default="lstm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("Prepare the canonical dataset first:")
    print(
        "  python prepare_dataset.py --output-dir data/training "
        "--raw-root data/raw/cropnet --crop-type corn --state-codes IA --years 2017 2018 2019 2020 2021 2022"
    )
    print("Then train from the prepared split files:")
    print(
        f"  python training/train.py --dataset-dir {Path(args.dataset_dir)} "
        f"--run-name {args.run_name} --models {args.model}"
    )


if __name__ == "__main__":
    main()
