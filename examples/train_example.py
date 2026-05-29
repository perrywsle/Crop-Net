from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.trainer import CropNetTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the validated research training flow through a clean wrapper.")
    parser.add_argument("--config", default="configs/residual_lstm_all.yaml")
    parser.add_argument("--mode", choices=["fit", "evaluate", "blank_fill"], default="fit")
    return parser


def main() -> None:
    # The config controls model_name, feature_group, target_mode, and where the legacy training wrapper writes outputs.
    args = build_parser().parse_args()
    trainer = CropNetTrainer.from_config(args.config)
    if args.mode == "fit":
        print("CropNetTrainer.fit() would launch the validated research script with the YAML config.")
    elif args.mode == "evaluate":
        print("CropNetTrainer.evaluate() wraps the existing resume/eval workflow.")
    else:
        print("CropNetTrainer.evaluate_blank_fill() wraps strict blank-fill evaluation.")
    trainer.save_artifacts(ROOT / "reports" / "metrics")


if __name__ == "__main__":
    main()
