from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.predictor import BlankFillPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a saved CropNet checkpoint and print a one-step forecast.")
    parser.add_argument("--checkpoint", default="weights/lstm_best.pt")
    parser.add_argument("--scaler", default="weights/scaler.csv")
    parser.add_argument("--config", default="configs/residual_lstm_all.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = BlankFillPredictor.from_artifacts(args.checkpoint, args.scaler, args.config)
    print("Loaded predictor")
    print(f"  model_name={predictor.model_name}")
    print(f"  feature_group_size={len(predictor.feature_names)}")
    print(f"  target_mode={predictor.target_mode}")
    print("Next step: pass a real seq_len x feature_count window into predictor.predict_next(...).")


if __name__ == "__main__":
    main()
