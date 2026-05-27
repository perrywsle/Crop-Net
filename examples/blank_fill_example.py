from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.data import load_monthly_features
from cropnet_forecasting.predictor import BlankFillPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run blank-fill forecasting from partial-year monthly features.")
    parser.add_argument("--monthly-table", required=True)
    parser.add_argument("--checkpoint", default="weights/lstm_best.pt")
    parser.add_argument("--scaler", default="weights/scaler.csv")
    parser.add_argument("--config", default="configs/residual_lstm_all.yaml")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--known-months", type=int, default=1)
    parser.add_argument("--output", default="outputs/blank_fill_predictions.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    monthly = load_monthly_features(args.monthly_table)
    predictor = BlankFillPredictor.from_artifacts(args.checkpoint, args.scaler, args.config)
    predictions = predictor.predict_remaining_year(monthly, year=args.year, known_months=args.known_months)
    predictor.save_predictions(args.output)
    print(f"Saved {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
