from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropnet_forecasting.yield_training import (
    BASELINE_MODELS,
    DEFAULT_YIELD_DATASET_DIR,
    DEFAULT_YIELD_RUNS_DIR,
    load_prepared_yield_dataset,
    train_yield_models,
)

YIELD_MODEL_FAMILIES = ("Ridge", "RandomForest", "ExtraTrees")
OPTIONAL_YIELD_MODELS = ("XGBoost", "LightGBM")


def _resolve_models(requested: list[str], *, include_optional_models: bool) -> list[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        models = list(YIELD_MODEL_FAMILIES + BASELINE_MODELS)
        if include_optional_models:
            models.extend(OPTIONAL_YIELD_MODELS)
        return models
    valid = set(YIELD_MODEL_FAMILIES) | set(BASELINE_MODELS) | set(OPTIONAL_YIELD_MODELS)
    resolved: list[str] = []
    for model in requested:
        if model not in valid:
            raise ValueError(f"Unsupported model: {model}. Valid models: {', '.join(sorted(valid))}")
        if model not in resolved:
            resolved.append(model)
    if include_optional_models:
        for model in OPTIONAL_YIELD_MODELS:
            if model not in resolved:
                resolved.append(model)
    return resolved


def _make_run_dir(output_dir: Path, run_name: str | None, models: list[str]) -> Path:
    if run_name is None:
        run_name = f"{'_'.join(m.lower() for m in models)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir / run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train official CropNet yield regressors from a prepared dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_YIELD_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_YIELD_RUNS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--tune-hyperparameters", action="store_true", default=True)
    parser.add_argument("--no-tune-hyperparameters", action="store_false", dest="tune_hyperparameters")
    parser.add_argument("--include-optional-models", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = _resolve_models(args.models, include_optional_models=args.include_optional_models)
    run_dir = _make_run_dir(args.output_dir, args.run_name, models)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Re-run with --overwrite or choose a new --run-name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared = load_prepared_yield_dataset(args.dataset_dir)
    results, metadata = train_yield_models(
        prepared,
        run_dir=run_dir,
        models=models,
        random_state=args.random_state,
        tune_hyperparameters=args.tune_hyperparameters,
        include_optional_models=args.include_optional_models,
    )
    print("\nRun summary:")
    print(results[["model", "model_type", "val_rmse", "val_r2", "test_rmse", "test_r2"]].to_string(index=False))
    print(f"Best trainable model: {metadata['best_trainable_model']}")
    print(f"Run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
