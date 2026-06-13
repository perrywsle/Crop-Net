from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass(slots=True)
class SweepJob:
    name: str
    models: list[str]
    base_args: dict[str, Any]
    grid: dict[str, list[Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _normalize_models(models: Any) -> list[str]:
    if models is None:
        return ["lstm"]
    if isinstance(models, str):
        return [models]
    return [str(model) for model in models]


def _normalize_grid(grid: Any) -> dict[str, list[Any]]:
    if not grid:
        return {}
    if not isinstance(grid, dict):
        raise ValueError("Sweep grid must be a JSON object mapping argument names to value lists.")
    normalized: dict[str, list[Any]] = {}
    for key, value in grid.items():
        values = _as_list(value)
        if not values:
            raise ValueError(f"Sweep grid key '{key}' has no values.")
        normalized[str(key)] = values
    return normalized


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[key] for key in keys)):
        combos.append(dict(zip(keys, values, strict=True)))
    return combos


def _slug_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p")
    text = str(value)
    text = text.replace("/", "-").replace(" ", "")
    text = text.replace(".", "p")
    return text


def _slug_combo(combo: dict[str, Any]) -> str:
    if not combo:
        return "base"
    parts = [f"{key.replace('_', '-')}-{_slug_value(value)}" for key, value in sorted(combo.items())]
    return "__".join(parts)


def _metric_column(objective: dict[str, Any]) -> str:
    if "column" in objective:
        return str(objective["column"])
    metric = str(objective.get("metric", "r2"))
    split = objective.get("split")
    if split:
        return f"{split}_{metric}"
    return metric


def _direction(objective: dict[str, Any]) -> str:
    return str(objective.get("direction", "max")).lower()


def _build_jobs(config: dict[str, Any]) -> list[SweepJob]:
    base_args = config.get("base_args", {})
    base_grid = _normalize_grid(config.get("grid", {}))
    base_models = _normalize_models(config.get("models"))

    jobs_cfg = config.get("jobs")
    jobs: list[SweepJob] = []
    if jobs_cfg is None:
        jobs.append(
            SweepJob(
                name=str(config.get("name", "sweep")),
                models=base_models,
                base_args=dict(base_args),
                grid=base_grid,
            )
        )
        return jobs

    if not isinstance(jobs_cfg, list):
        raise ValueError("'jobs' must be a list when provided.")

    for idx, job_cfg in enumerate(jobs_cfg, start=1):
        if not isinstance(job_cfg, dict):
            raise ValueError("Each job entry must be an object.")
        job_name = str(job_cfg.get("name", f"job_{idx}"))
        models = _normalize_models(job_cfg.get("models", base_models))
        merged_base = dict(base_args)
        merged_base.update(job_cfg.get("base_args", {}))
        merged_grid = dict(base_grid)
        merged_grid.update(_normalize_grid(job_cfg.get("grid", {})))
        jobs.append(SweepJob(name=job_name, models=models, base_args=merged_base, grid=merged_grid))
    return jobs


def _args_from_mapping(mapping: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in mapping.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, list):
            args.append(flag)
            args.extend(str(item) for item in value)
            continue
        if value is None:
            continue
        args.extend([flag, str(value)])
    return args


def _run_training_command(command: list[str], *, dry_run: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("\n> " + " ".join(command))
    if dry_run:
        return None
    return subprocess.run(command, check=True, text=True)


def _read_metrics(metrics_path: Path, objective_column: str, *, direction: str = "max") -> pd.Series | None:
    if not metrics_path.exists():
        return None
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        return None
    if objective_column not in frame.columns:
        return frame.iloc[0]
    frame = frame.copy()
    frame = frame[np.isfinite(frame[objective_column].to_numpy(dtype=float))] if objective_column in frame.columns else frame
    if frame.empty:
        return None
    ascending = direction == "min"
    return frame.sort_values(objective_column, ascending=ascending).iloc[0]


def _record_summary(
    *,
    job: SweepJob,
    run_name: str,
    combo: dict[str, Any],
    objective_column: str,
    objective_direction: str,
    run_dir: Path,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    best_row = _read_metrics(metrics_path, objective_column, direction=objective_direction)
    payload: dict[str, Any] = {
        "job": job.name,
        "run_name": run_name,
        "models": ",".join(job.models),
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "objective_column": objective_column,
        "objective_direction": objective_direction,
        "status": "ok" if best_row is not None else "missing_metrics",
        "combo": json.dumps(combo, default=_json_default),
    }
    if best_row is not None:
        for key, value in best_row.to_dict().items():
            payload[key] = value
        payload["score"] = float(best_row[objective_column]) if objective_column in best_row.index and pd.notna(best_row[objective_column]) else float("nan")
    else:
        payload["score"] = float("nan")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run JSON-driven hyperparameter sweeps for CropNet training.")
    parser.add_argument("--config", type=Path, required=True, help="Sweep JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap on the number of runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _load_json(args.config)
    jobs = _build_jobs(config)
    objective = config.get("objective", {})
    if not isinstance(objective, dict):
        raise ValueError("'objective' must be a JSON object when provided.")
    objective_column = _metric_column(objective)
    objective_direction = _direction(objective)
    if objective_direction not in {"max", "min"}:
        raise ValueError("Objective direction must be 'max' or 'min'.")

    train_script = Path(config.get("train_script", ROOT / "training" / "train.py"))
    output_root = Path(config.get("output_dir", ROOT / "training" / "sweeps"))
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_count = 0
    for job in jobs:
        combos = _expand_grid(job.grid)
        for combo_index, combo in enumerate(combos, start=1):
            if args.max_runs is not None and run_count >= args.max_runs:
                break
            merged_args = dict(job.base_args)
            merged_args.update(combo)
            run_name = merged_args.pop("run_name", None)
            if run_name is None:
                model_tag = "_".join(job.models)
                if len(job.models) == 1 and job.name == job.models[0]:
                    run_name = f"{job.name}_{combo_index:03d}_{_slug_combo(combo)}"
                else:
                    run_name = f"{job.name}_{model_tag}_{combo_index:03d}_{_slug_combo(combo)}"
            run_dir = output_root / str(run_name)
            if run_dir.exists() and any(run_dir.iterdir()) and not bool(merged_args.get("overwrite", False)):
                raise FileExistsError(f"Run directory already exists and is not empty: {run_dir}.")

            command = [
                sys.executable,
                str(train_script),
                "--run-name",
                str(run_name),
                "--output-dir",
                str(output_root),
            ]
            dataset_dir = config.get("dataset_dir", merged_args.pop("dataset_dir", None))
            if dataset_dir is not None:
                command.extend(["--dataset-dir", str(dataset_dir)])
            models = merged_args.pop("models", job.models)
            command.append("--models")
            command.extend(str(model) for model in models)
            command.extend(_args_from_mapping(merged_args))

            _run_training_command(command, dry_run=args.dry_run)
            run_count += 1
            if not args.dry_run:
                rows.append(
                    _record_summary(
                        job=job,
                        run_name=str(run_name),
                        combo=combo,
                        objective_column=objective_column,
                        objective_direction=objective_direction,
                        run_dir=run_dir,
                    )
                )

        if args.max_runs is not None and run_count >= args.max_runs:
            break

    summary_path = output_root / "sweep_results.csv"
    if rows and not args.dry_run:
        frame = pd.DataFrame(rows)
        if objective_column in frame.columns:
            ascending = objective_direction == "min"
            frame = frame.sort_values(["score", "run_name"], ascending=[ascending, True], na_position="last")
        frame.to_csv(summary_path, index=False)
        best = frame.iloc[0].to_dict() if not frame.empty else {}
        (output_root / "best_sweep.json").write_text(json.dumps(best, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(f"\nSweep complete. Summary written to {summary_path}")
    elif args.dry_run:
        print("\nDry run complete. No training jobs were executed.")
    else:
        print("\nSweep complete but no summary was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
