from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_NAME = "corn_ia_monthly_2017_2022"
BATCH_PREFIX = "corn_ia_monthly_2017_2022_batch_"
EXPECTED_BATCHES = tuple(range(1, 11))
EXPECTED_YEARS = tuple(range(2017, 2023))
EXPECTED_MONTHS = tuple(range(1, 13))
ERROR_PATTERNS = ("ERROR", "Traceback", "Exception", "failed", "exit=1")
FORBIDDEN_FORECAST_COLUMNS = {"forecast_step", "known_months", "source_note", "y_pred"}
REFERENCE_BASELINE = {
    "model": "BaselinePreviousYearSameCounty",
    "rmse": 24.02,
    "mae": 23.23,
    "r2": 0.231,
    "mape": 13.56,
}


@dataclass(frozen=True)
class BatchPaths:
    run_name: str
    output_dir: Path
    artifact_path: Path
    log_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_dir(repo_root: Path) -> Path:
    return repo_root / "outputs" / "experiments" / PROJECT_NAME


def default_manifest_path(repo_root: Path) -> Path:
    return project_dir(repo_root) / "batch_manifest.csv"


def status_path(repo_root: Path) -> Path:
    return project_dir(repo_root) / "colab_batch_status.json"


def runner_log_path(repo_root: Path) -> Path:
    return project_dir(repo_root) / "colab_batch_runner.log"


def merged_monthly_path(repo_root: Path) -> Path:
    return project_dir(repo_root) / "artifacts" / "official_monthly_feature_table.parquet"


def merge_diagnostic_path(repo_root: Path) -> Path:
    return project_dir(repo_root) / "artifacts" / "monthly_merge_diagnostic.json"


def retrain_output_dir(repo_root: Path) -> Path:
    return repo_root / "outputs" / "yield_baseline" / "corn_ia_2017_2022_monthly_full"


def batch_paths(repo_root: Path, run_name: str) -> BatchPaths:
    output_dir = repo_root / "outputs" / "experiments" / run_name
    return BatchPaths(
        run_name=run_name,
        output_dir=output_dir,
        artifact_path=output_dir / "artifacts" / "official_monthly_feature_table.parquet",
        log_path=output_dir / "logs" / f"{run_name}.log",
    )


def usda_paths(repo_root: Path) -> list[Path]:
    base = repo_root / "data" / "usda_labels" / "USDA Crop Dataset" / "Corn"
    return [base / str(year) / f"USDA_Corn_County_{year}.csv" for year in EXPECTED_YEARS]


def append_runner_log(repo_root: Path, message: str) -> None:
    path = runner_log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Batch manifest not found: {path}")
    manifest = pd.read_csv(path, dtype={"batch_id": int, "fips_codes": str, "years": str, "quarters": str, "run_name": str})
    required = {"batch_id", "fips_codes", "years", "quarters", "run_name"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    return manifest.sort_values("batch_id").reset_index(drop=True)


def parse_space_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part for part in str(value).split() if part]


def normalize_county_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(5)


def scan_log(path: Path, max_lines: int = 4000) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "matches": []}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-max_lines:]
    regex = re.compile("|".join(re.escape(pattern) for pattern in ERROR_PATTERNS), re.IGNORECASE)
    matches = [line for line in tail if regex.search(line)]
    return {"path": str(path), "exists": True, "matches": matches[-50:], "match_count": len(matches)}


def validate_monthly_table(
    artifact_path: Path,
    *,
    expected_fips: list[str] | None = None,
    expected_years: tuple[int, ...] = EXPECTED_YEARS,
    expected_months: tuple[int, ...] = EXPECTED_MONTHS,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(artifact_path),
        "exists": artifact_path.exists(),
        "valid": False,
        "errors": [],
        "warnings": [],
    }
    if not artifact_path.exists():
        result["errors"].append("artifact_missing")
        return result

    try:
        frame = pd.read_parquet(artifact_path)
    except Exception as exc:
        result["errors"].append(f"read_failed: {exc}")
        return result

    result["rows"] = int(len(frame))
    result["columns"] = int(len(frame.columns))
    if frame.empty:
        result["errors"].append("artifact_empty")
        return result

    required = {"county_id", "crop_type", "year", "month"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        result["errors"].append(f"missing_required_columns: {missing}")
        return result

    forbidden = sorted(FORBIDDEN_FORECAST_COLUMNS.intersection(frame.columns))
    if forbidden:
        result["errors"].append(f"forbidden_forecast_columns_present: {forbidden}")

    county_id = normalize_county_id(frame["county_id"])
    years = sorted(pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int).unique().tolist())
    months = sorted(pd.to_numeric(frame["month"], errors="coerce").dropna().astype(int).unique().tolist())
    counties = sorted(county_id.dropna().unique().tolist())
    result["years"] = years
    result["months"] = months
    result["county_count"] = int(len(counties))

    if years != list(expected_years):
        result["errors"].append(f"unexpected_years: {years}")
    if months != list(expected_months):
        result["errors"].append(f"unexpected_months: {months}")

    key_frame = pd.DataFrame(
        {
            "county_id": county_id,
            "crop_type": frame["crop_type"].astype(str).str.lower(),
            "year": pd.to_numeric(frame["year"], errors="coerce").astype("Int64"),
            "month": pd.to_numeric(frame["month"], errors="coerce").astype("Int64"),
        }
    )
    duplicate_count = int(key_frame.duplicated(["county_id", "crop_type", "year", "month"]).sum())
    result["duplicate_key_count"] = duplicate_count
    if duplicate_count:
        result["errors"].append(f"duplicate_key_rows: {duplicate_count}")

    numeric = frame.select_dtypes(include=[np.number])
    numeric_float = numeric.apply(pd.to_numeric, errors="coerce").astype(float) if not numeric.empty else numeric
    nan_count = int(numeric_float.isna().sum().sum()) if not numeric_float.empty else 0
    inf_count = int(np.isinf(numeric_float.to_numpy()).sum()) if not numeric_float.empty else 0
    result["numeric_nan_count"] = nan_count
    result["numeric_inf_count"] = inf_count
    if nan_count:
        result["warnings"].append(f"numeric_nan_count: {nan_count}")
    if inf_count:
        result["errors"].append(f"numeric_inf_count: {inf_count}")

    feature_cols = [
        col
        for col in numeric.columns
        if col not in {"year", "month"} and "yield" not in col.lower() and "target" not in col.lower()
    ]
    result["feature_column_count"] = int(len(feature_cols))
    if not feature_cols:
        result["errors"].append("no_numeric_feature_columns")
    elif int(numeric_float[feature_cols].notna().sum().sum()) == 0:
        result["errors"].append("numeric_features_all_missing")

    if expected_fips is not None:
        clean_expected = sorted(str(value).zfill(5) for value in expected_fips)
        missing_fips = sorted(set(clean_expected).difference(counties))
        extra_fips = sorted(set(counties).difference(clean_expected))
        expected_rows = len(clean_expected) * len(expected_years) * len(expected_months)
        result["expected_rows"] = int(expected_rows)
        result["expected_county_count"] = int(len(clean_expected))
        if len(frame) != expected_rows:
            result["errors"].append(f"unexpected_row_count: expected={expected_rows} actual={len(frame)}")
        if missing_fips:
            result["errors"].append(f"missing_fips: {missing_fips}")
        if extra_fips:
            result["warnings"].append(f"extra_fips: {extra_fips}")

    result["valid"] = not result["errors"]
    return result


def row_for_batch(manifest: pd.DataFrame, batch_id: int) -> pd.Series:
    match = manifest.loc[manifest["batch_id"].astype(int) == int(batch_id)]
    if match.empty:
        raise ValueError(f"Batch {batch_id} not found in manifest.")
    return match.iloc[0]


def build_extraction_command(repo_root: Path, row: pd.Series, hf_cache_dir: Path) -> tuple[list[str], Path]:
    run_name = str(row["run_name"])
    paths = batch_paths(repo_root, run_name)
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/research/cropnet_feature_forecasting_v12_server.py",
        "--full-run",
        "--extract-only",
        "--state-codes",
        "IA",
        "--crop",
        "Corn",
        "--years",
        *parse_space_list(row["years"]),
        "--quarters",
        *parse_space_list(row["quarters"]),
        "--fips-codes",
        *parse_space_list(row["fips_codes"]),
        "--run-name",
        run_name,
        "--experiment-root",
        "outputs/experiments",
        "--base-dir",
        ".",
        "--repo-dir",
        "Crop-Net",
        "--cache-dir",
        str(hf_cache_dir),
        "--log-file",
        str(paths.log_path),
        "--resume",
        "--delete-raw-after-extract",
    ]
    return cmd, paths.log_path


def run_logged_command(repo_root: Path, cmd: list[str], *, env: dict[str, str] | None = None) -> int:
    append_runner_log(repo_root, "RUN " + " ".join(cmd))
    runner_log = runner_log_path(repo_root)
    with runner_log.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
        return_code = process.wait()
    append_runner_log(repo_root, f"EXIT code={return_code}")
    return return_code


def build_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(repo_root / "src"), str(repo_root / "Crop-Net" / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def batch_validation(repo_root: Path, row: pd.Series) -> dict[str, Any]:
    paths = batch_paths(repo_root, str(row["run_name"]))
    validation = validate_monthly_table(paths.artifact_path, expected_fips=parse_space_list(row["fips_codes"]))
    validation["batch_id"] = int(row["batch_id"])
    validation["run_name"] = str(row["run_name"])
    validation["log"] = scan_log(paths.log_path)
    return validation


def build_status(repo_root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    batches: dict[str, Any] = {}
    completed: list[int] = []
    invalid: list[int] = []
    missing: list[int] = []
    for _, row in manifest.iterrows():
        batch_id = int(row["batch_id"])
        validation = batch_validation(repo_root, row)
        batches[str(batch_id)] = validation
        if validation["valid"]:
            completed.append(batch_id)
        elif validation["exists"]:
            invalid.append(batch_id)
        else:
            missing.append(batch_id)
    status = {
        "updated_at": utc_now(),
        "project": PROJECT_NAME,
        "completed_count": len(completed),
        "completed_batches": completed,
        "missing_batches": missing,
        "invalid_batches": invalid,
        "batches": batches,
        "runner_log": str(runner_log_path(repo_root)),
        "status_path": str(status_path(repo_root)),
    }
    save_json(status_path(repo_root), status)
    return status


def run_batch(repo_root: Path, manifest: pd.DataFrame, batch_id: int, hf_cache_dir: Path) -> dict[str, Any]:
    row = row_for_batch(manifest, batch_id)
    before = batch_validation(repo_root, row)
    if before["valid"]:
        append_runner_log(repo_root, f"SKIP batch={batch_id} artifact already valid")
        return {"batch_id": batch_id, "status": "skipped_valid", "validation": before}

    cmd, log_path = build_extraction_command(repo_root, row, hf_cache_dir)
    append_runner_log(repo_root, f"START batch={batch_id} run_name={row['run_name']} log={log_path}")
    return_code = run_logged_command(repo_root, cmd, env=build_env(repo_root))
    after = batch_validation(repo_root, row)
    status = "completed" if return_code == 0 and after["valid"] else "failed"
    result = {
        "batch_id": batch_id,
        "status": status,
        "return_code": int(return_code),
        "validation": after,
        "finished_at": utc_now(),
    }
    append_runner_log(repo_root, f"FINISH batch={batch_id} status={status}")
    return result


def run_missing(
    repo_root: Path,
    manifest: pd.DataFrame,
    *,
    hf_cache_dir: Path,
    parallelism: int = 1,
    max_runs: int | None = None,
) -> dict[str, Any]:
    current = build_status(repo_root, manifest)
    to_run = current["missing_batches"] + current["invalid_batches"]
    if max_runs is not None:
        to_run = to_run[:max_runs]
    results: list[dict[str, Any]] = []

    if parallelism <= 1:
        for batch_id in to_run:
            results.append(run_batch(repo_root, manifest, int(batch_id), hf_cache_dir))
            build_status(repo_root, manifest)
    else:
        if parallelism > 2:
            raise ValueError("Parallelism above 2 is intentionally disabled for Colab stability.")
        append_runner_log(repo_root, f"EXPERIMENTAL parallel run parallelism={parallelism} batches={to_run}")
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {
                executor.submit(run_batch, repo_root, manifest, int(batch_id), hf_cache_dir): int(batch_id)
                for batch_id in to_run
            }
            for future in as_completed(futures):
                results.append(future.result())
                build_status(repo_root, manifest)

    final_status = build_status(repo_root, manifest)
    return {"started_batches": to_run, "results": results, "status": final_status}


def run_merge(repo_root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    current = build_status(repo_root, manifest)
    if current["completed_count"] != len(manifest):
        raise RuntimeError(f"Merge requires {len(manifest)} valid batches; status={current}")
    monthly_paths = [
        str(batch_paths(repo_root, str(row["run_name"])).artifact_path)
        for _, row in manifest.iterrows()
    ]
    cmd = [
        sys.executable,
        "src/cropnet_forecasting/yield_batching.py",
        "merge",
        "--monthly-path",
        *monthly_paths,
        "--output-path",
        str(merged_monthly_path(repo_root)),
        "--diagnostic-path",
        str(merge_diagnostic_path(repo_root)),
    ]
    return_code = run_logged_command(repo_root, cmd, env=build_env(repo_root))
    validation = validate_monthly_table(merged_monthly_path(repo_root), expected_fips=None)
    rows = validation.get("rows", 0)
    if rows and not 7128 <= int(rows) <= 7200:
        validation.setdefault("warnings", []).append(f"merged_row_count_outside_expected_range: {rows}")
    if return_code != 0 or not validation["valid"]:
        raise RuntimeError(f"Merge failed or produced invalid table: return_code={return_code} validation={validation}")
    return {"return_code": return_code, "merged_path": str(merged_monthly_path(repo_root)), "validation": validation}


def run_retrain(repo_root: Path) -> dict[str, Any]:
    missing_usda = [str(path) for path in usda_paths(repo_root) if not path.exists()]
    if missing_usda:
        raise FileNotFoundError(f"Missing USDA label CSVs: {missing_usda}")
    if not merged_monthly_path(repo_root).exists():
        raise FileNotFoundError(f"Merged monthly table not found: {merged_monthly_path(repo_root)}")
    output_dir = retrain_output_dir(repo_root)
    cmd = [
        sys.executable,
        "src/cropnet_forecasting/yield_regression.py",
        "--monthly-path",
        str(merged_monthly_path(repo_root)),
        "--usda-path",
        *[str(path) for path in usda_paths(repo_root)],
        "--crop-type",
        "corn",
        "--feature-group",
        "all",
        "--target-grain",
        "monthly",
        "--output-dir",
        str(output_dir),
    ]
    return_code = run_logged_command(repo_root, cmd, env=build_env(repo_root))
    metrics_path = output_dir / "yield_model_benchmark.csv"
    metrics: list[dict[str, Any]] = []
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).to_dict(orient="records")
    if return_code != 0:
        raise RuntimeError(f"Retrain failed with return code {return_code}")
    return {
        "return_code": return_code,
        "output_dir": str(output_dir),
        "model_path": str(output_dir / "best_yield_model.joblib"),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "reference_baseline": REFERENCE_BASELINE,
    }


def final_report(repo_root: Path, manifest: pd.DataFrame) -> dict[str, Any]:
    status = build_status(repo_root, manifest)
    merged = validate_monthly_table(merged_monthly_path(repo_root), expected_fips=None)
    output_dir = retrain_output_dir(repo_root)
    model_path = output_dir / "best_yield_model.joblib"
    metrics_path = output_dir / "yield_model_benchmark.csv"
    metrics = pd.read_csv(metrics_path).to_dict(orient="records") if metrics_path.exists() else []
    ready = status["completed_count"] == len(manifest) and merged["valid"] and model_path.exists() and bool(metrics)
    return {
        "completed_batch_count": status["completed_count"],
        "completed_batches": status["completed_batches"],
        "missing_batches": status["missing_batches"],
        "invalid_batches": status["invalid_batches"],
        "merged_table_path": str(merged_monthly_path(repo_root)),
        "merged_table_valid": bool(merged["valid"]),
        "merged_table_validation": merged,
        "retrain_output_path": str(output_dir),
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "reference_baseline": REFERENCE_BASELINE,
        "ready_for_yield_prediction": ready,
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab runner for Iowa Corn monthly 10-batch CropNet extraction.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Write and print batch status.")

    validate = subparsers.add_parser("validate-batch", help="Validate one batch artifact.")
    validate.add_argument("--batch-id", type=int, required=True)

    run_one = subparsers.add_parser("run-one", help="Run one batch if its artifact is missing or invalid.")
    run_one.add_argument("--batch-id", type=int, required=True)

    run_all = subparsers.add_parser("run-missing", help="Run all missing or invalid batches.")
    run_all.add_argument("--parallelism", type=int, default=1)
    run_all.add_argument("--max-runs", type=int, default=None)

    subparsers.add_parser("merge", help="Merge all valid batch artifacts.")
    subparsers.add_parser("retrain", help="Retrain monthly yield regression from merged table.")
    subparsers.add_parser("final-report", help="Print final extraction/retrain readiness report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = (args.manifest_path or default_manifest_path(repo_root)).resolve()
    hf_cache_dir = (args.hf_cache_dir or repo_root.parent / "hf_cache").resolve()

    try:
        manifest = read_manifest(manifest_path)
        if args.command == "status":
            print_json(build_status(repo_root, manifest))
            return 0
        if args.command == "validate-batch":
            print_json(batch_validation(repo_root, row_for_batch(manifest, args.batch_id)))
            return 0
        if args.command == "run-one":
            result = run_batch(repo_root, manifest, args.batch_id, hf_cache_dir)
            build_status(repo_root, manifest)
            print_json(result)
            return 0 if result["status"] in {"completed", "skipped_valid"} else 1
        if args.command == "run-missing":
            result = run_missing(
                repo_root,
                manifest,
                hf_cache_dir=hf_cache_dir,
                parallelism=args.parallelism,
                max_runs=args.max_runs,
            )
            print_json(result)
            return 0 if not result["status"]["missing_batches"] and not result["status"]["invalid_batches"] else 1
        if args.command == "merge":
            print_json(run_merge(repo_root, manifest))
            return 0
        if args.command == "retrain":
            print_json(run_retrain(repo_root))
            return 0
        if args.command == "final-report":
            print_json(final_report(repo_root, manifest))
            return 0
    except Exception as exc:
        print_json({"error": str(exc), "command": args.command})
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
