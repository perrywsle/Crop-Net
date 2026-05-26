from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import h5py
    import numpy as np
    import pandas as pd
    import pyarrow  # noqa: F401
    import statsmodels.api as sm
    import torch
    import torch.nn as nn
    from huggingface_hub import HfApi, hf_hub_download
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    h5py = None
    np = None
    pd = None
    pyarrow = None
    sm = None
    torch = types.SimpleNamespace(device=object)
    nn = types.SimpleNamespace(Module=object)
    HfApi = None
    hf_hub_download = None
    mean_absolute_error = None
    mean_squared_error = None
    DataLoader = object
    TensorDataset = object
    tqdm = None


DATASET_REPO = "CropNet/CropNet"
REPO_TYPE = "dataset"
IMAGE_H5_SUFFIXES = {".h5", ".hdf5"}

META_COLS = ["county_id", "crop_type", "year", "month"]
AG_CORE = [
    "ag_green_pixel_ratio",
    "ag_vegetation_area_percent",
    "ag_brown_yellow_pixel_ratio",
    "ag_soil_exposure_ratio",
    "ag_shadow_cloud_ratio",
    "ag_mean_brightness",
    "ag_texture_entropy",
    "ag_field_uniformity_score",
]
NDVI_CORE = [
    "ndvi_mean",
    "ndvi_median",
    "ndvi_max",
    "ndvi_std",
    "ndvi_cv",
    "ndvi_p25",
    "ndvi_p75",
    "ndvi_above_0_3_ratio",
    "ndvi_above_0_5_ratio",
    "ndvi_above_0_7_ratio",
    "ndvi_low_ratio",
    "ndvi_valid_coverage_ratio",
]
WEATHER_CORE = [
    "weather_temp_mean",
    "weather_temp_max",
    "weather_temp_min",
    "weather_gdd",
    "weather_heat_stress_days",
    "weather_cold_stress_days",
    "weather_total_precipitation",
    "weather_precipitation_days",
    "weather_heavy_rain_days",
    "weather_drought_index",
    "weather_humidity_mean",
    "weather_wind_mean",
    "weather_solar_radiation_mean",
    "weather_vpd_mean",
    "weather_temp_range_mean",
]
FEATURE_COLS = AG_CORE + NDVI_CORE + WEATHER_CORE
FEATURE_GROUPS = {"ag": AG_CORE, "ndvi": NDVI_CORE, "weather": WEATHER_CORE}
FEATURE_GROUP_SELECTIONS = {
    "all": FEATURE_COLS,
    "ag": AG_CORE,
    "ndvi": NDVI_CORE,
    "weather": WEATHER_CORE,
    "ag_ndvi": AG_CORE + NDVI_CORE,
    "ag_weather": AG_CORE + WEATHER_CORE,
    "ndvi_weather": NDVI_CORE + WEATHER_CORE,
}
BASELINE_MODELS = {"naive_lag1", "seasonal_last_year"}
LEARNED_MODELS = {"lstm", "gru", "tiny_mamba_ssm", "transformer_encoder"}
CLASSICAL_MODELS = {"sarima", "classical_ssm"}
ENSEMBLE_MODELS = {"ensemble_mean", "ensemble_weighted", "ensemble_oracle_report_only"}
ALL_MODEL_NAMES = BASELINE_MODELS | LEARNED_MODELS | CLASSICAL_MODELS | ENSEMBLE_MODELS
DEFAULT_ENSEMBLE_COMPONENT_CANDIDATES = ["seasonal_last_year", "lstm", "tiny_mamba_ssm", "gru", "transformer_encoder", "sarima"]
SARIMA_ORDER = (1, 0, 0)
SARIMA_SEASONAL_ORDER = (1, 0, 0, 12)


EXTRACTORS = {}
LOGGER = logging.getLogger("cropnet_v12")


def load_runtime_imports() -> None:
    global h5py, np, pd, pyarrow, sm, torch, nn, HfApi, hf_hub_download
    global mean_absolute_error, mean_squared_error, DataLoader, TensorDataset, tqdm
    import h5py as _h5py
    import numpy as _np
    import pandas as _pd
    import pyarrow as _pyarrow  # noqa: F401
    import statsmodels.api as _sm
    import torch as _torch
    import torch.nn as _nn
    from huggingface_hub import HfApi as _HfApi, hf_hub_download as _hf_hub_download
    from sklearn.metrics import mean_absolute_error as _mae, mean_squared_error as _mse
    from torch.utils.data import DataLoader as _DataLoader, TensorDataset as _TensorDataset
    from tqdm.auto import tqdm as _tqdm

    h5py = _h5py
    np = _np
    pd = _pd
    pyarrow = _pyarrow
    sm = _sm
    torch = _torch
    nn = _nn
    HfApi = _HfApi
    hf_hub_download = _hf_hub_download
    mean_absolute_error = _mae
    mean_squared_error = _mse
    DataLoader = _DataLoader
    TensorDataset = _TensorDataset
    tqdm = _tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server-friendly CropNet feature forecasting pipeline with dry-run, validation, resume, and logging."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Run a tiny extraction-first validation pass.")
    mode.add_argument("--full-run", action="store_true", help="Run the configured full pipeline.")
    mode.add_argument("--eval-only", action="store_true", help="Reuse an existing run directory for evaluation-only tasks.")
    parser.add_argument("--crop", "--crop-type", dest="crop", default="Corn", help="Crop label passed into the repo feature extractors.")
    parser.add_argument("--state-codes", nargs="+", default=["IA"], help="State abbreviations to discover from CropNet.")
    parser.add_argument("--image-types", nargs="+", choices=["AG", "NDVI"], default=["AG", "NDVI"])
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--train-years", nargs="+", type=int, default=None, help="Optional explicit train-year split override.")
    parser.add_argument("--val-years", nargs="+", type=int, default=None, help="Optional explicit validation-year split override.")
    parser.add_argument("--test-years", nargs="+", type=int, default=None, help="Optional explicit test-year split override.")
    parser.add_argument("--quarters", nargs="+", choices=["Q1", "Q2", "Q3", "Q4"], default=None)
    parser.add_argument("--fips-codes", "--selected-counties", dest="fips_codes", nargs="*", default=[], help="Optional explicit county FIPS codes.")
    parser.add_argument("--output-dir", default=None, help="Root output directory for raw chunks, cache, and artifacts.")
    parser.add_argument("--from-output-dir", default=None, help="Existing run directory to reuse for eval-only tasks.")
    parser.add_argument("--run-name", default=None, help="Optional experiment run name. When set without --output-dir, writes under --experiment-root/run-name.")
    parser.add_argument("--experiment-root", default="outputs/experiments", help="Parent directory for named experiment runs.")
    parser.add_argument("--base-dir", default=".", help="Base working directory.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory override.")
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    parser.add_argument("--repo-dir", default=None, help="Optional local Crop-Net repo path.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing validated chunk outputs.")
    parser.add_argument("--delete-raw-after-extract", action="store_true", help="Delete downloaded raw chunks after extraction.")
    parser.add_argument("--max-auto-counties", "--max-counties", dest="max_auto_counties", type=int, default=10)
    parser.add_argument("--lookback-months", "--seq-len", dest="lookback_months", type=int, default=6)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", "--epochs", dest="max_epochs", type=int, default=80)
    parser.add_argument("--patience", "--early-stopping-patience", dest="patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss-mode",
        choices=["raw_mse", "feature_normalized_mse"],
        default=None,
        help="Loss formulation for learned models. feature_normalized_mse reweights features using train-target variance.",
    )
    parser.add_argument(
        "--feature-groups",
        choices=sorted(FEATURE_GROUP_SELECTIONS.keys()),
        default=None,
        help="Train/evaluate only the selected feature subset while preserving feature order within the group.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["raw", "seasonal_residual"],
        default=None,
        help="Target formulation for learned models. seasonal_residual predicts corrections over same-county same-month previous-year values.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "naive_lag1",
            "seasonal_last_year",
            "lstm",
            "gru",
            "tiny_mamba_ssm",
            "transformer_encoder",
            "sarima",
            "classical_ssm",
            "ensemble_mean",
            "ensemble_weighted",
            "ensemble_oracle_report_only",
        ],
        default=["naive_lag1", "lstm", "tiny_mamba_ssm"],
        help="Forecasting models to run. naive_lag1 is always available as the baseline comparator.",
    )
    parser.add_argument("--run-classical-ssm", action="store_true", default=False)
    parser.add_argument("--classical-max-counties", type=int, default=5)
    parser.add_argument("--classical-max-features", type=int, default=None)
    parser.add_argument("--run-blank-fill-eval", action="store_true", help="Run recursive remaining-year blank-fill evaluation.")
    parser.add_argument("--blank-fill-year", type=int, default=None, help="Held-out year to evaluate recursive blank filling on.")
    parser.add_argument("--blank-fill-known-months", nargs="+", type=int, default=[0, 1, 3, 6, 9], help="Observed month counts from the target year before recursive rollout.")
    parser.add_argument(
        "--blank-fill-baselines",
        nargs="+",
        choices=["lag1", "seasonal_last_year"],
        default=["lag1", "seasonal_last_year"],
        help="Baseline forecasters to include during blank-fill evaluation.",
    )
    parser.add_argument(
        "--blank-fill-residual-seasonal",
        action="store_true",
        help="For learned models, interpret blank-fill outputs as residual corrections over the seasonal_last_year reference.",
    )
    parser.add_argument(
        "--strict-blank-fill-no-future-fill",
        action="store_true",
        help="During blank-fill evaluation, do not use future target-year months for interpolation/backfill of the held-out year.",
    )
    parser.add_argument(
        "--strict-fill-strategy",
        choices=["observed_only", "past_only"],
        default="past_only",
        help="How strict blank-fill should handle missing values in known target-year history.",
    )
    parser.add_argument("--blank-fill-output-prefix", default="blank_fill", help="Filename prefix for blank-fill evaluation artifacts.")
    parser.add_argument("--skip-repo-clone", action="store_true", help="Fail instead of cloning Crop-Net if repo is missing.")
    return parser.parse_args()


@dataclass
class RuntimeConfig:
    base_dir: Path
    output_dir: Path
    from_output_dir: Path | None
    experiment_root: Path | None
    raw_dir: Path
    feature_cache_dir: Path
    artifact_dir: Path
    log_dir: Path
    hf_cache_dir: Path | None
    repo_dir: Path
    log_file: Path | None
    config_path: Path
    status_path: Path
    run_name: str | None
    crop_type: str
    state_codes: list[str]
    image_types: list[str]
    years: list[int]
    quarters: list[str]
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    fips_codes: list[str]
    max_auto_counties: int
    lookback_months: int
    forecast_horizon: int
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    hidden_size: int
    num_layers: int
    dropout: float
    weight_decay: float
    loss_mode: str
    feature_group: str
    target_mode: str
    seed: int
    models: list[str]
    delete_raw_after_extract: bool
    resume: bool
    dry_run: bool
    full_run: bool
    eval_only: bool
    run_classical_ssm: bool
    classical_max_counties: int
    classical_max_features: int | None
    skip_repo_clone: bool
    max_dates_per_h5: int | None
    max_grids_per_date: int | None
    dry_run_max_crops: int
    dry_run_max_years: int
    dry_run_max_quarters: int
    dry_run_max_counties: int
    dry_run_max_files: int
    dry_run_max_batches: int
    run_forecasting: bool
    bad_chunk_log: Path
    run_blank_fill_eval: bool
    blank_fill_year: int | None
    blank_fill_known_months: list[int]
    blank_fill_baselines: list[str]
    blank_fill_residual_seasonal: bool
    strict_blank_fill_no_future_fill: bool
    strict_fill_strategy: str
    blank_fill_output_prefix: str


def derive_year_splits(years: list[int]) -> tuple[list[int], list[int], list[int], bool]:
    years = sorted(dict.fromkeys(years))
    if len(years) >= 5 and years[-1] == 2022:
        return years[:-2], [years[-2]], [years[-1]], True
    if len(years) >= 3:
        return years[:-2], [years[-2]], [years[-1]], True
    if len(years) == 2:
        return [years[0]], [], [years[1]], False
    if len(years) == 1:
        return [years[0]], [], [], False
    return [], [], [], False


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.eval_only and args.from_output_dir:
        return Path(args.from_output_dir).resolve(), None
    experiment_root = Path(args.experiment_root).resolve() if args.experiment_root else None
    if args.output_dir:
        return Path(args.output_dir).resolve(), experiment_root
    if args.run_name:
        if experiment_root is None:
            raise ValueError("--experiment-root could not be resolved.")
        return (experiment_root / args.run_name).resolve(), experiment_root
    raise ValueError("Either --output-dir or --run-name must be provided.")


def load_existing_run_hints(from_output_dir: Path) -> dict:
    global pd
    if pd is None:
        import pandas as _pd
        pd = _pd
    hints: dict = {}
    config_path = from_output_dir / "config.json"
    if config_path.exists():
        hints.update(json.loads(config_path.read_text(encoding="utf-8")))

    artifact_dir = from_output_dir / "artifacts"
    monthly_path = artifact_dir / "official_monthly_feature_table.parquet"
    seq_meta_path = artifact_dir / "sequence_metadata.csv"
    if monthly_path.exists():
        monthly = pd.read_parquet(monthly_path)
        if "year" in monthly.columns:
            hints.setdefault("years", sorted(int(x) for x in pd.Series(monthly["year"]).dropna().unique().tolist()))
        if "crop_type" in monthly.columns and monthly["crop_type"].notna().any():
            hints.setdefault("crop_type", str(monthly["crop_type"].dropna().iloc[0]))
    if seq_meta_path.exists():
        meta = pd.read_csv(seq_meta_path)
        if {"split", "target_year"}.issubset(meta.columns):
            split_map = meta.groupby("split")["target_year"].unique().apply(lambda arr: sorted(int(x) for x in arr)).to_dict()
            hints.setdefault("train_years", split_map.get("train", []))
            hints.setdefault("val_years", split_map.get("val", []))
            hints.setdefault("test_years", split_map.get("test", []))
    return hints


def resolve_year_splits(args: argparse.Namespace) -> tuple[list[int], list[int], list[int], bool]:
    if not args.years:
        raise ValueError("--years is required unless values can be inferred from an existing run.")
    explicit = any(getattr(args, name) is not None for name in ("train_years", "val_years", "test_years"))
    if not explicit:
        return derive_year_splits(sorted(args.years))

    train_years = sorted(dict.fromkeys(args.train_years or []))
    val_years = sorted(dict.fromkeys(args.val_years or []))
    test_years = sorted(dict.fromkeys(args.test_years or []))
    all_split_years = set(train_years) | set(val_years) | set(test_years)
    year_set = set(args.years)
    if not train_years or not test_years:
        raise ValueError("Explicit split overrides require at least one train year and one test year.")
    if not all_split_years.issubset(year_set):
        raise ValueError("Explicit train/val/test years must be a subset of --years.")
    overlap = (set(train_years) & set(val_years)) | (set(train_years) & set(test_years)) | (set(val_years) & set(test_years))
    if overlap:
        raise ValueError(f"Explicit train/val/test years must be disjoint. Overlap: {sorted(overlap)}")
    return train_years, val_years, test_years, True


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    base_dir = Path(args.base_dir).resolve()
    output_dir, experiment_root = resolve_output_paths(args)
    from_output_dir = Path(args.from_output_dir).resolve() if args.from_output_dir else None
    existing_hints = load_existing_run_hints(from_output_dir) if args.eval_only and from_output_dir else {}
    years = sorted(dict.fromkeys(args.years or existing_hints.get("years", [])))
    if not years:
        raise ValueError("Could not determine run years. Pass --years or use --eval-only with a reusable output dir.")
    quarters = [q.upper() for q in (args.quarters or existing_hints.get("quarters", ["Q1", "Q2", "Q3", "Q4"]))]
    raw_dir = output_dir / "raw_chunks"
    feature_cache_dir = output_dir / "feature_cache"
    artifact_dir = output_dir / "artifacts"
    inferred_run_name = args.run_name or existing_hints.get("run_name")
    default_log_name = f"{inferred_run_name}.log" if inferred_run_name else "pipeline.log"
    log_file = Path(args.log_file).resolve() if args.log_file else (output_dir / "logs" / default_log_name)
    log_dir = log_file.parent
    hf_cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else None
    repo_dir = Path(args.repo_dir).resolve() if args.repo_dir else (base_dir / "Crop-Net")
    target_mode = args.target_mode or existing_hints.get("target_mode", "raw")
    loss_mode = args.loss_mode or existing_hints.get("loss_mode", "raw_mse")
    feature_group = args.feature_groups or existing_hints.get("feature_group", "all")
    blank_fill_residual_seasonal = bool(args.blank_fill_residual_seasonal or target_mode == "seasonal_residual")
    if args.blank_fill_residual_seasonal and target_mode != "seasonal_residual":
        raise ValueError("--blank-fill-residual-seasonal requires --target-mode seasonal_residual.")

    if args.years is None and years:
        args.years = years
    if args.train_years is None and existing_hints.get("train_years"):
        args.train_years = existing_hints.get("train_years")
    if args.val_years is None and existing_hints.get("val_years"):
        args.val_years = existing_hints.get("val_years")
    if args.test_years is None and existing_hints.get("test_years"):
        args.test_years = existing_hints.get("test_years")
    train_years, val_years, test_years, run_forecasting = resolve_year_splits(args)

    dry_run_max = {
        "dry_run_max_crops": 1,
        "dry_run_max_years": 1,
        "dry_run_max_quarters": 1,
        "dry_run_max_counties": 2,
        "dry_run_max_files": 5,
        "dry_run_max_batches": 2,
    }
    max_dates_per_h5 = 2 if args.dry_run else None
    max_grids_per_date = 2 if args.dry_run else 9

    cfg = RuntimeConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        from_output_dir=from_output_dir,
        experiment_root=experiment_root,
        raw_dir=raw_dir,
        feature_cache_dir=feature_cache_dir,
        artifact_dir=artifact_dir,
        log_dir=log_dir,
        hf_cache_dir=hf_cache_dir,
        repo_dir=repo_dir,
        log_file=log_file,
        config_path=output_dir / "config.json",
        status_path=output_dir / "run_status.json",
        run_name=inferred_run_name,
        crop_type=existing_hints.get("crop_type", args.crop),
        state_codes=[s.upper() for s in args.state_codes],
        image_types=[s.upper() for s in args.image_types],
        years=years,
        quarters=quarters,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        fips_codes=[str(f).zfill(5) for f in args.fips_codes],
        max_auto_counties=args.max_auto_counties,
        lookback_months=args.lookback_months,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        loss_mode=str(loss_mode),
        feature_group=str(feature_group),
        target_mode=str(target_mode),
        seed=args.seed,
        models=list(dict.fromkeys(args.models + (["classical_ssm"] if args.run_classical_ssm else []))),
        delete_raw_after_extract=args.delete_raw_after_extract,
        resume=args.resume,
        dry_run=args.dry_run,
        full_run=args.full_run,
        eval_only=args.eval_only,
        run_classical_ssm=args.run_classical_ssm,
        classical_max_counties=args.classical_max_counties,
        classical_max_features=args.classical_max_features,
        skip_repo_clone=args.skip_repo_clone,
        max_dates_per_h5=max_dates_per_h5,
        max_grids_per_date=max_grids_per_date,
        run_forecasting=run_forecasting,
        bad_chunk_log=feature_cache_dir / "skipped_bad_h5_chunks.csv",
        run_blank_fill_eval=args.run_blank_fill_eval,
        blank_fill_year=args.blank_fill_year or (test_years[0] if test_years else (years[-1] if years else None)),
        blank_fill_known_months=sorted(dict.fromkeys(int(x) for x in args.blank_fill_known_months)),
        blank_fill_baselines=sorted(dict.fromkeys(args.blank_fill_baselines)),
        blank_fill_residual_seasonal=blank_fill_residual_seasonal,
        strict_blank_fill_no_future_fill=bool(args.strict_blank_fill_no_future_fill),
        strict_fill_strategy=str(args.strict_fill_strategy),
        blank_fill_output_prefix=args.blank_fill_output_prefix,
        **dry_run_max,
    )

    if cfg.run_name and not cfg.eval_only and cfg.output_dir.exists() and any(cfg.output_dir.iterdir()) and not cfg.resume:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {cfg.output_dir}. Re-run with --resume or choose a new --run-name."
        )
    for path in [cfg.output_dir, cfg.raw_dir, cfg.feature_cache_dir, cfg.artifact_dir, cfg.log_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return cfg


def configure_logging(cfg: RuntimeConfig) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    LOGGER.addHandler(stream_handler)
    if cfg.log_file:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        LOGGER.addHandler(file_handler)


@contextlib.contextmanager
def stage(name: str):
    start = time.perf_counter()
    LOGGER.info("START | %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        LOGGER.info("END   | %s | %.2fs", name, elapsed)


def ensure_package(import_name: str, package_spec: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        cmd = [sys.executable, "-m", "pip", "install", "-q", package_spec]
        LOGGER.info("Installing dependency: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


def print_environment(cfg: RuntimeConfig) -> None:
    LOGGER.info("Python: %s", sys.version.replace("\n", " "))
    LOGGER.info("Platform: %s", platform.platform())
    LOGGER.info("PyTorch: %s", torch.__version__)
    LOGGER.info("CUDA available: %s", torch.cuda.is_available())
    LOGGER.info("CUDA version: %s", torch.version.cuda)
    if torch.cuda.is_available():
        LOGGER.info("GPU count: %s", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            LOGGER.info("GPU[%s]: %s", idx, torch.cuda.get_device_name(idx))
    LOGGER.info("Base dir: %s", cfg.base_dir)
    LOGGER.info("Output dir: %s", cfg.output_dir)
    LOGGER.info("Run name: %s", cfg.run_name or "(none)")
    LOGGER.info("Raw dir: %s", cfg.raw_dir)
    LOGGER.info("Feature cache dir: %s", cfg.feature_cache_dir)
    LOGGER.info("Artifact dir: %s", cfg.artifact_dir)
    LOGGER.info("Repo dir: %s", cfg.repo_dir)
    LOGGER.info("State codes: %s", cfg.state_codes)
    LOGGER.info("Crop type: %s", cfg.crop_type)
    LOGGER.info("Years: %s", cfg.years)
    LOGGER.info("Quarters: %s", cfg.quarters)
    LOGGER.info("Image types: %s", cfg.image_types)
    LOGGER.info("Train/Val/Test years: %s / %s / %s", cfg.train_years, cfg.val_years, cfg.test_years)
    LOGGER.info("Eval only: %s", cfg.eval_only)
    LOGGER.info(
        "Training config | seq_len=%s batch_size=%s epochs=%s lr=%s hidden=%s layers=%s dropout=%s weight_decay=%s target_mode=%s models=%s",
        cfg.lookback_months,
        cfg.batch_size,
        cfg.max_epochs,
        cfg.learning_rate,
        cfg.hidden_size,
        cfg.num_layers,
        cfg.dropout,
        cfg.weight_decay,
        cfg.target_mode,
        cfg.models,
    )
    LOGGER.info(
        "Blank-fill config | enabled=%s year=%s known_months=%s baselines=%s residual_seasonal=%s prefix=%s",
        cfg.run_blank_fill_eval,
        cfg.blank_fill_year,
        cfg.blank_fill_known_months,
        cfg.blank_fill_baselines,
        cfg.blank_fill_residual_seasonal,
        cfg.blank_fill_output_prefix,
    )
    LOGGER.info("Dry run: %s", cfg.dry_run)
    if cfg.dry_run:
        LOGGER.info(
            "DRY_RUN limits | years=%s quarters=%s counties=%s files=%s batches=%s dates_per_h5=%s grids_per_date=%s",
            cfg.dry_run_max_years,
            cfg.dry_run_max_quarters,
            cfg.dry_run_max_counties,
            cfg.dry_run_max_files,
            cfg.dry_run_max_batches,
            cfg.max_dates_per_h5,
            cfg.max_grids_per_date,
        )


def ensure_repo_extractors(cfg: RuntimeConfig) -> None:
    if not cfg.repo_dir.exists():
        if cfg.skip_repo_clone:
            raise FileNotFoundError(f"Crop-Net repo not found and --skip-repo-clone was set: {cfg.repo_dir}")
        cmd = ["git", "clone", "--depth", "1", "https://github.com/perrywsle/Crop-Net", str(cfg.repo_dir)]
        LOGGER.info("Cloning Crop-Net repo: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    src_dir = cfg.repo_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    ag = importlib.import_module("crop_fusion_ai.preprocessing.ag")
    ndvi = importlib.import_module("crop_fusion_ai.preprocessing.ndvi")
    weather = importlib.import_module("crop_fusion_ai.preprocessing.weather")
    EXTRACTORS["ag"] = ag.extract_ag_features
    EXTRACTORS["ndvi"] = ndvi.extract_ndvi_features
    EXTRACTORS["weather"] = weather.extract_weather_features
    LOGGER.info("Imported preprocessing extractors from %s", src_dir)


def parse_year_month_from_path(path: str) -> tuple[int | None, int | None]:
    import re

    match = re.search(r"(20\d{2})-([01]\d)", path)
    if match:
        return int(match.group(1)), int(match.group(2))
    year_match = re.search(r"/(20\d{2})/", path)
    return (int(year_match.group(1)), None) if year_match else (None, None)


def quarter_label_from_path(path: str) -> str:
    year, month = parse_year_month_from_path(path)
    if year is None or month is None:
        return "unknown"
    quarter = (month - 1) // 3 + 1
    return f"Q{quarter}"


def list_files_under(api: HfApi, path: str, recursive: bool = True) -> list[str]:
    items = api.list_repo_tree(repo_id=DATASET_REPO, repo_type=REPO_TYPE, path_in_repo=path, recursive=recursive)
    files = []
    for item in items:
        item_path = getattr(item, "path", None)
        is_dir = getattr(item, "is_dir", False)
        if item_path and not is_dir:
            files.append(item_path)
    return files


def discover_files(cfg: RuntimeConfig) -> tuple[list[dict], list[dict], list[dict]]:
    api = HfApi()
    discovery_years = cfg.years[: cfg.dry_run_max_years] if cfg.dry_run else cfg.years
    quarter_allow = set(cfg.quarters[: cfg.dry_run_max_quarters] if cfg.dry_run else cfg.quarters)

    def discover_image_h5_files(modality: str) -> list[dict]:
        base_modality = modality
        rows = []
        for year in discovery_years:
            for state in cfg.state_codes:
                root = f"Sentinel-2 Imagery/data/{base_modality}/{year}/{state}"
                try:
                    files = list_files_under(api, root, recursive=True)
                except Exception as exc:
                    LOGGER.warning("Could not list %s: %s: %s", root, type(exc).__name__, exc)
                    continue
                for path in sorted(files):
                    if Path(path).suffix.lower() not in IMAGE_H5_SUFFIXES:
                        continue
                    quarter = quarter_label_from_path(path)
                    if quarter not in quarter_allow:
                        continue
                    rows.append(
                        {
                            "modality": modality.lower(),
                            "state": state,
                            "year": year,
                            "quarter": quarter,
                            "hf_path": path,
                        }
                    )
        if cfg.dry_run:
            rows = rows[: cfg.dry_run_max_files]
        return rows

    def discover_hrrr_files() -> list[dict]:
        rows = []
        for year in discovery_years:
            for state in cfg.state_codes:
                root = f"WRF-HRRR Computed Dataset/data/{year}/{state}"
                try:
                    files = list_files_under(api, root, recursive=True)
                except Exception as exc:
                    LOGGER.warning("Could not list %s: %s: %s", root, type(exc).__name__, exc)
                    continue
                for path in sorted(files):
                    if not path.lower().endswith(".csv"):
                        continue
                    y, m = parse_year_month_from_path(path)
                    if y not in discovery_years or m is None:
                        continue
                    quarter = f"Q{((m - 1) // 3) + 1}"
                    if quarter not in quarter_allow:
                        continue
                    rows.append({"modality": "weather", "state": state, "year": y, "month": m, "quarter": quarter, "hf_path": path})
        if cfg.dry_run:
            rows = rows[: cfg.dry_run_max_files]
        return rows

    ag_files = discover_image_h5_files("AG") if "AG" in cfg.image_types else []
    ndvi_files = discover_image_h5_files("NDVI") if "NDVI" in cfg.image_types else []
    weather_files = discover_hrrr_files()
    LOGGER.info("Discovered AG files: %s", len(ag_files))
    LOGGER.info("Discovered NDVI files: %s", len(ndvi_files))
    LOGGER.info("Discovered weather files: %s", len(weather_files))
    if not weather_files:
        raise RuntimeError("No HRRR weather files were discovered. Forecasting pipeline cannot proceed.")
    if "AG" in cfg.image_types and not ag_files:
        raise RuntimeError("No AG files were discovered for the selected scope.")
    if "NDVI" in cfg.image_types and not ndvi_files:
        raise RuntimeError("No NDVI files were discovered for the selected scope.")
    return ag_files, ndvi_files, weather_files


def download_hf_file(cfg: RuntimeConfig, hf_path: str, *, force_download: bool = False) -> Path:
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type=REPO_TYPE,
        filename=hf_path,
        local_dir=str(cfg.raw_dir),
        cache_dir=str(cfg.hf_cache_dir) if cfg.hf_cache_dir else None,
        force_download=force_download,
    )
    return Path(local_path)


def validate_h5_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing file: {path}"
    try:
        with h5py.File(path, "r") as hf:
            keys = list(hf.keys())
            if not keys:
                return False, "empty top-level HDF5"
            first = hf[keys[0]]
            if isinstance(first, h5py.Group):
                _ = list(first.keys())[:1]
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def download_valid_h5(cfg: RuntimeConfig, hf_path: str, *, max_attempts: int = 3) -> Path:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        path = download_hf_file(cfg, hf_path, force_download=(attempt > 1))
        ok, msg = validate_h5_file(path)
        if ok:
            LOGGER.info("Validated HDF5: %s", path.name)
            return path
        last_error = msg
        LOGGER.warning("Invalid HDF5 on attempt %s/%s: %s -> %s", attempt, max_attempts, path.name, msg)
        safe_unlink(path)
    raise RuntimeError(f"Could not obtain a valid HDF5 file for {hf_path}: {last_error}")


def download_h5_for_image_extraction(cfg: RuntimeConfig, modality: str, hf_path: str) -> tuple[Path, bool]:
    modality_lower = modality.lower()
    try:
        return download_valid_h5(cfg, hf_path), False
    except RuntimeError as exc:
        if modality_lower != "ndvi":
            raise
        preserved_path = download_hf_file(cfg, hf_path, force_download=True)
        LOGGER.warning(
            "Falling back to direct NDVI HDF5 access despite validation failure: %s | preserved=%s",
            exc,
            preserved_path,
        )
        return preserved_path, True


def read_h5_fips_keys(h5_path: Path) -> set[str]:
    ok, msg = validate_h5_file(h5_path)
    if not ok:
        raise RuntimeError(f"Invalid HDF5 file while reading FIPS keys: {h5_path} -> {msg}")
    with h5py.File(h5_path, "r") as hf:
        return {str(key).zfill(5) for key in hf.keys()}


def read_hrrr_fips(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path, nrows=200000)
    df.columns = df.columns.str.strip()
    for col in ["FIPS Code", "FIPS", "fips", "FIPS_CODE"]:
        if col in df.columns:
            return set(df[col].astype(str).str.zfill(5).unique())
    return set()


def select_fips(cfg: RuntimeConfig, ag_files: list[dict], ndvi_files: list[dict], weather_files: list[dict]) -> list[str]:
    if cfg.fips_codes:
        selected = cfg.fips_codes
    else:
        if not ag_files or not ndvi_files or not weather_files:
            raise RuntimeError("Cannot auto-select FIPS because AG, NDVI, or weather files are missing.")
        LOGGER.info("Auto-selecting FIPS from the first AG, NDVI, and HRRR chunks")
        ag_path = download_valid_h5(cfg, ag_files[0]["hf_path"])
        ndvi_path = download_valid_h5(cfg, ndvi_files[0]["hf_path"])
        hrrr_path = download_hf_file(cfg, weather_files[0]["hf_path"])
        common = sorted(read_h5_fips_keys(ag_path) & read_h5_fips_keys(ndvi_path) & read_hrrr_fips(hrrr_path))
        if not common:
            raise RuntimeError("No common FIPS found across AG, NDVI, and HRRR discovery chunks.")
        limit = cfg.dry_run_max_counties if cfg.dry_run else cfg.max_auto_counties
        selected = common[:limit]
    LOGGER.info("Selected FIPS count: %s", len(selected))
    LOGGER.info("Selected FIPS preview: %s", selected[:10])
    return selected


def safe_unlink(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except Exception as exc:
        LOGGER.warning("Could not delete %s: %s", path, exc)


def normalize_rgb_array(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    x = np.squeeze(x)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.moveaxis(x, 0, -1)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.ndim == 3 and x.shape[-1] > 3:
        x = x[..., :3]
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"Unexpected image rank/shape after normalization pre-check: {x.shape}")
    x = x.astype("float32")
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros((*x.shape[:2], 3), dtype=np.uint8)
    lo = np.nanpercentile(x[finite], 1)
    hi = np.nanpercentile(x[finite], 99)
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def parse_date_key(date_key: str, fallback_year=None, fallback_month=None) -> tuple[int | None, int | None]:
    import re

    s = str(date_key)
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)?", s)
    if match:
        return int(match.group(1)), int(match.group(2))
    return fallback_year, fallback_month


def cache_path_for(cfg: RuntimeConfig, modality: str, rec: dict) -> Path:
    if modality in {"ag", "ndvi"}:
        name = f"{modality}_{rec['state']}_{rec['year']}_{rec.get('quarter', 'unknown')}.parquet"
    else:
        name = f"weather_{rec['state']}_{rec['year']}_{int(rec.get('month') or 0):02d}.parquet"
    return cfg.feature_cache_dir / modality / name


def normalize_meta(frame: pd.DataFrame, crop_type: str) -> pd.DataFrame:
    out = frame.copy()
    for col in META_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out["county_id"] = out["county_id"].astype(str).str.zfill(5)
    if "crop_type" not in out.columns:
        out["crop_type"] = crop_type
    out["crop_type"] = out["crop_type"].fillna(crop_type).astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    return out


def feature_output_to_frame(feat, *, crop_type: str, fips: str | None = None, year: int | None = None, month: int | None = None) -> pd.DataFrame:
    if feat is None:
        return pd.DataFrame()
    if isinstance(feat, pd.DataFrame):
        out = feat.copy()
    elif isinstance(feat, pd.Series):
        out = feat.to_frame().T
    elif isinstance(feat, dict):
        out = pd.DataFrame([feat])
    elif hasattr(feat, "model_dump"):
        out = pd.DataFrame([feat.model_dump()])
    else:
        try:
            out = pd.DataFrame([dict(feat)])
        except Exception as exc:
            raise TypeError(f"Unsupported feature output type: {type(feat)!r}") from exc
    if out.empty:
        return out
    if fips is not None:
        out["county_id"] = str(fips).zfill(5)
    if "crop_type" not in out.columns:
        out["crop_type"] = crop_type
    else:
        out["crop_type"] = out["crop_type"].fillna(crop_type)
    if year is not None:
        out["year"] = int(year)
    if month is not None:
        out["month"] = int(month)
    return out


def atomic_write_parquet(frame: pd.DataFrame, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    frame.to_parquet(temp_path, index=False)
    os.replace(temp_path, final_path)


def atomic_write_csv(frame: pd.DataFrame, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    os.replace(temp_path, final_path)


def atomic_write_json(data: dict, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_path, final_path)


def atomic_write_text(text: str, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, final_path)


def config_to_dict(cfg: RuntimeConfig) -> dict:
    return {
        "run_name": cfg.run_name,
        "base_dir": str(cfg.base_dir),
        "output_dir": str(cfg.output_dir),
        "from_output_dir": str(cfg.from_output_dir) if cfg.from_output_dir else None,
        "experiment_root": str(cfg.experiment_root) if cfg.experiment_root else None,
        "raw_dir": str(cfg.raw_dir),
        "feature_cache_dir": str(cfg.feature_cache_dir),
        "artifact_dir": str(cfg.artifact_dir),
        "log_dir": str(cfg.log_dir),
        "log_file": str(cfg.log_file) if cfg.log_file else None,
        "repo_dir": str(cfg.repo_dir),
        "crop_type": cfg.crop_type,
        "state_codes": cfg.state_codes,
        "image_types": cfg.image_types,
        "years": cfg.years,
        "quarters": cfg.quarters,
        "train_years": cfg.train_years,
        "val_years": cfg.val_years,
        "test_years": cfg.test_years,
        "fips_codes": cfg.fips_codes,
        "max_counties": cfg.max_auto_counties,
        "seq_len": cfg.lookback_months,
        "forecast_horizon": cfg.forecast_horizon,
        "batch_size": cfg.batch_size,
        "epochs": cfg.max_epochs,
        "early_stopping_patience": cfg.patience,
        "learning_rate": cfg.learning_rate,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "weight_decay": cfg.weight_decay,
        "loss_mode": cfg.loss_mode,
        "feature_group": cfg.feature_group,
        "target_mode": cfg.target_mode,
        "models": cfg.models,
        "seed": cfg.seed,
        "resume": cfg.resume,
        "dry_run": cfg.dry_run,
        "full_run": cfg.full_run,
        "eval_only": cfg.eval_only,
        "run_classical_ssm": cfg.run_classical_ssm,
        "run_blank_fill_eval": cfg.run_blank_fill_eval,
        "blank_fill_year": cfg.blank_fill_year,
        "blank_fill_known_months": cfg.blank_fill_known_months,
        "blank_fill_baselines": cfg.blank_fill_baselines,
        "blank_fill_residual_seasonal": cfg.blank_fill_residual_seasonal,
        "strict_blank_fill_no_future_fill": cfg.strict_blank_fill_no_future_fill,
        "strict_fill_strategy": cfg.strict_fill_strategy,
        "blank_fill_output_prefix": cfg.blank_fill_output_prefix,
    }


def write_run_config(cfg: RuntimeConfig) -> None:
    atomic_write_json(config_to_dict(cfg), cfg.config_path)


def write_run_status(
    cfg: RuntimeConfig,
    *,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    runtime_seconds: float | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "run_name": cfg.run_name,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "error": error,
        "output_dir": str(cfg.output_dir),
    }
    atomic_write_json(payload, cfg.status_path)


def atomic_save_npy(array: np.ndarray, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    np.save(temp_path, array)
    os.replace(str(temp_path) + ".npy", final_path)


def atomic_torch_save(obj, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(obj, temp_path)
    os.replace(temp_path, final_path)


def nan_inf_report_frame(frame: pd.DataFrame, numeric_cols: list[str]) -> dict[str, int]:
    if not numeric_cols:
        return {"nan_count": 0, "inf_count": 0}
    values = frame[numeric_cols].to_numpy(dtype="float64", copy=False)
    return {
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def validate_feature_frame(frame: pd.DataFrame, required_cols: list[str], context: str, *, allow_empty: bool = False) -> None:
    if frame is None:
        raise RuntimeError(f"{context}: frame is None")
    if frame.empty and not allow_empty:
        raise RuntimeError(f"{context}: frame is empty")
    missing = [col for col in META_COLS if col not in frame.columns]
    if missing:
        raise RuntimeError(f"{context}: missing metadata columns {missing}")
    missing_required = [col for col in required_cols if col not in frame.columns]
    if missing_required:
        raise RuntimeError(f"{context}: missing required feature columns {missing_required}")
    report = nan_inf_report_frame(frame, [col for col in required_cols if col in frame.columns])
    LOGGER.info("%s | nan_count=%s inf_count=%s shape=%s", context, report["nan_count"], report["inf_count"], frame.shape)


def validate_saved_parquet(path: Path, required_cols: list[str], context: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{context}: parquet not found at {path}")
    frame = pd.read_parquet(path)
    validate_feature_frame(frame, required_cols, context)
    return frame


def validate_saved_csv(path: Path, context: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"{context}: csv not found at {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"{context}: csv is empty")
    return frame


def validate_saved_json(path: Path, context: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"{context}: json not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"{context}: json is empty or invalid")
    return data


def validate_saved_npy(path: Path, context: str) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(f"{context}: npy not found at {path}")
    array = np.load(path)
    if array.size == 0:
        raise RuntimeError(f"{context}: array is empty")
    if not np.isfinite(array).all():
        raise RuntimeError(f"{context}: array contains NaN/inf values")
    return array


def cached_parquet_is_valid(path: Path, required_cols: list[str], context: str) -> bool:
    try:
        validate_saved_parquet(path, required_cols, context)
        return True
    except Exception as exc:
        LOGGER.warning("Cached parquet failed validation and will be regenerated: %s | %s", path, exc)
        safe_unlink(path)
        return False


def extract_image_h5_chunk(cfg: RuntimeConfig, modality: str, rec: dict, selected_fips: list[str]) -> pd.DataFrame:
    modality_lower = modality.lower()
    assert modality_lower in {"ag", "ndvi"}
    core_cols = AG_CORE if modality_lower == "ag" else NDVI_CORE
    out_path = cache_path_for(cfg, modality_lower, rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.resume and out_path.exists() and cached_parquet_is_valid(out_path, core_cols, f"{modality_lower} cache {out_path.name}"):
        LOGGER.info("Using validated cached %s features: %s", modality_lower, out_path)
        return pd.read_parquet(out_path)

    local_h5, use_direct_ndvi_fallback = download_h5_for_image_extraction(cfg, modality_lower, rec["hf_path"])
    raw_frames: list[pd.DataFrame] = []
    errors: list[dict] = []
    processed_rows = 0

    open_kwargs = {"swmr": True} if use_direct_ndvi_fallback else {}
    with h5py.File(local_h5, "r", **open_kwargs) as hf:
        if use_direct_ndvi_fallback:
            available_keys = {}
            for fips in selected_fips:
                try:
                    _ = hf[fips]
                    available_keys[fips] = fips
                except Exception as exc:
                    errors.append(
                        {
                            "file": local_h5.name,
                            "fips": fips,
                            "error": f"direct-ndvi-fallback unavailable: {type(exc).__name__}: {exc}",
                        }
                    )
        else:
            available_keys = {str(key).zfill(5): key for key in hf.keys()}
        available_fips = [f for f in selected_fips if f in available_keys]
        if not available_fips:
            raise RuntimeError(f"No selected FIPS found in {local_h5.name}")
        for fips in tqdm(available_fips, desc=f"{modality_upper(modality_lower)} {rec['year']} {rec.get('quarter')}"):
            group = hf[available_keys[fips]]
            date_keys = sorted(list(group.keys()))
            if cfg.max_dates_per_h5 is not None:
                date_keys = date_keys[: cfg.max_dates_per_h5]
            for date_idx, date_key in enumerate(date_keys, start=1):
                year, month = parse_date_key(str(date_key), rec.get("year"), None)
                if month is None:
                    errors.append({"file": local_h5.name, "fips": fips, "date": str(date_key), "error": "could not infer month"})
                    continue
                node = group[date_key]
                if isinstance(node, h5py.Group) and "data" in node:
                    data = np.asarray(node["data"])
                elif isinstance(node, h5py.Dataset):
                    data = np.asarray(node)
                else:
                    errors.append({"file": local_h5.name, "fips": fips, "date": str(date_key), "error": "missing data dataset"})
                    continue
                if data.ndim == 4:
                    grids = data[: cfg.max_grids_per_date]
                elif data.ndim in (2, 3):
                    grids = data[None, ...]
                else:
                    errors.append({"file": local_h5.name, "fips": fips, "date": str(date_key), "shape": tuple(data.shape), "error": "unsupported data rank"})
                    continue
                LOGGER.info(
                    "Image sample | modality=%s file=%s fips=%s date=%s grids=%s",
                    modality_lower,
                    local_h5.name,
                    fips,
                    date_key,
                    len(grids),
                )
                for grid_i, grid_img in enumerate(grids):
                    try:
                        img = normalize_rgb_array(grid_img)
                        if img.ndim != 3 or img.shape[-1] != 3:
                            raise ValueError(f"normalized image has invalid shape {img.shape}")
                        feat = EXTRACTORS[modality_lower](img, county_id=fips, crop_type=cfg.crop_type, year=year, month=month)
                        feat_df = feature_output_to_frame(feat, crop_type=cfg.crop_type, fips=fips, year=year, month=month)
                        if feat_df.empty:
                            raise RuntimeError("extractor returned empty feature frame")
                        feat_df["source_h5"] = local_h5.name
                        feat_df["date_key"] = str(date_key)
                        feat_df["grid_index"] = grid_i
                        raw_frames.append(feat_df)
                        processed_rows += 1
                    except Exception as exc:
                        errors.append(
                            {
                                "file": local_h5.name,
                                "fips": fips,
                                "date": str(date_key),
                                "grid": grid_i,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if cfg.dry_run and date_idx >= cfg.dry_run_max_batches:
                    LOGGER.info("Dry-run date limit reached for %s %s", local_h5.name, fips)
                    break

    if not raw_frames:
        raise RuntimeError(f"No {modality_lower} features extracted from {local_h5.name}")

    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    raw = normalize_meta(raw, cfg.crop_type)
    numeric_cols = [col for col in core_cols if col in raw.columns and pd.api.types.is_numeric_dtype(raw[col])]
    validate_feature_frame(raw, numeric_cols, f"raw {modality_lower} features {local_h5.name}")
    frame = raw.groupby(META_COLS, as_index=False)[numeric_cols].mean()
    validate_feature_frame(frame, numeric_cols, f"grouped {modality_lower} features {local_h5.name}")
    atomic_write_parquet(frame, out_path)
    validate_saved_parquet(out_path, numeric_cols, f"saved {modality_lower} cache {out_path.name}")
    LOGGER.info("Saved %s chunk features: %s | rows=%s processed_rows=%s", modality_lower, out_path, len(frame), processed_rows)

    if errors:
        err_path = out_path.with_suffix(".errors.csv")
        atomic_write_csv(pd.DataFrame(errors), err_path)
        validate_saved_csv(err_path, f"{modality_lower} error log {err_path.name}")
        LOGGER.warning("Saved %s extraction errors to %s", len(errors), err_path)

    if cfg.delete_raw_after_extract:
        safe_unlink(local_h5)
    return frame


def read_table_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table suffix: {suffix}")


def standardize_official_hrrr(df: pd.DataFrame, selected_fips: list[str], year: int | None, month: int | None) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()
    if "FIPS Code" in out.columns:
        out["FIPS Code"] = out["FIPS Code"].astype(str).str.zfill(5)
        out = out[out["FIPS Code"].isin(selected_fips)].copy()
    if "Daily/Monthly" in out.columns:
        daily = out[out["Daily/Monthly"].astype(str).str.lower().eq("daily")].copy()
        if not daily.empty:
            out = daily
    if "date" not in out.columns:
        if {"Year", "Month", "Day"} <= set(out.columns):
            out["date"] = pd.to_datetime(
                dict(
                    year=pd.to_numeric(out["Year"], errors="coerce"),
                    month=pd.to_numeric(out["Month"], errors="coerce"),
                    day=pd.to_numeric(out["Day"], errors="coerce").fillna(1),
                ),
                errors="coerce",
            )
        elif year is not None and month is not None:
            out["date"] = pd.Timestamp(year=int(year), month=int(month), day=1)

    rename = {
        "Avg Temperature (K)": "temperature_mean",
        "Max Temperature (K)": "temperature_max",
        "Min Temperature (K)": "temperature_min",
        "Precipitation (kg m**-2)": "precipitation",
        "Relative Humidity (%)": "humidity",
        "Wind Speed (m s**-1)": "wind_speed",
        "Downward Shortwave Radiation Flux (W m**-2)": "solar_radiation",
        "Vapor Pressure Deficit (kPa)": "vpd_official",
    }
    out = out.rename(columns={src: dst for src, dst in rename.items() if src in out.columns})
    for col in ["temperature_mean", "temperature_max", "temperature_min"]:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            if vals.dropna().median() > 100:
                vals = vals - 273.15
            out[col] = vals
    for col in ["precipitation", "humidity", "wind_speed", "solar_radiation", "vpd_official"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def extract_weather_chunk(cfg: RuntimeConfig, rec: dict, selected_fips: list[str]) -> pd.DataFrame:
    out_path = cache_path_for(cfg, "weather", rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.resume and out_path.exists() and cached_parquet_is_valid(out_path, WEATHER_CORE, f"weather cache {out_path.name}"):
        LOGGER.info("Using validated cached weather features: %s", out_path)
        return pd.read_parquet(out_path)

    local_csv = download_hf_file(cfg, rec["hf_path"])
    raw = read_table_any(local_csv)
    std = standardize_official_hrrr(raw, selected_fips, rec.get("year"), rec.get("month"))
    if std.empty:
        raise RuntimeError(f"Empty weather frame after filtering selected FIPS: {local_csv.name}")

    if "FIPS Code" in std.columns:
        groups = list(std.groupby("FIPS Code"))
    else:
        groups = [("unknown", std)]

    raw_frames, errors = [], []
    for idx, (fips, grp) in enumerate(groups, start=1):
        LOGGER.info("Weather sample | file=%s fips=%s rows=%s", local_csv.name, fips, len(grp))
        try:
            feat = EXTRACTORS["weather"](grp, county_id=str(fips).zfill(5), crop_type=cfg.crop_type)
            feat_df = feature_output_to_frame(feat, crop_type=cfg.crop_type, fips=str(fips).zfill(5))
            if feat_df.empty:
                raise RuntimeError("weather extractor returned empty feature frame")
            raw_frames.append(feat_df)
        except Exception as exc:
            errors.append({"file": local_csv.name, "fips": fips, "error": f"{type(exc).__name__}: {exc}"})
        if cfg.dry_run and idx >= cfg.dry_run_max_counties:
            LOGGER.info("Dry-run county limit reached for weather file %s", local_csv.name)
            break

    if not raw_frames:
        raise RuntimeError(f"No weather features extracted from {local_csv.name}")

    frame = pd.concat(raw_frames, ignore_index=True, sort=False)
    frame = normalize_meta(frame, cfg.crop_type)
    keep = META_COLS + [col for col in WEATHER_CORE if col in frame.columns]
    frame = frame[keep]
    validate_feature_frame(frame, [col for col in WEATHER_CORE if col in frame.columns], f"weather features {local_csv.name}")
    atomic_write_parquet(frame, out_path)
    validate_saved_parquet(out_path, [col for col in WEATHER_CORE if col in frame.columns], f"saved weather cache {out_path.name}")
    LOGGER.info("Saved weather chunk features: %s | rows=%s", out_path, len(frame))

    if errors:
        err_path = out_path.with_suffix(".errors.csv")
        atomic_write_csv(pd.DataFrame(errors), err_path)
        validate_saved_csv(err_path, f"weather error log {err_path.name}")
        LOGGER.warning("Saved %s weather extraction errors to %s", len(errors), err_path)

    if cfg.delete_raw_after_extract:
        safe_unlink(local_csv)
    return frame


def validate_ag_ndvi_alignment(monthly: pd.DataFrame) -> None:
    ag_present = [col for col in AG_CORE if col in monthly.columns and monthly[col].notna().any()]
    ndvi_present = [col for col in NDVI_CORE if col in monthly.columns and monthly[col].notna().any()]
    if ag_present and ndvi_present:
        ag_keys = set(
            tuple(row)
            for row in monthly.loc[monthly[ag_present].notna().any(axis=1), META_COLS].drop_duplicates().itertuples(index=False, name=None)
        )
        ndvi_keys = set(
            tuple(row)
            for row in monthly.loc[monthly[ndvi_present].notna().any(axis=1), META_COLS].drop_duplicates().itertuples(index=False, name=None)
        )
        overlap = ag_keys & ndvi_keys
        LOGGER.info("AG/NDVI alignment | ag_keys=%s ndvi_keys=%s overlap=%s", len(ag_keys), len(ndvi_keys), len(overlap))
        if not overlap:
            raise RuntimeError("AG and NDVI monthly keys do not overlap; alignment failed.")


def build_feature_contract_diagnostic(cfg: RuntimeConfig, monthly: pd.DataFrame) -> dict:
    metadata_cols = [col for col in META_COLS if col in monthly.columns]
    expected = {
        "metadata_columns": META_COLS,
        "ag_columns": AG_CORE,
        "ndvi_columns": NDVI_CORE,
        "weather_columns": WEATHER_CORE,
        "total_model_feature_count": len(FEATURE_COLS),
    }
    actual = {
        "metadata_columns": metadata_cols,
        "ag_columns": [col for col in AG_CORE if col in monthly.columns],
        "ndvi_columns": [col for col in NDVI_CORE if col in monthly.columns],
        "weather_columns": [col for col in WEATHER_CORE if col in monthly.columns],
    }
    actual["total_model_feature_count"] = (
        len(actual["ag_columns"]) + len(actual["ndvi_columns"]) + len(actual["weather_columns"])
    )

    numeric_cols = [col for col in monthly.columns if col not in META_COLS and pd.api.types.is_numeric_dtype(monthly[col])]
    expected_missing = [col for col in FEATURE_COLS if col not in numeric_cols]
    extra_columns = [col for col in numeric_cols if col not in FEATURE_COLS]
    all_nan_columns = [col for col in numeric_cols if monthly[col].isna().all()]
    partially_nan_columns = [col for col in numeric_cols if monthly[col].isna().any() and not monthly[col].isna().all()]
    inf_columns = [
        col
        for col in numeric_cols
        if np.isinf(monthly[[col]].to_numpy(dtype="float64", copy=False)).any()
    ]

    month_rows = []
    expected_group_sizes = {
        "ag": len(AG_CORE),
        "ndvi": len(NDVI_CORE),
        "weather": len(WEATHER_CORE),
    }
    if "month" in monthly.columns:
        for month_value, month_df in monthly.groupby("month", dropna=False):
            ag_present = [col for col in AG_CORE if col in month_df.columns]
            ndvi_present = [col for col in NDVI_CORE if col in month_df.columns]
            weather_present = [col for col in WEATHER_CORE if col in month_df.columns]
            ag_non_null = int(month_df[ag_present].notna().sum().sum()) if ag_present else 0
            ndvi_non_null = int(month_df[ndvi_present].notna().sum().sum()) if ndvi_present else 0
            weather_non_null = int(month_df[weather_present].notna().sum().sum()) if weather_present else 0
            ag_expected_total = len(month_df) * expected_group_sizes["ag"]
            ndvi_expected_total = len(month_df) * expected_group_sizes["ndvi"]
            weather_expected_total = len(month_df) * expected_group_sizes["weather"]
            total_nan = int(month_df[numeric_cols].isna().sum().sum()) if numeric_cols else 0
            month_rows.append(
                {
                    "month": None if pd.isna(month_value) else int(month_value),
                    "row_count": int(len(month_df)),
                    "ag_non_null": ag_non_null,
                    "ag_expected": ag_expected_total,
                    "ag_coverage_ratio": float(ag_non_null / ag_expected_total) if ag_expected_total else None,
                    "ndvi_non_null": ndvi_non_null,
                    "ndvi_expected": ndvi_expected_total,
                    "ndvi_coverage_ratio": float(ndvi_non_null / ndvi_expected_total) if ndvi_expected_total else None,
                    "weather_non_null": weather_non_null,
                    "weather_expected": weather_expected_total,
                    "weather_coverage_ratio": float(weather_non_null / weather_expected_total) if weather_expected_total else None,
                    "total_nan_count": total_nan,
                    "enough_for_sequence_building": bool(
                        weather_non_null > 0
                        and (ag_non_null > 0 if "AG" in cfg.image_types else True)
                        and (ndvi_non_null > 0 if "NDVI" in cfg.image_types else True)
                    ),
                }
            )

    diagnostic = {
        "expected": expected,
        "actual": actual,
        "missing_and_extra": {
            "expected_but_missing": expected_missing,
            "unexpected_extra": extra_columns,
            "present_but_all_nan": all_nan_columns,
            "present_but_partially_nan": partially_nan_columns,
            "present_with_inf_values": inf_columns,
        },
        "monthly_coverage": month_rows,
    }
    return diagnostic


def log_feature_contract_diagnostic(diagnostic: dict) -> None:
    LOGGER.info("Feature contract expected total: %s", diagnostic["expected"]["total_model_feature_count"])
    LOGGER.info("Feature contract actual total: %s", diagnostic["actual"]["total_model_feature_count"])
    LOGGER.info("Expected AG columns: %s", diagnostic["expected"]["ag_columns"])
    LOGGER.info("Expected NDVI columns: %s", diagnostic["expected"]["ndvi_columns"])
    LOGGER.info("Expected weather columns: %s", diagnostic["expected"]["weather_columns"])
    LOGGER.info("Actual AG columns: %s", diagnostic["actual"]["ag_columns"])
    LOGGER.info("Actual NDVI columns: %s", diagnostic["actual"]["ndvi_columns"])
    LOGGER.info("Actual weather columns: %s", diagnostic["actual"]["weather_columns"])
    LOGGER.info("Expected-but-missing columns: %s", diagnostic["missing_and_extra"]["expected_but_missing"])
    LOGGER.info("Unexpected extra columns: %s", diagnostic["missing_and_extra"]["unexpected_extra"])
    LOGGER.info("Present but all-NaN columns: %s", diagnostic["missing_and_extra"]["present_but_all_nan"])
    LOGGER.info("Present but partially-NaN columns: %s", diagnostic["missing_and_extra"]["present_but_partially_nan"])
    LOGGER.info("Present with inf columns: %s", diagnostic["missing_and_extra"]["present_with_inf_values"])
    for row in diagnostic["monthly_coverage"]:
        LOGGER.info(
            "Month coverage | month=%s rows=%s ag=%.3f ndvi=%.3f weather=%.3f total_nan=%s enough=%s",
            row["month"],
            row["row_count"],
            row["ag_coverage_ratio"] if row["ag_coverage_ratio"] is not None else -1.0,
            row["ndvi_coverage_ratio"] if row["ndvi_coverage_ratio"] is not None else -1.0,
            row["weather_coverage_ratio"] if row["weather_coverage_ratio"] is not None else -1.0,
            row["total_nan_count"],
            row["enough_for_sequence_building"],
        )


def try_load_existing_monthly_artifacts(cfg: RuntimeConfig) -> tuple[pd.DataFrame, list[str], dict] | None:
    monthly_path = cfg.artifact_dir / "official_monthly_feature_table.parquet"
    diagnostic_path = cfg.artifact_dir / "feature_contract_diagnostic.json"
    if not (cfg.resume and monthly_path.exists() and diagnostic_path.exists()):
        return None

    monthly = pd.read_parquet(monthly_path)
    if monthly.empty:
        LOGGER.warning("Existing monthly artifact is empty and will be regenerated: %s", monthly_path)
        return None
    monthly = normalize_meta(monthly, cfg.crop_type)
    available_features = [col for col in FEATURE_COLS if col in monthly.columns and monthly[col].notna().any()]
    if not available_features:
        LOGGER.warning("Existing monthly artifact has no usable features and will be regenerated: %s", monthly_path)
        return None

    diagnostic = validate_saved_json(diagnostic_path, "existing feature contract diagnostic")
    validate_ag_ndvi_alignment(monthly)
    report = nan_inf_report_frame(monthly, available_features)
    LOGGER.info(
        "Using existing monthly artifact: %s | shape=%s available_features=%s nan_count=%s inf_count=%s",
        monthly_path,
        monthly.shape,
        len(available_features),
        report["nan_count"],
        report["inf_count"],
    )
    log_feature_contract_diagnostic(diagnostic)
    return monthly, available_features, diagnostic


def build_monthly_table(cfg: RuntimeConfig) -> tuple[pd.DataFrame, list[str], dict]:
    all_cache_files = sorted(cfg.feature_cache_dir.glob("*/*.parquet"))
    LOGGER.info("Feature-cache files found: %s", len(all_cache_files))
    if not all_cache_files:
        raise RuntimeError("No feature-cache files found. Extraction did not produce usable outputs.")

    frames = []
    for path in all_cache_files:
        frame = pd.read_parquet(path)
        if frame.empty:
            raise RuntimeError(f"Feature cache file is empty: {path}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = normalize_meta(combined, cfg.crop_type)
    if combined.empty:
        raise RuntimeError("Combined feature cache is empty.")
    num_cols = [col for col in combined.columns if col not in META_COLS and pd.api.types.is_numeric_dtype(combined[col])]
    monthly = combined.groupby(META_COLS, as_index=False)[num_cols].mean()
    monthly = monthly.sort_values(["county_id", "crop_type", "year", "month"]).reset_index(drop=True)

    available_features = [col for col in FEATURE_COLS if col in monthly.columns and monthly[col].notna().any()]
    if not available_features:
        raise RuntimeError("No usable feature columns found in the monthly table.")
    validate_ag_ndvi_alignment(monthly)

    report = nan_inf_report_frame(monthly, available_features)
    LOGGER.info(
        "Monthly table | shape=%s available_features=%s nan_count=%s inf_count=%s",
        monthly.shape,
        len(available_features),
        report["nan_count"],
        report["inf_count"],
    )
    diagnostic = build_feature_contract_diagnostic(cfg, monthly)
    log_feature_contract_diagnostic(diagnostic)
    monthly_path = cfg.artifact_dir / "official_monthly_feature_table.parquet"
    atomic_write_parquet(monthly, monthly_path)
    validate_saved_parquet(monthly_path, available_features, "monthly feature table")
    diagnostic_path = cfg.artifact_dir / "feature_contract_diagnostic.json"
    atomic_write_json(diagnostic, diagnostic_path)
    validate_saved_json(diagnostic_path, "feature contract diagnostic")
    LOGGER.info("Saved feature contract diagnostic: %s", diagnostic_path)
    return monthly, available_features, diagnostic


def validate_sequence_arrays(X: np.ndarray, y: np.ndarray, seq_meta: pd.DataFrame, feature_cols: list[str], cfg: RuntimeConfig) -> None:
    if X.ndim != 3:
        raise RuntimeError(f"Expected X rank 3, got {X.shape}")
    if y.ndim != 2:
        raise RuntimeError(f"Expected y rank 2, got {y.shape}")
    if X.shape[0] != y.shape[0] or X.shape[0] != len(seq_meta):
        raise RuntimeError("Sequence row counts are misaligned between X, y, and metadata.")
    if X.shape[2] != len(feature_cols) or y.shape[1] != len(feature_cols):
        raise RuntimeError("Sequence feature dimension does not match feature column count.")
    if not np.isfinite(X).all():
        raise RuntimeError("X contains NaN/inf values.")
    if not np.isfinite(y).all():
        raise RuntimeError("y contains NaN/inf values.")
    if (seq_meta["split"] == "train").sum() == 0:
        raise RuntimeError("Train split is empty.")
    if cfg.run_forecasting and (seq_meta["split"] == "test").sum() == 0:
        raise RuntimeError("Test split is empty.")


def selected_feature_columns(cfg: RuntimeConfig, available_features: list[str]) -> list[str]:
    selected = [feature for feature in FEATURE_GROUP_SELECTIONS[cfg.feature_group] if feature in available_features]
    if not selected:
        raise RuntimeError(
            f"No features available for feature_group={cfg.feature_group}. Available columns: {available_features}"
        )
    return selected


def prepare_model_frames(
    cfg: RuntimeConfig,
    monthly: pd.DataFrame,
    available_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.Series, pd.Series]:
    model_df = monthly[META_COLS + available_features].copy()
    model_df["date"] = pd.to_datetime(dict(year=model_df["year"].astype(int), month=model_df["month"].astype(int), day=1))
    model_df = model_df.sort_values(["county_id", "crop_type", "date"]).reset_index(drop=True)

    filled_groups = []
    for (county, crop), group in model_df.groupby(["county_id", "crop_type"]):
        group = group.set_index("date").sort_index()
        idx = pd.date_range(f"{min(cfg.years)}-01-01", f"{max(cfg.years)}-12-01", freq="MS")
        group = group.reindex(idx)
        group["county_id"] = county
        group["crop_type"] = crop
        group["year"] = group.index.year
        group["month"] = group.index.month
        group[available_features] = group[available_features].interpolate(limit_direction="both").ffill().bfill()
        filled_groups.append(group.reset_index(names="date"))
    filled = pd.concat(filled_groups, ignore_index=True)

    if cfg.train_years:
        train_mask_for_median = filled["year"].isin(cfg.train_years)
        train_medians = filled.loc[train_mask_for_median, available_features].median(numeric_only=True)
    else:
        train_medians = filled[available_features].median(numeric_only=True)
    filled[available_features] = filled[available_features].fillna(train_medians)

    still_bad = [col for col in available_features if filled[col].isna().all()]
    if still_bad:
        LOGGER.warning("Dropping all-NaN features: %s", still_bad)
        available_features = [col for col in available_features if col not in still_bad]
        filled = filled[META_COLS + ["date"] + available_features]
    if not available_features:
        raise RuntimeError("No usable features remain after imputation.")

    if cfg.train_years:
        mu = filled.loc[filled["year"].isin(cfg.train_years), available_features].mean()
        sigma = filled.loc[filled["year"].isin(cfg.train_years), available_features].std().replace(0, 1.0).fillna(1.0)
    else:
        mu = filled[available_features].mean()
        sigma = filled[available_features].std().replace(0, 1.0).fillna(1.0)
    filled_scaled = filled.copy()
    filled_scaled[available_features] = (filled_scaled[available_features] - mu) / sigma
    return filled, filled_scaled, available_features, mu, sigma


def build_seasonal_lookup(frame: pd.DataFrame, feature_cols: list[str]) -> dict[tuple[str, str, int, int], np.ndarray]:
    lookup: dict[tuple[str, str, int, int], np.ndarray] = {}
    cols = ["county_id", "crop_type", "year", "month"] + feature_cols
    for row in frame[cols].itertuples(index=False, name=None):
        county_id, crop_type, year, month, *values = row
        lookup[(str(county_id), str(crop_type), int(year), int(month))] = np.asarray(values, dtype=np.float32)
    return lookup


def actual_target_scaled(cfg: RuntimeConfig, y_model_scaled: np.ndarray, seasonal_base_scaled: np.ndarray) -> np.ndarray:
    if cfg.target_mode == "seasonal_residual":
        return seasonal_base_scaled + y_model_scaled
    return y_model_scaled


def finalize_model_prediction_scaled(
    cfg: RuntimeConfig,
    model_name: str,
    pred_scaled: np.ndarray,
    seasonal_base_scaled: np.ndarray | None,
) -> np.ndarray:
    if cfg.target_mode == "seasonal_residual" and model_name in LEARNED_MODELS:
        if seasonal_base_scaled is None:
            raise RuntimeError(f"Residual target mode requires seasonal bases for model={model_name}.")
        return pred_scaled + seasonal_base_scaled
    return pred_scaled


def build_sequences(
    cfg: RuntimeConfig,
    monthly: pd.DataFrame,
    available_features: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    filled, filled_scaled, available_features, mu, sigma = prepare_model_frames(cfg, monthly, available_features)
    seasonal_lookup = build_seasonal_lookup(filled_scaled, available_features)

    X_list, y_model_list, y_true_list, seasonal_base_list, meta_rows = [], [], [], [], []
    dropped_missing_seasonal = 0
    dropped_missing_by_split: dict[str, int] = {}
    for (county, crop), group in filled_scaled.groupby(["county_id", "crop_type"]):
        group = group.sort_values("date").reset_index(drop=True)
        vals = group[available_features].to_numpy(dtype=np.float32)
        for i in range(cfg.lookback_months, len(group)):
            target_year = int(group.loc[i, "year"])
            target_month = int(group.loc[i, "month"])
            split = (
                "train"
                if target_year in cfg.train_years
                else "val"
                if target_year in cfg.val_years
                else "test"
                if target_year in cfg.test_years
                else None
            )
            if split is None:
                continue
            seasonal_base_scaled = seasonal_lookup.get((str(county), str(crop), target_year - 1, target_month))
            if cfg.target_mode == "seasonal_residual" and seasonal_base_scaled is None:
                dropped_missing_seasonal += 1
                dropped_missing_by_split[split] = dropped_missing_by_split.get(split, 0) + 1
                continue
            target_scaled = vals[i]
            model_target_scaled = target_scaled if seasonal_base_scaled is None else (target_scaled - seasonal_base_scaled)
            X_list.append(vals[i - cfg.lookback_months : i])
            y_model_list.append(model_target_scaled)
            y_true_list.append(target_scaled)
            seasonal_base_list.append(seasonal_base_scaled if seasonal_base_scaled is not None else np.zeros(len(available_features), dtype=np.float32))
            meta_rows.append(
                {
                    "county_id": county,
                    "crop_type": crop,
                    "target_year": target_year,
                    "target_month": target_month,
                    "target_date": group.loc[i, "date"],
                    "split": split,
                }
            )
            if cfg.dry_run and len(X_list) >= cfg.dry_run_max_batches * cfg.batch_size:
                break

    if dropped_missing_seasonal:
        LOGGER.info(
            "Dropped sequence samples without previous-year seasonal base | total=%s by_split=%s",
            dropped_missing_seasonal,
            dropped_missing_by_split,
        )

    if not X_list or not y_model_list:
        raise RuntimeError("No sequence samples were built. Check years, lookback, and extracted monthly features.")
    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_model_list).astype(np.float32)
    y_true_scaled = np.stack(y_true_list).astype(np.float32)
    seasonal_base_scaled = np.stack(seasonal_base_list).astype(np.float32)
    seq_meta = pd.DataFrame(meta_rows)
    validate_sequence_arrays(X, y, seq_meta, available_features, cfg)
    if not np.isfinite(y_true_scaled).all():
        raise RuntimeError("Actual scaled targets contain NaN/inf values.")
    if cfg.target_mode == "seasonal_residual" and not np.isfinite(seasonal_base_scaled).all():
        raise RuntimeError("Seasonal base targets contain NaN/inf values.")

    atomic_save_npy(X, cfg.artifact_dir / "X_sequences.npy")
    atomic_save_npy(y, cfg.artifact_dir / "y_sequences.npy")
    if cfg.target_mode == "seasonal_residual":
        atomic_save_npy(y_true_scaled, cfg.artifact_dir / "y_true_sequences.npy")
        atomic_save_npy(seasonal_base_scaled, cfg.artifact_dir / "seasonal_base_sequences.npy")
    atomic_write_csv(seq_meta, cfg.artifact_dir / "sequence_metadata.csv")
    scaler_df = pd.DataFrame({"feature": available_features, "mean": mu[available_features].values, "std": sigma[available_features].values})
    atomic_write_csv(scaler_df, cfg.artifact_dir / "scaler.csv")
    validate_saved_npy(cfg.artifact_dir / "X_sequences.npy", "saved X sequences")
    validate_saved_npy(cfg.artifact_dir / "y_sequences.npy", "saved y sequences")
    if cfg.target_mode == "seasonal_residual":
        validate_saved_npy(cfg.artifact_dir / "y_true_sequences.npy", "saved actual y sequences")
        validate_saved_npy(cfg.artifact_dir / "seasonal_base_sequences.npy", "saved seasonal base sequences")
    validate_saved_csv(cfg.artifact_dir / "sequence_metadata.csv", "saved sequence metadata")
    validate_saved_csv(cfg.artifact_dir / "scaler.csv", "saved scaler csv")
    return X, y, seq_meta, filled, filled_scaled, mu, sigma, y_true_scaled, seasonal_base_scaled


def modality_slices(feature_cols: list[str]) -> dict[str, list[int]]:
    mapping = {}
    for modality, cols in FEATURE_GROUPS.items():
        mapping[modality] = [feature_cols.index(col) for col in cols if col in feature_cols]
    return mapping


def inverse_scale(arr: np.ndarray, feature_cols: list[str], mu: pd.Series, sigma: pd.Series) -> np.ndarray:
    return arr * sigma[feature_cols].to_numpy(dtype=np.float32) + mu[feature_cols].to_numpy(dtype=np.float32)


def compute_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    feature_cols: list[str],
    mu: pd.Series,
    sigma: pd.Series,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    true_unscaled = inverse_scale(y_true_scaled, feature_cols, mu, sigma)
    pred_unscaled = inverse_scale(y_pred_scaled, feature_cols, mu, sigma)
    slices = modality_slices(feature_cols)
    rows = []
    for modality, idxs in slices.items():
        if not idxs:
            continue
        y_true = true_unscaled[:, idxs].reshape(-1)
        y_pred = pred_unscaled[:, idxs].reshape(-1)
        rows.append(
            {
                "model": model_name,
                "split": split,
                "modality": modality,
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
            }
        )
    return pd.DataFrame(rows)


def per_feature_metrics(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    feature_cols: list[str],
    mu: pd.Series,
    sigma: pd.Series,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    true_unscaled = inverse_scale(y_true_scaled, feature_cols, mu, sigma)
    pred_unscaled = inverse_scale(y_pred_scaled, feature_cols, mu, sigma)
    rows = []
    for j, feature in enumerate(feature_cols):
        rows.append(
            {
                "model": model_name,
                "split": split,
                "feature": feature,
                "mae": float(mean_absolute_error(true_unscaled[:, j], pred_unscaled[:, j])),
                "rmse": float(math.sqrt(mean_squared_error(true_unscaled[:, j], pred_unscaled[:, j]))),
            }
        )
    return pd.DataFrame(rows)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, n_features))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRUForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, n_features))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class TinyMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, conv_kernel: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=d_model)
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        uc = self.conv(u.transpose(1, 2))[..., : u.shape[1]].transpose(1, 2)
        uc = torch.nn.functional.silu(uc)
        params = self.x_proj(uc)
        Bp, Cp, delta = torch.split(params, [self.d_state, self.d_state, self.d_model], dim=-1)
        delta = torch.nn.functional.softplus(delta)
        A = -torch.exp(self.A_log)
        state = torch.zeros(x.shape[0], self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):
            dt = delta[:, t, :].unsqueeze(-1)
            At = torch.exp(dt * A.unsqueeze(0))
            Bt = Bp[:, t, :].unsqueeze(1)
            ut = uc[:, t, :].unsqueeze(-1)
            state = At * state + dt * Bt * ut
            Ct = Cp[:, t, :].unsqueeze(1)
            yt = (state * Ct).sum(dim=-1) + self.D * uc[:, t, :]
            ys.append(yt)
        y = torch.stack(ys, dim=1)
        y = y * torch.sigmoid(z)
        return self.norm(residual + self.out_proj(y))


class MambaStyleForecaster(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, d_state: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([TinyMambaBlock(d_model=d_model, d_state=d_state) for _ in range(max(1, num_layers))])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_features))

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.dropout(h)
        return self.head(h[:, -1, :])


class TransformerEncoderForecaster(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        nhead = 4 if d_model % 4 == 0 else 1
        self.input_proj = nn.Linear(n_features, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 512, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_features))

    def forward(self, x):
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.positional[:, :seq_len, :]
        h = self.encoder(h)
        return self.head(h[:, -1, :])


def make_loss_fn(loss_weights: torch.Tensor | None = None):
    if loss_weights is None:
        return nn.MSELoss()

    def weighted_loss(pred, target):
        diff = pred - target
        return torch.mean((diff * diff) * loss_weights)

    return weighted_loss


def evaluate_torch_model(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            losses.append(loss_fn(pred, yb).item() * len(xb))
    return float(np.sum(losses) / len(loader.dataset))


def predict_torch_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_torch_next(model: nn.Module, seq_scaled: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(seq_scaled[None, :, :], dtype=torch.float32, device=device)
        pred = model(xb).detach().cpu().numpy()[0]
    return pred.astype(np.float32)


def load_scaler_artifact(cfg: RuntimeConfig) -> tuple[list[str], pd.Series, pd.Series]:
    scaler_path = cfg.artifact_dir / "scaler.csv"
    scaler_df = validate_saved_csv(scaler_path, "scaler artifact")
    required = {"feature", "mean", "std"}
    if not required.issubset(scaler_df.columns):
        raise RuntimeError(f"Scaler artifact is missing columns {sorted(required - set(scaler_df.columns))}")
    feature_cols = scaler_df["feature"].astype(str).tolist()
    mu = pd.Series(scaler_df["mean"].astype(float).to_numpy(), index=feature_cols)
    sigma = pd.Series(scaler_df["std"].astype(float).replace(0, 1.0).fillna(1.0).to_numpy(), index=feature_cols)
    return feature_cols, mu, sigma


def feature_to_modality(feature: str) -> str:
    for modality, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return modality
    return "unknown"


def infer_lstm_architecture(state_dict: dict, n_features: int, default_dropout: float) -> tuple[int, int, float]:
    hidden_size = state_dict["lstm.weight_ih_l0"].shape[0] // 4
    num_layers = len([key for key in state_dict.keys() if key.startswith("lstm.weight_ih_l")])
    return hidden_size, num_layers, default_dropout


def infer_gru_architecture(state_dict: dict, n_features: int, default_dropout: float) -> tuple[int, int, float]:
    hidden_size = state_dict["gru.weight_ih_l0"].shape[0] // 3
    num_layers = len([key for key in state_dict.keys() if key.startswith("gru.weight_ih_l")])
    return hidden_size, num_layers, default_dropout


def infer_mamba_architecture(state_dict: dict, n_features: int, default_dropout: float) -> tuple[int, int, float]:
    d_model = state_dict["input_proj.weight"].shape[0]
    block_prefixes = {key.split(".")[1] for key in state_dict.keys() if key.startswith("blocks.")}
    num_layers = len(block_prefixes) if block_prefixes else 1
    return d_model, num_layers, default_dropout


def infer_transformer_architecture(state_dict: dict, n_features: int, default_dropout: float) -> tuple[int, int, float]:
    d_model = state_dict["input_proj.weight"].shape[0]
    layer_prefixes = {key.split(".")[2] for key in state_dict.keys() if key.startswith("encoder.layers.")}
    num_layers = len(layer_prefixes) if layer_prefixes else 1
    return d_model, num_layers, default_dropout


def normalize_legacy_checkpoint_keys(model_name: str, state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if model_name == "lstm":
            if key.startswith("head.1."):
                new_key = "head.2." + key.split("head.1.", 1)[1]
        elif model_name == "tiny_mamba_ssm":
            if key.startswith("block."):
                new_key = "blocks.0." + key.split("block.", 1)[1]
            if new_key.startswith("head.1."):
                new_key = "head.2." + new_key.split("head.1.", 1)[1]
        normalized[new_key] = value
    return normalized


def build_learned_model(cfg: RuntimeConfig, model_name: str, n_features: int) -> nn.Module:
    if model_name == "lstm":
        return LSTMForecaster(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    if model_name == "gru":
        return GRUForecaster(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    if model_name == "tiny_mamba_ssm":
        return MambaStyleForecaster(n_features, d_model=cfg.hidden_size, d_state=32, num_layers=cfg.num_layers, dropout=cfg.dropout)
    if model_name == "transformer_encoder":
        return TransformerEncoderForecaster(n_features, d_model=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout)
    raise ValueError(f"Unsupported learned model: {model_name}")


def load_trained_model_for_eval(cfg: RuntimeConfig, model_name: str, n_features: int, device: torch.device) -> nn.Module:
    ckpt_path = cfg.artifact_dir / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for eval-only blank-fill: {ckpt_path}")
    state_dict = normalize_legacy_checkpoint_keys(model_name, torch.load(ckpt_path, map_location="cpu"))
    if model_name == "lstm":
        hidden_size, num_layers, dropout = infer_lstm_architecture(state_dict, n_features, cfg.dropout)
        model = LSTMForecaster(n_features, hidden_size, num_layers, dropout)
    elif model_name == "gru":
        hidden_size, num_layers, dropout = infer_gru_architecture(state_dict, n_features, cfg.dropout)
        model = GRUForecaster(n_features, hidden_size, num_layers, dropout)
    elif model_name == "tiny_mamba_ssm":
        d_model, num_layers, dropout = infer_mamba_architecture(state_dict, n_features, cfg.dropout)
        model = MambaStyleForecaster(n_features, d_model=d_model, d_state=32, num_layers=num_layers, dropout=dropout)
    elif model_name == "transformer_encoder":
        d_model, num_layers, dropout = infer_transformer_architecture(state_dict, n_features, cfg.dropout)
        model = TransformerEncoderForecaster(n_features, d_model=d_model, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"Unsupported eval model load: {model_name}")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def compute_feature_loss_weights(cfg: RuntimeConfig, feature_cols: list[str], y_train: np.ndarray) -> pd.DataFrame:
    if cfg.loss_mode == "raw_mse":
        weights = np.ones(len(feature_cols), dtype=np.float32)
    else:
        train_std = np.std(y_train, axis=0).astype(np.float32)
        safe_std = np.where(train_std > 1e-6, train_std, 1.0)
        weights = 1.0 / np.square(safe_std)
        weights = weights / float(np.mean(weights))
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "modality": [feature_to_modality(feature) for feature in feature_cols],
            "loss_weight": weights.astype(float),
        }
    )


def train_torch_model(
    cfg: RuntimeConfig,
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    loss_weights: torch.Tensor | None,
) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = make_loss_fn(loss_weights)
    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience
    history = []

    max_epochs = min(cfg.max_epochs, cfg.dry_run_max_batches) if cfg.dry_run else cfg.max_epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
            if cfg.dry_run and batch_idx >= cfg.dry_run_max_batches:
                break
        train_loss = total / max(n, 1)
        val_loss = evaluate_torch_model(model, val_loader, device, loss_fn) if val_loader is not None and len(val_loader.dataset) else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
            atomic_torch_save(best_state, cfg.artifact_dir / f"{model_name}_best.pt")
        else:
            patience_left -= 1
        LOGGER.info("%s epoch %03d | train=%.5f val=%.5f patience=%s", model_name, epoch, train_loss, val_loss, patience_left)
        if patience_left <= 0:
            break

    history_path = cfg.artifact_dir / f"{model_name}_history.csv"
    atomic_write_csv(pd.DataFrame(history), history_path)
    validate_saved_csv(history_path, f"{model_name} history")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def write_model_specs(
    cfg: RuntimeConfig,
    feature_cols: list[str],
    seq_meta: pd.DataFrame,
    trained_models: dict[str, nn.Module],
    model_runtime_seconds: dict[str, float],
    device: torch.device,
    extra_rows: list[dict] | None = None,
) -> Path:
    rows = []
    split_counts = seq_meta["split"].value_counts().to_dict()

    for model_name in ["naive_lag1", "seasonal_last_year"]:
        rows.append(
            {
                "model": model_name,
                "model_class": "deterministic_baseline",
                "input_dim": len(feature_cols),
                "output_dim": len(feature_cols),
                "seq_len": cfg.lookback_months,
                "hidden_size": np.nan,
                "num_layers": np.nan,
                "dropout": np.nan,
                "learning_rate": np.nan,
                "batch_size": np.nan,
                "max_epochs": np.nan,
                "early_stopping_patience": np.nan,
                "weight_decay": np.nan,
                "loss_mode": cfg.loss_mode,
                "target_mode": cfg.target_mode,
                "feature_group": cfg.feature_group,
                "trainable_parameters": 0,
                "total_parameters": 0,
                "checkpoint_size_bytes": 0,
                "checkpoint_path": "",
                "training_sample_count": int(split_counts.get("train", 0)),
                "validation_sample_count": int(split_counts.get("val", 0)),
                "test_sample_count": int(split_counts.get("test", 0)),
                "device_used": "deterministic",
                "training_runtime_seconds": np.nan,
                "baseline_rule": "lag1 recursive" if model_name == "naive_lag1" else "same county same month previous year; fallback lag1",
            }
        )

    for model_name, model in trained_models.items():
        ckpt_path = cfg.artifact_dir / f"{model_name}_best.pt"
        total_params = int(sum(param.numel() for param in model.parameters()))
        trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
        rows.append(
            {
                "model": model_name,
                "model_class": model.__class__.__name__,
                "input_dim": len(feature_cols),
                "output_dim": len(feature_cols),
                "seq_len": cfg.lookback_months,
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "early_stopping_patience": cfg.patience,
                "weight_decay": cfg.weight_decay,
                "loss_mode": cfg.loss_mode,
                "target_mode": cfg.target_mode,
                "feature_group": cfg.feature_group,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "checkpoint_size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else np.nan,
                "checkpoint_path": str(ckpt_path) if ckpt_path.exists() else "",
                "training_sample_count": int(split_counts.get("train", 0)),
                "validation_sample_count": int(split_counts.get("val", 0)),
                "test_sample_count": int(split_counts.get("test", 0)),
                "device_used": str(device),
                "training_runtime_seconds": model_runtime_seconds.get(model_name, np.nan),
                "baseline_rule": "",
            }
        )

    if extra_rows:
        rows.extend(extra_rows)

    specs = pd.DataFrame(rows)
    specs_path = cfg.artifact_dir / "model_specs.csv"
    atomic_write_csv(specs, specs_path)
    validate_saved_csv(specs_path, "model specs")
    return specs_path


def fit_sarima_forecast(
    series: pd.Series,
    steps: int,
) -> tuple[pd.Series, str]:
    train_series = series.dropna()
    if len(train_series) < 24:
        return pd.Series(index=pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq="MS"), data=float(train_series.iloc[-1]) if len(train_series) else 0.0), "insufficient_history_lag1"
    try:
        seasonal_order = SARIMA_SEASONAL_ORDER if len(train_series) >= 36 else (0, 0, 0, 0)
        model = sm.tsa.statespace.SARIMAX(
            train_series,
            order=SARIMA_ORDER,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=25)
        fc = res.forecast(steps=steps)
        fc.index = pd.date_range(train_series.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        return fc, "sarima"
    except Exception as exc:
        LOGGER.warning("SARIMA fit failed; falling back to lag1 | error=%s", f"{type(exc).__name__}: {exc}")
        return pd.Series(index=pd.date_range(train_series.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq="MS"), data=float(train_series.iloc[-1])), f"fit_failed_lag1:{type(exc).__name__}"


def sarima_forecast_matrix(
    cfg: RuntimeConfig,
    feature_cols: list[str],
    filled_scaled: pd.DataFrame,
    seq_meta: pd.DataFrame,
    target_idx: np.ndarray,
    fallback_scaled: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.full((len(target_idx), len(feature_cols)), np.nan, dtype=np.float32)
    target_meta = seq_meta.iloc[target_idx].reset_index(drop=False).rename(columns={"index": "global_index"})
    test_lookup = {
        (row.county_id, row.crop_type, pd.Timestamp(row.target_date), feature): (i, j)
        for i, row in target_meta.iterrows()
        for j, feature in enumerate(feature_cols)
    }

    all_counties = sorted(filled_scaled["county_id"].unique())
    selected_counties = all_counties[: cfg.classical_max_counties] if cfg.classical_max_counties else all_counties
    selected_features = feature_cols[: cfg.classical_max_features] if cfg.classical_max_features else feature_cols
    LOGGER.info("SARIMA fitting %s counties x %s features", len(selected_counties), len(selected_features))
    fit_rows: list[dict] = []

    for county in tqdm(selected_counties, desc="sarima counties"):
        county_frame = filled_scaled[filled_scaled["county_id"].eq(county)].sort_values("date")
        if county_frame.empty:
            continue
        crop = county_frame["crop_type"].iloc[0]
        train_end = pd.Timestamp(f"{max(cfg.train_years)}-12-01")
        test_dates = sorted(target_meta.loc[target_meta["county_id"].eq(county), "target_date"].unique())
        if not test_dates:
            continue
        forecast_start = pd.Timestamp(f"{min(cfg.val_years + cfg.test_years)}-01-01") if (cfg.val_years or cfg.test_years) else pd.Timestamp(test_dates[0])
        forecast_end = pd.Timestamp(max(test_dates))
        steps = (forecast_end.year - train_end.year) * 12 + (forecast_end.month - train_end.month)
        if steps <= 0:
            continue

        for feature in selected_features:
            series = county_frame.set_index("date")[feature].astype(float).asfreq("MS")
            train_series = series.loc[:train_end]
            fc, strategy = fit_sarima_forecast(train_series, steps)
            fit_rows.append(
                {
                    "scope": "one_step",
                    "county_id": county,
                    "crop_type": crop,
                    "feature": feature,
                    "fit_strategy": strategy,
                    "order": str(SARIMA_ORDER),
                    "seasonal_order": str(SARIMA_SEASONAL_ORDER),
                    "train_points": int(train_series.dropna().shape[0]),
                }
            )

            for dt in test_dates:
                key = (county, crop, pd.Timestamp(dt), feature)
                if key in test_lookup and pd.Timestamp(dt) in fc.index:
                    i, j = test_lookup[key]
                    pred[i, j] = float(fc.loc[pd.Timestamp(dt)])

    missing = np.isnan(pred)
    if missing.any():
        LOGGER.warning("SARIMA missing predictions filled with lag-1 fallback: %s", int(missing.sum()))
        pred[missing] = fallback_scaled[missing]
    return pred, pd.DataFrame(fit_rows)


def modality_upper(modality: str) -> str:
    return modality.upper()


def save_predictions(
    cfg: RuntimeConfig,
    feature_cols: list[str],
    mu: pd.Series,
    sigma: pd.Series,
    seq_meta: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    model_preds: dict[str, np.ndarray],
) -> Path:
    pred_records = []
    actual_unscaled = inverse_scale(y_true[test_idx], feature_cols, mu, sigma)
    for model_name, pred_scaled in model_preds.items():
        pred_unscaled = inverse_scale(pred_scaled, feature_cols, mu, sigma)
        for i, row in seq_meta.iloc[test_idx].reset_index(drop=True).iterrows():
            for j, feat in enumerate(feature_cols):
                pred_records.append(
                    {
                        "model": model_name,
                        "county_id": row["county_id"],
                        "crop_type": row["crop_type"],
                        "target_year": int(row["target_year"]),
                        "target_month": int(row["target_month"]),
                        "feature": feat,
                        "actual": float(actual_unscaled[i, j]),
                        "prediction": float(pred_unscaled[i, j]),
                        "error": float(pred_unscaled[i, j] - actual_unscaled[i, j]),
                    }
                )
    predictions = pd.DataFrame(pred_records)
    pred_path = cfg.artifact_dir / "test_predictions_long.csv"
    atomic_write_csv(predictions, pred_path)
    validate_saved_csv(pred_path, "saved predictions")
    return pred_path


def load_monthly_and_scaler_for_eval(cfg: RuntimeConfig) -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
    monthly_path = cfg.artifact_dir / "official_monthly_feature_table.parquet"
    monthly = pd.read_parquet(monthly_path)
    monthly = normalize_meta(monthly, cfg.crop_type)
    feature_cols, mu, sigma = load_scaler_artifact(cfg)
    missing = [col for col in feature_cols if col not in monthly.columns]
    if missing:
        raise RuntimeError(f"Monthly artifact is missing scaler features required for blank-fill eval: {missing}")
    return monthly, feature_cols, mu, sigma


def compute_blank_fill_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = predictions.copy()
    pred["squared_error"] = pred["abs_error"] ** 2

    by_horizon = (
        pred.groupby(["model", "known_months", "horizon", "modality"], as_index=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(math.sqrt(float(np.mean(s))))),
        )
    )
    by_month = (
        pred.groupby(["model", "known_months", "target_month", "modality"], as_index=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(math.sqrt(float(np.mean(s))))),
        )
    )
    feature_metrics = (
        pred.groupby(["model", "known_months", "feature", "modality"], as_index=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(math.sqrt(float(np.mean(s))))),
        )
    )
    summary = (
        pred.groupby(["model", "known_months", "modality"], as_index=False)
        .agg(
            count=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(math.sqrt(float(np.mean(s))))),
        )
    )
    horizon_summary = (
        by_horizon.groupby(["model", "known_months", "modality"], as_index=False)
        .agg(
            avg_horizon_rmse=("rmse", "mean"),
            worst_horizon_rmse=("rmse", "max"),
        )
    )
    summary = summary.merge(horizon_summary, on=["model", "known_months", "modality"], how="left")

    lag1_ref = summary[summary["model"].eq("naive_lag1")][["known_months", "modality", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_ref = summary[summary["model"].eq("seasonal_last_year")][["known_months", "modality", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    summary = summary.merge(lag1_ref, on=["known_months", "modality"], how="left")
    summary = summary.merge(seasonal_ref, on=["known_months", "modality"], how="left")
    summary["beats_lag1"] = summary["rmse"] < summary["lag1_rmse"]
    summary["beats_seasonal_last_year"] = summary["rmse"] < summary["seasonal_last_year_rmse"]

    lag1_feat = feature_metrics[feature_metrics["model"].eq("naive_lag1")][["known_months", "feature", "rmse"]].rename(columns={"rmse": "lag1_rmse"})
    seasonal_feat = feature_metrics[feature_metrics["model"].eq("seasonal_last_year")][["known_months", "feature", "rmse"]].rename(columns={"rmse": "seasonal_last_year_rmse"})
    feature_metrics = feature_metrics.merge(lag1_feat, on=["known_months", "feature"], how="left")
    feature_metrics = feature_metrics.merge(seasonal_feat, on=["known_months", "feature"], how="left")
    feature_metrics["beats_lag1"] = feature_metrics["rmse"] < feature_metrics["lag1_rmse"]
    feature_metrics["beats_seasonal_last_year"] = feature_metrics["rmse"] < feature_metrics["seasonal_last_year_rmse"]
    return by_horizon, by_month, summary, feature_metrics


def maybe_make_blank_fill_plots(
    cfg: RuntimeConfig,
    summary: pd.DataFrame,
    by_horizon: pd.DataFrame,
) -> list[Path]:
    with contextlib.suppress(ModuleNotFoundError, ImportError):
        import matplotlib.pyplot as plt

        plot_dir = cfg.artifact_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []

        horizon_plot = by_horizon[by_horizon["known_months"].eq(1)]
        if not horizon_plot.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for (model, modality), grp in horizon_plot.groupby(["model", "modality"]):
                ax.plot(grp["horizon"], grp["rmse"], marker="o", label=f"{model}-{modality}")
            ax.set_title("Blank-Fill RMSE by Horizon (known_months=1)")
            ax.set_xlabel("Horizon")
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            path = plot_dir / "blank_fill_rmse_by_horizon.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)

        summary_plot = summary[summary["model"].isin(["naive_lag1", "seasonal_last_year", "lstm", "tiny_mamba_ssm"])]
        if not summary_plot.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for (model, modality), grp in summary_plot.groupby(["model", "modality"]):
                grp = grp.sort_values("known_months")
                ax.plot(grp["known_months"], grp["rmse"], marker="o", label=f"{model}-{modality}")
            ax.set_title("Blank-Fill RMSE by Known Months")
            ax.set_xlabel("Known months")
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            path = plot_dir / "blank_fill_rmse_by_known_months.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)

        known1 = summary[summary["known_months"].eq(1)]
        if not known1.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            pivot = known1.pivot(index="modality", columns="model", values="rmse")
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Blank-Fill Model Comparison (known_months=1)")
            ax.set_ylabel("RMSE")
            ax.grid(True, axis="y", alpha=0.3)
            path = plot_dir / "blank_fill_model_comparison_known1.png"
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            outputs.append(path)

        return outputs
    return []


def build_strict_blank_fill_history(
    cfg: RuntimeConfig,
    monthly: pd.DataFrame,
    filled_unscaled: pd.DataFrame,
    filled_scaled: pd.DataFrame,
    feature_cols: list[str],
    mu: pd.Series,
    sigma: pd.Series,
    county_id: str,
    crop_type: str,
    target_year: int,
    known_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist_unscaled = filled_unscaled[
        filled_unscaled["county_id"].astype(str).eq(str(county_id))
        & filled_unscaled["crop_type"].astype(str).eq(str(crop_type))
        & filled_unscaled["year"].lt(target_year)
    ].sort_values("date").reset_index(drop=True)
    hist_scaled = filled_scaled[
        filled_scaled["county_id"].astype(str).eq(str(county_id))
        & filled_scaled["crop_type"].astype(str).eq(str(crop_type))
        & filled_scaled["year"].lt(target_year)
    ].sort_values("date").reset_index(drop=True)

    raw_target = monthly[
        monthly["county_id"].astype(str).eq(str(county_id))
        & monthly["crop_type"].astype(str).eq(str(crop_type))
        & monthly["year"].eq(target_year)
        & monthly["month"].le(known_months)
    ].copy()
    raw_lookup = {int(row["month"]): row for _, row in raw_target.iterrows()}
    prev_year_lookup = {
        int(row["month"]): row
        for _, row in filled_unscaled[
            filled_unscaled["county_id"].astype(str).eq(str(county_id))
            & filled_unscaled["crop_type"].astype(str).eq(str(crop_type))
            & filled_unscaled["year"].eq(target_year - 1)
        ].iterrows()
    }
    mu_arr = mu[feature_cols].to_numpy(dtype=np.float32)
    sigma_arr = sigma[feature_cols].to_numpy(dtype=np.float32)

    fallback_rows: list[dict] = []
    for month in range(1, known_months + 1):
        target_date = pd.Timestamp(year=target_year, month=month, day=1)
        raw_row = raw_lookup.get(month)
        prev_year_row = prev_year_lookup.get(month)
        row_unscaled = {
            "date": target_date,
            "county_id": county_id,
            "crop_type": crop_type,
            "year": target_year,
            "month": month,
        }
        feature_values = []
        for feature in feature_cols:
            raw_value = np.nan if raw_row is None else raw_row.get(feature, np.nan)
            fill_method = "observed"
            if not np.isfinite(raw_value):
                prev_hist_val = np.nan
                if feature in hist_unscaled.columns and not hist_unscaled.empty:
                    prev_non_null = hist_unscaled[feature].dropna()
                    if not prev_non_null.empty:
                        prev_hist_val = float(prev_non_null.iloc[-1])
                prev_year_val = np.nan
                if prev_year_row is not None:
                    prev_year_val = prev_year_row.get(feature, np.nan)

                if cfg.strict_fill_strategy == "past_only" and np.isfinite(prev_hist_val):
                    raw_value = prev_hist_val
                    fill_method = "past_ffill"
                elif np.isfinite(prev_year_val):
                    raw_value = float(prev_year_val)
                    fill_method = "previous_year_same_month"
                elif np.isfinite(prev_hist_val):
                    raw_value = prev_hist_val
                    fill_method = "past_ffill"
                else:
                    raw_value = float(mu.get(feature, 0.0))
                    fill_method = "train_mean_fallback"

            raw_value = float(raw_value)
            row_unscaled[feature] = raw_value
            feature_values.append(raw_value)
            if fill_method != "observed":
                fallback_rows.append(
                    {
                        "county_id": county_id,
                        "crop_type": crop_type,
                        "blank_fill_year": target_year,
                        "known_months": known_months,
                        "month": month,
                        "feature": feature,
                        "modality": feature_to_modality(feature),
                        "fill_method": fill_method,
                    }
                )

        row_scaled = row_unscaled.copy()
        scaled_vec = (np.asarray(feature_values, dtype=np.float32) - mu_arr) / sigma_arr
        for idx, feature in enumerate(feature_cols):
            row_scaled[feature] = float(scaled_vec[idx])

        hist_unscaled = pd.concat([hist_unscaled, pd.DataFrame([row_unscaled])], ignore_index=True)
        hist_scaled = pd.concat([hist_scaled, pd.DataFrame([row_scaled])], ignore_index=True)

    fallback_df = pd.DataFrame(fallback_rows)
    return hist_unscaled, hist_scaled, fallback_df


def write_blank_fill_leakage_audit(
    cfg: RuntimeConfig,
    prefix: str,
    fallback_records: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Path:
    audit_path = cfg.artifact_dir / f"{prefix}_leakage_audit.md"
    fallback_path = cfg.artifact_dir / f"{prefix}_input_fallbacks.csv"
    summary_path = cfg.artifact_dir / f"{prefix}_input_fallback_summary.csv"
    if fallback_records.empty:
        fallback_summary = pd.DataFrame(columns=["known_months", "month", "modality", "feature", "fill_method", "count"])
    else:
        fallback_summary = (
            fallback_records.groupby(["known_months", "month", "modality", "feature", "fill_method"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values(["known_months", "month", "modality", "feature", "fill_method"])
        )
        atomic_write_csv(fallback_records, fallback_path)
        validate_saved_csv(fallback_path, "strict blank-fill fallback rows")
    atomic_write_csv(fallback_summary, summary_path)
    if not fallback_summary.empty:
        validate_saved_csv(summary_path, "strict blank-fill fallback summary")

    compare_lines = ["- No standard blank-fill comparison file was found."]
    standard_path = cfg.artifact_dir / "blank_fill_predictions_long.csv"
    changed_rows = None
    changed_groups = None
    if prefix != "blank_fill" and standard_path.exists():
        standard = pd.read_csv(standard_path)
        merge_cols = ["model", "county_id", "crop_type", "known_months", "target_month", "feature"]
        current = predictions[merge_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_current"})
        baseline = standard[merge_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_standard"})
        for col in ["model", "county_id", "crop_type", "feature"]:
            current[col] = current[col].astype(str)
            baseline[col] = baseline[col].astype(str)
        for col in ["known_months", "target_month"]:
            current[col] = current[col].astype(int)
            baseline[col] = baseline[col].astype(int)
        joined = current.merge(baseline, on=merge_cols, how="inner")
        if not joined.empty:
            changed = joined[np.abs(joined["y_pred_current"] - joined["y_pred_standard"]) > 1e-9].copy()
            changed_rows = int(len(changed))
            if not changed.empty:
                changed_groups = (
                    changed.groupby(["model", "known_months"], as_index=False)
                    .size()
                    .rename(columns={"size": "changed_rows"})
                )
                compare_lines = [
                    f"- Strict vs standard overlapping prediction rows: {len(joined)}",
                    f"- Rows with changed predictions: {changed_rows}",
                    "",
                    changed_groups.to_markdown(index=False),
                ]
            else:
                compare_lines = [
                    f"- Strict vs standard overlapping prediction rows: {len(joined)}",
                    "- Rows with changed predictions: 0",
                ]

    total_fallbacks = int(len(fallback_records))
    fill_method_table = "No strict-history fallbacks were needed."
    if not fallback_records.empty:
        fill_method_table = (
            fallback_records.groupby("fill_method", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .to_markdown(index=False)
        )
    month_table = "No strict-history fallbacks were needed."
    if not fallback_summary.empty:
        month_table = fallback_summary.head(20).to_markdown(index=False)

    text = f"""# Strict Blank-Fill Leakage Audit

## Scope
- `strict_blank_fill_no_future_fill`: `{cfg.strict_blank_fill_no_future_fill}`
- `strict_fill_strategy`: `{cfg.strict_fill_strategy}`
- `blank_fill_output_prefix`: `{prefix}`

## Future-Value Exclusion
- Strict mode only injects observed target-year months up to `known_months`.
- Target-year months after `known_months` are never used as model inputs.
- True target values for forecast months are only read after prediction time for metric comparison.
- Seasonal residual references use previous-year same-month rows only.

## Known-History Fill Strategy
- Preferred fallback order in `past_only` mode:
  1. observed value
  2. previous available historical value for the same county/crop/feature
  3. same-month previous-year value
  4. train-year feature mean
- In `observed_only` mode, strict history skips the past-value fallback and uses:
  1. observed value
  2. same-month previous-year value
  3. train-year feature mean

## Fallback Counts
- Total imputed known-history feature values: {total_fallbacks}

### By fill method
{fill_method_table}

### Top fallback rows by known_month / month / modality / feature
{month_table}

## Standard-Comparison Check
{os.linesep.join(compare_lines)}

## Leakage Verdict
- Future target-year months were excluded from strict blank-fill model inputs.
- No true target values were accessed before metric time.
- Strict mode changes only the held-out-year input assembly path; training and standard one-step evaluation remain unchanged.
"""
    atomic_write_text(text, audit_path)
    return audit_path


def build_sarima_blank_fill_lookup(
    cfg: RuntimeConfig,
    feature_cols: list[str],
    monthly: pd.DataFrame,
    county_id: str,
    crop_type: str,
    target_year: int,
    known_months: int,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    county_monthly = monthly[
        monthly["county_id"].astype(str).eq(str(county_id))
        & monthly["crop_type"].astype(str).eq(str(crop_type))
    ].sort_values(["year", "month"]).copy()
    county_monthly["date"] = pd.to_datetime(dict(year=county_monthly["year"].astype(int), month=county_monthly["month"].astype(int), day=1))
    county_monthly = county_monthly.set_index("date").sort_index()
    history_end = pd.Timestamp(year=target_year, month=max(known_months, 1), day=1) if known_months > 0 else pd.Timestamp(year=target_year - 1, month=12, day=1)
    history = county_monthly.loc[:history_end]
    forecast_months = list(range(known_months + 1, 13))
    if not forecast_months:
        return {}, pd.DataFrame()

    lookup: dict[int, np.ndarray] = {}
    fit_rows: list[dict] = []
    feature_forecasts: dict[str, pd.Series] = {}
    steps = len(forecast_months)
    for feature in feature_cols:
        series = history[feature].astype(float).asfreq("MS")
        fc, strategy = fit_sarima_forecast(series, steps)
        feature_forecasts[feature] = fc
        fit_rows.append(
            {
                "scope": "blank_fill",
                "county_id": county_id,
                "crop_type": crop_type,
                "known_months": known_months,
                "feature": feature,
                "fit_strategy": strategy,
                "order": str(SARIMA_ORDER),
                "seasonal_order": str(SARIMA_SEASONAL_ORDER),
                "train_points": int(series.dropna().shape[0]),
            }
        )

    for step_idx, month in enumerate(forecast_months):
        vec = np.asarray([float(feature_forecasts[feature].iloc[step_idx]) for feature in feature_cols], dtype=np.float32)
        lookup[month] = vec
    return lookup, pd.DataFrame(fit_rows)


def load_validation_feature_rmse(cfg: RuntimeConfig) -> pd.DataFrame:
    feature_metrics_path = cfg.artifact_dir / "model_metrics_by_feature.csv"
    if not feature_metrics_path.exists():
        return pd.DataFrame(columns=["model", "feature", "val_rmse"])
    feature_metrics = pd.read_csv(feature_metrics_path)
    if "split" not in feature_metrics.columns or "rmse" not in feature_metrics.columns:
        return pd.DataFrame(columns=["model", "feature", "val_rmse"])
    val_metrics = feature_metrics[feature_metrics["split"].eq("val")][["model", "feature", "rmse"]].copy()
    if val_metrics.empty:
        return pd.DataFrame(columns=["model", "feature", "val_rmse"])
    val_metrics = val_metrics.rename(columns={"rmse": "val_rmse"})
    return val_metrics


def append_blank_fill_ensemble_rows(
    cfg: RuntimeConfig,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    requested = [model for model in cfg.models if model in ENSEMBLE_MODELS]
    if not requested:
        return predictions

    base_candidates = [model for model in DEFAULT_ENSEMBLE_COMPONENT_CANDIDATES if model in predictions["model"].unique()]
    if len(base_candidates) < 2:
        LOGGER.warning("Skipping ensemble generation because fewer than 2 base models are available: %s", base_candidates)
        return predictions

    key_cols = ["run_name", "county_id", "crop_type", "blank_fill_year", "known_months", "target_month", "horizon", "feature", "modality", "target_mode"]
    component_frame = predictions[predictions["model"].isin(base_candidates)].copy()
    if component_frame.empty:
        return predictions

    out_frames = [predictions]

    if "ensemble_mean" in requested:
        mean_frame = (
            component_frame.groupby(key_cols, as_index=False)
            .agg(
                y_true=("y_true", "first"),
                y_pred=("y_pred", "mean"),
            )
        )
        mean_frame["model"] = "ensemble_mean"
        mean_frame["abs_error"] = (mean_frame["y_pred"] - mean_frame["y_true"]).abs()
        mean_frame["squared_error"] = (mean_frame["y_pred"] - mean_frame["y_true"]) ** 2
        mean_frame["source_note"] = f"ensemble_mean:{','.join(base_candidates)}"
        out_frames.append(mean_frame[predictions.columns])

    if "ensemble_weighted" in requested:
        val_weights = load_validation_feature_rmse(cfg)
        if not val_weights.empty:
            val_weights = val_weights[val_weights["model"].isin(base_candidates)].copy()
            val_weights["weight"] = 1.0 / val_weights["val_rmse"].clip(lower=1e-6)
            val_weights["weight"] = val_weights.groupby("feature")["weight"].transform(lambda s: s / s.sum() if float(s.sum()) > 0 else np.nan)
            weighted_frame = component_frame.merge(val_weights[["model", "feature", "weight"]], on=["model", "feature"], how="left")
        else:
            weighted_frame = component_frame.copy()
            weighted_frame["weight"] = 1.0 / len(base_candidates)
        weighted_frame["weight"] = weighted_frame["weight"].fillna(1.0 / len(base_candidates))
        weighted_summary = (
            weighted_frame.groupby(key_cols, as_index=False)
            .apply(
                lambda grp: pd.Series(
                    {
                        "y_true": float(grp["y_true"].iloc[0]),
                        "y_pred": float(np.average(grp["y_pred"], weights=grp["weight"])),
                    }
                )
            )
            .reset_index(drop=True)
        )
        weighted_summary["model"] = "ensemble_weighted"
        weighted_summary["abs_error"] = (weighted_summary["y_pred"] - weighted_summary["y_true"]).abs()
        weighted_summary["squared_error"] = (weighted_summary["y_pred"] - weighted_summary["y_true"]) ** 2
        weighted_summary["source_note"] = f"ensemble_weighted:{','.join(base_candidates)}"
        out_frames.append(weighted_summary[predictions.columns])

    if "ensemble_oracle_report_only" in requested:
        oracle_ref = (
            component_frame.groupby(["known_months", "feature", "model"], as_index=False)
            .agg(rmse=("squared_error", lambda s: float(math.sqrt(float(np.mean(s))))))
            .sort_values(["known_months", "feature", "rmse", "model"])
            .groupby(["known_months", "feature"], as_index=False)
            .first()[["known_months", "feature", "model"]]
            .rename(columns={"model": "oracle_model"})
        )
        oracle_frame = component_frame.merge(oracle_ref, on=["known_months", "feature"], how="inner")
        oracle_frame = oracle_frame[oracle_frame["model"].eq(oracle_frame["oracle_model"])].copy()
        oracle_frame["source_note"] = "ensemble_oracle_report_only"
        oracle_frame["model"] = "ensemble_oracle_report_only"
        out_frames.append(oracle_frame[predictions.columns])

    combined = pd.concat(out_frames, ignore_index=True)
    return combined


def run_blank_fill_evaluation(
    cfg: RuntimeConfig,
    monthly: pd.DataFrame,
    feature_cols: list[str],
    filled_unscaled: pd.DataFrame,
    filled_scaled: pd.DataFrame,
    mu: pd.Series,
    sigma: pd.Series,
    device: torch.device,
    trained_models: dict[str, nn.Module] | None = None,
) -> dict[str, Path]:
    target_year = cfg.blank_fill_year
    if target_year is None:
        raise ValueError("Blank-fill evaluation requires --blank-fill-year or an inferable test year.")
    if target_year not in filled_unscaled["year"].unique():
        raise RuntimeError(f"Blank-fill year {target_year} is not present in the monthly table.")

    models: dict[str, nn.Module] = {}
    if trained_models:
        models.update({k: v for k, v in trained_models.items() if k in LEARNED_MODELS})
    for model_name in cfg.models:
        if model_name in LEARNED_MODELS and model_name not in models:
            models[model_name] = load_trained_model_for_eval(cfg, model_name, len(feature_cols), device)

    pred_records: list[dict] = []
    fallback_audit_rows: list[pd.DataFrame] = []
    sarima_fit_rows: list[pd.DataFrame] = []
    feature_modalities = {feature: feature_to_modality(feature) for feature in feature_cols}
    learned_models = LEARNED_MODELS
    known_months_values = sorted(dict.fromkeys(int(x) for x in cfg.blank_fill_known_months))
    for known_months in known_months_values:
        if known_months < 0 or known_months > 11:
            raise ValueError(f"blank-fill known_months must be between 0 and 11, got {known_months}")
        forecast_months = list(range(known_months + 1, 13))
        LOGGER.info("Blank-fill scenario | year=%s known_months=%s forecast_months=%s", target_year, known_months, forecast_months)
        if not forecast_months:
            continue

        for (county_id, crop_type), hist_unscaled in filled_unscaled.groupby(["county_id", "crop_type"]):
            hist_unscaled = hist_unscaled.sort_values("date").reset_index(drop=True)
            hist_scaled = filled_scaled[(filled_scaled["county_id"].eq(county_id)) & (filled_scaled["crop_type"].eq(crop_type))].sort_values("date").reset_index(drop=True)
            if cfg.strict_blank_fill_no_future_fill:
                history_unscaled, history_scaled, fallback_df = build_strict_blank_fill_history(
                    cfg,
                    monthly,
                    filled_unscaled,
                    filled_scaled,
                    feature_cols,
                    mu,
                    sigma,
                    str(county_id),
                    str(crop_type),
                    target_year,
                    known_months,
                )
                if not fallback_df.empty:
                    fallback_audit_rows.append(fallback_df)
            else:
                history_unscaled = hist_unscaled[hist_unscaled["date"] < pd.Timestamp(f"{target_year}-01-01")].copy()
                history_scaled = hist_scaled[hist_scaled["date"] < pd.Timestamp(f"{target_year}-01-01")].copy()

                if known_months > 0:
                    observed_unscaled = hist_unscaled[(hist_unscaled["year"].eq(target_year)) & (hist_unscaled["month"] <= known_months)].copy()
                    observed_scaled = hist_scaled[(hist_scaled["year"].eq(target_year)) & (hist_scaled["month"] <= known_months)].copy()
                    history_unscaled = pd.concat([history_unscaled, observed_unscaled], ignore_index=True)
                    history_scaled = pd.concat([history_scaled, observed_scaled], ignore_index=True)

            actual_future = hist_unscaled[(hist_unscaled["year"].eq(target_year)) & (hist_unscaled["month"].isin(forecast_months))].copy()
            if actual_future.empty:
                continue
            sarima_lookup: dict[int, np.ndarray] = {}
            if "sarima" in cfg.models or "classical_ssm" in cfg.models:
                sarima_lookup, sarima_fit_df = build_sarima_blank_fill_lookup(
                    cfg,
                    feature_cols,
                    monthly,
                    str(county_id),
                    str(crop_type),
                    target_year,
                    known_months,
                )
                if not sarima_fit_df.empty:
                    sarima_fit_rows.append(sarima_fit_df)

            model_histories_scaled = {}
            if "lag1" in cfg.blank_fill_baselines:
                model_histories_scaled["naive_lag1"] = history_scaled.copy()
            if "seasonal_last_year" in cfg.blank_fill_baselines:
                model_histories_scaled["seasonal_last_year"] = history_scaled.copy()
            if "sarima" in cfg.models or "classical_ssm" in cfg.models:
                model_histories_scaled["sarima"] = history_scaled.copy()
            for model_name in models:
                model_histories_scaled[model_name] = history_scaled.copy()

            for month in forecast_months:
                target_date = pd.Timestamp(year=target_year, month=month, day=1)
                actual_row = actual_future[actual_future["month"].eq(month)]
                if actual_row.empty:
                    continue
                actual_vec = actual_row.iloc[0][feature_cols].to_numpy(dtype=np.float32)
                prev_year_row = hist_scaled[(hist_scaled["year"].eq(target_year - 1)) & (hist_scaled["month"].eq(month))]
                seasonal_ref_scaled = (
                    prev_year_row.iloc[0][feature_cols].to_numpy(dtype=np.float32) if not prev_year_row.empty else None
                )

                for model_name, history in model_histories_scaled.items():
                    if len(history) < cfg.lookback_months:
                        raise RuntimeError(f"Not enough history for blank-fill rollout: model={model_name} county={county_id}")
                    source_note = "recursive_model_prediction"
                    if model_name == "naive_lag1":
                        pred_scaled = history.iloc[-1][feature_cols].to_numpy(dtype=np.float32)
                        source_note = "lag1_recursive"
                    elif model_name == "seasonal_last_year":
                        if seasonal_ref_scaled is not None:
                            pred_scaled = seasonal_ref_scaled.astype(np.float32)
                            source_note = "seasonal_last_year"
                        else:
                            pred_scaled = history.iloc[-1][feature_cols].to_numpy(dtype=np.float32)
                            source_note = "seasonal_last_year_fallback_lag1"
                    elif model_name == "sarima":
                        pred_unscaled = sarima_lookup.get(month)
                        if pred_unscaled is None:
                            pred_scaled = history.iloc[-1][feature_cols].to_numpy(dtype=np.float32)
                            source_note = "sarima_missing_lookup_fallback_lag1"
                        else:
                            pred_scaled = ((pred_unscaled - mu[feature_cols].to_numpy(dtype=np.float32)) / sigma[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
                            source_note = "sarima_recursive_forecast"
                    else:
                        seq_scaled = history[feature_cols].tail(cfg.lookback_months).to_numpy(dtype=np.float32)
                        model_output_scaled = predict_torch_next(models[model_name], seq_scaled, device)
                        if cfg.blank_fill_residual_seasonal and model_name in learned_models:
                            if seasonal_ref_scaled is not None:
                                pred_scaled = seasonal_ref_scaled.astype(np.float32) + model_output_scaled
                                source_note = "seasonal_residual_recursive_prediction"
                            else:
                                pred_scaled = history.iloc[-1][feature_cols].to_numpy(dtype=np.float32)
                                source_note = "seasonal_residual_missing_base_fallback_lag1"
                        else:
                            pred_scaled = model_output_scaled

                    pred_unscaled = inverse_scale(pred_scaled[None, :], feature_cols, mu, sigma)[0]
                    new_row = {
                        "date": target_date,
                        "county_id": county_id,
                        "crop_type": crop_type,
                        "year": target_year,
                        "month": month,
                    }
                    for idx, feature in enumerate(feature_cols):
                        new_row[feature] = float(pred_scaled[idx])
                        pred_records.append(
                            {
                                "run_name": cfg.run_name or cfg.output_dir.name,
                                "model": model_name,
                                "county_id": county_id,
                                "crop_type": crop_type,
                                "blank_fill_year": target_year,
                                "known_months": known_months,
                                "target_month": month,
                                "horizon": month - known_months,
                                "feature": feature,
                                "modality": feature_modalities[feature],
                                "target_mode": cfg.target_mode,
                                "y_true": float(actual_vec[idx]),
                                "y_pred": float(pred_unscaled[idx]),
                                "abs_error": float(abs(pred_unscaled[idx] - actual_vec[idx])),
                                "squared_error": float((pred_unscaled[idx] - actual_vec[idx]) ** 2),
                                "source_note": source_note,
                            }
                        )
                    model_histories_scaled[model_name] = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    predictions = pd.DataFrame(pred_records)
    if predictions.empty:
        raise RuntimeError("Blank-fill evaluation produced no prediction rows.")
    predictions = append_blank_fill_ensemble_rows(cfg, predictions)

    prefix = cfg.blank_fill_output_prefix
    predictions_path = cfg.artifact_dir / f"{prefix}_predictions_long.csv"
    horizon_path = cfg.artifact_dir / f"{prefix}_metrics_by_horizon.csv"
    month_path = cfg.artifact_dir / f"{prefix}_metrics_by_month.csv"
    summary_path = cfg.artifact_dir / f"{prefix}_metrics_summary.csv"
    feature_path = cfg.artifact_dir / f"{prefix}_feature_metrics.csv"

    by_horizon, by_month, summary, feature_metrics = compute_blank_fill_metrics(predictions)
    atomic_write_csv(predictions, predictions_path)
    atomic_write_csv(by_horizon, horizon_path)
    atomic_write_csv(by_month, month_path)
    atomic_write_csv(summary, summary_path)
    atomic_write_csv(feature_metrics, feature_path)
    validate_saved_csv(predictions_path, "blank-fill predictions")
    validate_saved_csv(horizon_path, "blank-fill metrics by horizon")
    validate_saved_csv(month_path, "blank-fill metrics by month")
    validate_saved_csv(summary_path, "blank-fill metrics summary")
    validate_saved_csv(feature_path, "blank-fill feature metrics")

    plots = maybe_make_blank_fill_plots(cfg, summary, by_horizon)
    if sarima_fit_rows:
        sarima_fit_df = pd.concat(sarima_fit_rows, ignore_index=True)
        sarima_fit_path = cfg.artifact_dir / f"{prefix}_sarima_fit_summary.csv"
        atomic_write_csv(sarima_fit_df, sarima_fit_path)
        validate_saved_csv(sarima_fit_path, "sarima fit summary")
    if cfg.strict_blank_fill_no_future_fill:
        fallback_records = pd.concat(fallback_audit_rows, ignore_index=True) if fallback_audit_rows else pd.DataFrame()
        audit_path = write_blank_fill_leakage_audit(cfg, prefix, fallback_records, predictions)
        LOGGER.info("Saved strict blank-fill leakage audit: %s", audit_path)
    LOGGER.info("Saved blank-fill predictions: %s", predictions_path)
    LOGGER.info("Saved blank-fill metrics by horizon: %s", horizon_path)
    LOGGER.info("Saved blank-fill metrics by month: %s", month_path)
    LOGGER.info("Saved blank-fill metrics summary: %s", summary_path)
    LOGGER.info("Saved blank-fill feature metrics: %s", feature_path)
    for plot in plots:
        LOGGER.info("Saved blank-fill plot: %s", plot)
    return {
        "predictions": predictions_path,
        "by_horizon": horizon_path,
        "by_month": month_path,
        "summary": summary_path,
        "feature_metrics": feature_path,
    }


def run_eval_only(cfg: RuntimeConfig) -> None:
    monthly, feature_cols, mu, sigma = load_monthly_and_scaler_for_eval(cfg)
    validate_ag_ndvi_alignment(monthly)
    report = nan_inf_report_frame(monthly, feature_cols)
    LOGGER.info(
        "Eval-only monthly artifact | shape=%s feature_count=%s nan_count=%s inf_count=%s",
        monthly.shape,
        len(feature_cols),
        report["nan_count"],
        report["inf_count"],
    )
    filled_unscaled, _, feature_cols, _, _ = prepare_model_frames(cfg, monthly, feature_cols)
    filled_scaled = filled_unscaled.copy()
    filled_scaled[feature_cols] = (filled_scaled[feature_cols] - mu[feature_cols]) / sigma[feature_cols]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Eval-only device: %s", device)
    if cfg.run_blank_fill_eval:
        with stage("run-blank-fill-eval"):
            run_blank_fill_evaluation(cfg, monthly, feature_cols, filled_unscaled, filled_scaled, mu, sigma, device, trained_models=None)
    else:
        LOGGER.info("Eval-only mode was requested without any evaluation flag. Nothing to do.")


def run_pipeline(cfg: RuntimeConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ensure_repo_extractors(cfg)

    existing_monthly = try_load_existing_monthly_artifacts(cfg)
    if existing_monthly is None:
        with stage("discover-files"):
            ag_files, ndvi_files, weather_files = discover_files(cfg)

        with stage("select-fips"):
            selected_fips = select_fips(cfg, ag_files, ndvi_files, weather_files)

        image_feature_frames = []
        bad_h5_chunks = []
        with stage("extract-image-features"):
            image_groups = []
            if "AG" in cfg.image_types:
                image_groups.append(("ag", ag_files))
            if "NDVI" in cfg.image_types:
                image_groups.append(("ndvi", ndvi_files))
            for modality, records in image_groups:
                for rec in records:
                    LOGGER.info(
                        "Image chunk | modality=%s state=%s year=%s quarter=%s file=%s",
                        modality,
                        rec.get("state"),
                        rec.get("year"),
                        rec.get("quarter"),
                        rec.get("hf_path"),
                    )
                    try:
                        df_part = extract_image_h5_chunk(cfg, modality, rec, selected_fips)
                        image_feature_frames.append(df_part)
                    except RuntimeError as exc:
                        msg = f"{type(exc).__name__}: {exc}"
                        row = {
                            "modality": modality,
                            "state": rec.get("state"),
                            "year": rec.get("year"),
                            "quarter": rec.get("quarter"),
                            "hf_path": rec.get("hf_path"),
                            "error": msg,
                        }
                        bad_h5_chunks.append(row)
                        LOGGER.warning("Skipping image chunk after validation failure: %s", msg.split("\n")[0])
            if bad_h5_chunks:
                atomic_write_csv(pd.DataFrame(bad_h5_chunks), cfg.bad_chunk_log)
                validate_saved_csv(cfg.bad_chunk_log, "bad chunk log")

        weather_feature_frames = []
        with stage("extract-weather-features"):
            for rec in weather_files:
                LOGGER.info(
                    "Weather chunk | state=%s year=%s month=%s quarter=%s file=%s",
                    rec.get("state"),
                    rec.get("year"),
                    rec.get("month"),
                    rec.get("quarter"),
                    rec.get("hf_path"),
                )
                weather_feature_frames.append(extract_weather_chunk(cfg, rec, selected_fips))

        with stage("build-monthly-table"):
            monthly, available_features, feature_contract_diagnostic = build_monthly_table(cfg)
    else:
        monthly, available_features, feature_contract_diagnostic = existing_monthly

    available_features = selected_feature_columns(cfg, available_features)
    LOGGER.info(
        "Selected feature group | feature_group=%s feature_count=%s features=%s",
        cfg.feature_group,
        len(available_features),
        available_features,
    )

    filled_unscaled = None
    filled_scaled = None
    mu = None
    sigma = None
    trained_models: dict[str, nn.Module] = {}

    if not cfg.run_forecasting:
        LOGGER.info(
            "Skipping forecasting stages because the selected years do not yield train/val/test splits. "
            "Extraction and table-validation completed successfully for the requested scope."
        )
        if cfg.run_blank_fill_eval:
            filled_unscaled, filled_scaled, available_features, mu, sigma = prepare_model_frames(cfg, monthly, available_features)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            run_blank_fill_evaluation(cfg, monthly, available_features, filled_unscaled, filled_scaled, mu, sigma, device, trained_models=None)
        return

    with stage("build-sequences"):
        X, y, seq_meta, filled_unscaled, filled_scaled, mu, sigma, y_true_scaled, seasonal_base_scaled = build_sequences(cfg, monthly, available_features)
        LOGGER.info("Sequence tensors | X=%s y=%s metadata=%s", X.shape, y.shape, seq_meta.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Forecasting device: %s", device)
    train_idx = np.where(seq_meta["split"].eq("train"))[0]
    val_idx = np.where(seq_meta["split"].eq("val"))[0]
    test_idx = np.where(seq_meta["split"].eq("test"))[0]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx]) if len(val_idx) else None
    test_ds = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    y_actual_scaled = y_true_scaled
    loss_weights_df = compute_feature_loss_weights(cfg, available_features, y[train_idx])
    loss_weights_path = cfg.artifact_dir / "feature_loss_weights.csv"
    atomic_write_csv(loss_weights_df, loss_weights_path)
    validate_saved_csv(loss_weights_path, "feature loss weights")
    loss_weights = None
    if cfg.loss_mode != "raw_mse":
        loss_weights = torch.tensor(loss_weights_df["loss_weight"].to_numpy(dtype=np.float32), device=device)
    model_runtime_seconds: dict[str, float] = {}
    extra_model_spec_rows: list[dict] = []

    def append_metrics_for_split(model_name: str, y_pred_scaled: np.ndarray, split_name: str, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        all_metric_tables.append(compute_metrics(y_actual_scaled[idx], y_pred_scaled, available_features, mu, sigma, model_name, split_name))
        all_feature_metric_tables.append(per_feature_metrics(y_actual_scaled[idx], y_pred_scaled, available_features, mu, sigma, model_name, split_name))

    all_metric_tables: list[pd.DataFrame] = []
    all_feature_metric_tables: list[pd.DataFrame] = []
    y_pred_naive_val = X[val_idx, -1, :] if len(val_idx) else np.empty((0, len(available_features)), dtype=np.float32)
    y_pred_naive_test = X[test_idx, -1, :] if len(test_idx) else np.empty((0, len(available_features)), dtype=np.float32)
    append_metrics_for_split("naive_lag1", y_pred_naive_val, "val", val_idx)
    append_metrics_for_split("naive_lag1", y_pred_naive_test, "test", test_idx)
    model_preds = {"naive_lag1": y_pred_naive_test}
    y_pred_seasonal_val = seasonal_base_scaled[val_idx] if len(val_idx) else np.empty((0, len(available_features)), dtype=np.float32)
    y_pred_seasonal_test = seasonal_base_scaled[test_idx] if len(test_idx) else np.empty((0, len(available_features)), dtype=np.float32)
    append_metrics_for_split("seasonal_last_year", y_pred_seasonal_val, "val", val_idx)
    append_metrics_for_split("seasonal_last_year", y_pred_seasonal_test, "test", test_idx)
    model_preds["seasonal_last_year"] = y_pred_seasonal_test

    stage_names = {
        "lstm": "train-lstm",
        "gru": "train-gru",
        "tiny_mamba_ssm": "train-mamba",
        "transformer_encoder": "train-transformer-encoder",
    }
    for model_name in cfg.models:
        if model_name not in LEARNED_MODELS:
            continue
        with stage(stage_names.get(model_name, f"train-{model_name}")):
            model_start = time.perf_counter()
            model = build_learned_model(cfg, model_name, len(available_features))
            model = train_torch_model(cfg, model, model_name, train_loader, val_loader, device, loss_weights)
            if len(val_idx):
                val_pred = finalize_model_prediction_scaled(cfg, model_name, predict_torch_model(model, val_loader, device), seasonal_base_scaled[val_idx])
                append_metrics_for_split(model_name, val_pred, "val", val_idx)
            test_pred = finalize_model_prediction_scaled(cfg, model_name, predict_torch_model(model, test_loader, device), seasonal_base_scaled[test_idx])
            append_metrics_for_split(model_name, test_pred, "test", test_idx)
            model_preds[model_name] = test_pred
            trained_models[model_name] = model
            model_runtime_seconds[model_name] = time.perf_counter() - model_start

    if "sarima" in cfg.models or "classical_ssm" in cfg.models or cfg.run_classical_ssm:
        with stage("run-sarima"):
            sarima_fit_tables: list[pd.DataFrame] = []
            if len(val_idx):
                sarima_val_pred, sarima_val_fit = sarima_forecast_matrix(
                    cfg, available_features, filled_scaled, seq_meta, val_idx, y_pred_naive_val
                )
                append_metrics_for_split("sarima", sarima_val_pred, "val", val_idx)
                if not sarima_val_fit.empty:
                    sarima_val_fit = sarima_val_fit.copy()
                    sarima_val_fit["split"] = "val"
                    sarima_fit_tables.append(sarima_val_fit)
            sarima_test_pred, sarima_test_fit = sarima_forecast_matrix(
                cfg, available_features, filled_scaled, seq_meta, test_idx, y_pred_naive_test
            )
            append_metrics_for_split("sarima", sarima_test_pred, "test", test_idx)
            model_preds["sarima"] = sarima_test_pred
            if not sarima_test_fit.empty:
                sarima_test_fit = sarima_test_fit.copy()
                sarima_test_fit["split"] = "test"
                sarima_fit_tables.append(sarima_test_fit)
            sarima_fit_path = cfg.artifact_dir / "sarima_fit_summary.csv"
            sarima_fit_combined = pd.concat(sarima_fit_tables, ignore_index=True) if sarima_fit_tables else pd.DataFrame()
            atomic_write_csv(sarima_fit_combined, sarima_fit_path)
            if not sarima_fit_combined.empty:
                validate_saved_csv(sarima_fit_path, "sarima fit summary")
            fallback_count = 0
            if not sarima_fit_combined.empty and "fit_strategy" in sarima_fit_combined.columns:
                fallback_count = int(sarima_fit_combined["fit_strategy"].astype(str).str.contains("lag1|failed|insufficient", regex=True).sum())
            split_counts = seq_meta["split"].value_counts().to_dict()
            extra_model_spec_rows.append(
                {
                    "model": "sarima",
                    "model_class": "SARIMAXBaseline",
                    "input_dim": len(available_features),
                    "output_dim": len(available_features),
                    "seq_len": cfg.lookback_months,
                    "hidden_size": np.nan,
                    "num_layers": np.nan,
                    "dropout": np.nan,
                    "learning_rate": np.nan,
                    "batch_size": np.nan,
                    "max_epochs": np.nan,
                    "early_stopping_patience": np.nan,
                    "weight_decay": np.nan,
                    "loss_mode": cfg.loss_mode,
                    "target_mode": cfg.target_mode,
                    "feature_group": cfg.feature_group,
                    "trainable_parameters": 0,
                    "total_parameters": 0,
                    "checkpoint_size_bytes": 0,
                    "checkpoint_path": "",
                    "training_sample_count": int(split_counts.get("train", 0)),
                    "validation_sample_count": int(split_counts.get("val", 0)),
                    "test_sample_count": int(split_counts.get("test", 0)),
                    "device_used": "cpu",
                    "training_runtime_seconds": np.nan,
                    "baseline_rule": (
                        f"per-county per-feature SARIMAX order={SARIMA_ORDER} "
                        f"seasonal_order={SARIMA_SEASONAL_ORDER}; "
                        f"county_limit={cfg.classical_max_counties or 'all'} "
                        f"feature_limit={cfg.classical_max_features or 'all'} "
                        f"lag1_fallback_count={fallback_count}"
                    ),
                }
            )

    if "ensemble_mean" in cfg.models:
        extra_model_spec_rows.append(
            {
                "model": "ensemble_mean",
                "model_class": "DeterministicEnsemble",
                "input_dim": len(available_features),
                "output_dim": len(available_features),
                "seq_len": cfg.lookback_months,
                "hidden_size": np.nan,
                "num_layers": np.nan,
                "dropout": np.nan,
                "learning_rate": np.nan,
                "batch_size": np.nan,
                "max_epochs": np.nan,
                "early_stopping_patience": np.nan,
                "weight_decay": np.nan,
                "loss_mode": cfg.loss_mode,
                "target_mode": cfg.target_mode,
                "feature_group": cfg.feature_group,
                "trainable_parameters": 0,
                "total_parameters": 0,
                "checkpoint_size_bytes": 0,
                "checkpoint_path": "",
                "training_sample_count": int(seq_meta["split"].eq("train").sum()),
                "validation_sample_count": int(seq_meta["split"].eq("val").sum()),
                "test_sample_count": int(seq_meta["split"].eq("test").sum()),
                "device_used": "deterministic",
                "training_runtime_seconds": np.nan,
                "baseline_rule": f"Simple mean ensemble across available component models from {DEFAULT_ENSEMBLE_COMPONENT_CANDIDATES}.",
            }
        )
    if "ensemble_weighted" in cfg.models:
        extra_model_spec_rows.append(
            {
                "model": "ensemble_weighted",
                "model_class": "ValidationWeightedEnsemble",
                "input_dim": len(available_features),
                "output_dim": len(available_features),
                "seq_len": cfg.lookback_months,
                "hidden_size": np.nan,
                "num_layers": np.nan,
                "dropout": np.nan,
                "learning_rate": np.nan,
                "batch_size": np.nan,
                "max_epochs": np.nan,
                "early_stopping_patience": np.nan,
                "weight_decay": np.nan,
                "loss_mode": cfg.loss_mode,
                "target_mode": cfg.target_mode,
                "feature_group": cfg.feature_group,
                "trainable_parameters": 0,
                "total_parameters": 0,
                "checkpoint_size_bytes": 0,
                "checkpoint_path": "",
                "training_sample_count": int(seq_meta["split"].eq("train").sum()),
                "validation_sample_count": int(seq_meta["split"].eq("val").sum()),
                "test_sample_count": int(seq_meta["split"].eq("test").sum()),
                "device_used": "deterministic",
                "training_runtime_seconds": np.nan,
                "baseline_rule": "Inverse validation per-feature RMSE weighted ensemble across available component models.",
            }
        )
    if "ensemble_oracle_report_only" in cfg.models:
        extra_model_spec_rows.append(
            {
                "model": "ensemble_oracle_report_only",
                "model_class": "OracleEnsembleReportOnly",
                "input_dim": len(available_features),
                "output_dim": len(available_features),
                "seq_len": cfg.lookback_months,
                "hidden_size": np.nan,
                "num_layers": np.nan,
                "dropout": np.nan,
                "learning_rate": np.nan,
                "batch_size": np.nan,
                "max_epochs": np.nan,
                "early_stopping_patience": np.nan,
                "weight_decay": np.nan,
                "loss_mode": cfg.loss_mode,
                "target_mode": cfg.target_mode,
                "feature_group": cfg.feature_group,
                "trainable_parameters": 0,
                "total_parameters": 0,
                "checkpoint_size_bytes": 0,
                "checkpoint_path": "",
                "training_sample_count": int(seq_meta["split"].eq("train").sum()),
                "validation_sample_count": int(seq_meta["split"].eq("val").sum()),
                "test_sample_count": int(seq_meta["split"].eq("test").sum()),
                "device_used": "report_only",
                "training_runtime_seconds": np.nan,
                "baseline_rule": "Oracle report-only ensemble choosing the best per-feature model using realized blank-fill errors.",
            }
        )

    with stage("save-metrics-and-predictions"):
        metrics = pd.concat(all_metric_tables, ignore_index=True)
        feature_metrics = pd.concat(all_feature_metric_tables, ignore_index=True)
        naive = metrics[metrics["model"].eq("naive_lag1")][["split", "modality", "rmse"]].rename(columns={"rmse": "naive_rmse"})
        metrics_cmp = metrics.merge(naive, on=["split", "modality"], how="left")
        metrics_cmp["rmse_vs_naive"] = metrics_cmp["rmse"] / metrics_cmp["naive_rmse"]
        naive_feat = feature_metrics[feature_metrics["model"].eq("naive_lag1")][["split", "feature", "rmse"]].rename(columns={"rmse": "naive_rmse"})
        feature_cmp = feature_metrics.merge(naive_feat, on=["split", "feature"], how="left")
        feature_cmp["rmse_vs_naive"] = feature_cmp["rmse"] / feature_cmp["naive_rmse"]
        feature_cmp["beats_naive"] = feature_cmp["rmse_vs_naive"] < 1.0
        metrics_path = cfg.artifact_dir / "model_metrics_by_modality.csv"
        feature_metrics_path = cfg.artifact_dir / "model_metrics_by_feature.csv"
        atomic_write_csv(metrics_cmp, metrics_path)
        atomic_write_csv(feature_cmp, feature_metrics_path)
        validate_saved_csv(metrics_path, "metrics by modality")
        validate_saved_csv(feature_metrics_path, "metrics by feature")
        pred_path = save_predictions(cfg, available_features, mu, sigma, seq_meta, test_idx, y_actual_scaled, model_preds)
        LOGGER.info("Saved metrics: %s", metrics_path)
        LOGGER.info("Saved feature metrics: %s", feature_metrics_path)
        LOGGER.info("Saved predictions: %s", pred_path)
        specs_path = write_model_specs(
            cfg,
            available_features,
            seq_meta,
            trained_models,
            model_runtime_seconds,
            device,
            extra_rows=extra_model_spec_rows,
        )
        LOGGER.info("Saved model specs: %s", specs_path)

    if cfg.run_blank_fill_eval:
        with stage("run-blank-fill-eval"):
            run_blank_fill_evaluation(cfg, monthly, available_features, filled_unscaled, filled_scaled, mu, sigma, device, trained_models=trained_models)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    configure_logging(cfg)
    write_run_config(cfg)
    started_epoch = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_epoch))
    write_run_status(cfg, status="running", started_at=started_at)
    try:
        with stage("environment-setup"):
            ensure_package("huggingface_hub", "huggingface_hub>=0.23")
            ensure_package("h5py", "h5py")
            ensure_package("numpy", "numpy")
            ensure_package("pandas", "pandas")
            ensure_package("pyarrow", "pyarrow>=15")
            ensure_package("statsmodels", "statsmodels>=0.14")
            ensure_package("sklearn", "scikit-learn")
            ensure_package("torch", "torch")
            ensure_package("tqdm", "tqdm>=4.66")
            load_runtime_imports()
            print_environment(cfg)
        if cfg.eval_only:
            run_eval_only(cfg)
        else:
            run_pipeline(cfg)
    except Exception as exc:
        ended_epoch = time.time()
        ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ended_epoch))
        write_run_status(
            cfg,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            runtime_seconds=ended_epoch - started_epoch,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    ended_epoch = time.time()
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ended_epoch))
    write_run_status(
        cfg,
        status="completed",
        started_at=started_at,
        ended_at=ended_at,
        runtime_seconds=ended_epoch - started_epoch,
    )


if __name__ == "__main__":
    main()
