#!/usr/bin/env python3
"""Auto-download missing CropNet files and export modality-specific JSONL splits."""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_fusion_ai.preprocessing import (  # noqa: E402
    build_modality_records_by_split,
    build_usda_selection,
    build_usda_records_from_snapshot,
    get_state_abbr,
    write_modality_jsonl_splits,
)

DEFAULT_BASE_DIR = Path(os.environ.get("CROPNET_BASE_DIR", "/mnt/data/CropNet"))
DEFAULT_YEAR = 2022
DEFAULT_YEARS = [DEFAULT_YEAR]
DEFAULT_FIPS_CODES = ["10003", "22007"]
DEFAULT_CROPS = ["soybeans"]
DEFAULT_MODALITIES = ["ag", "ndvi", "weather"]
DEFAULT_TARGET = "yield"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "cropnet"
DEFAULT_REPO_ID = "CropNet/CropNet"

_CROPNET_CROP_NAMES = {
    "corn": ("Corn", "Corn"),
    "cotton": ("Cotton", "Cotton"),
    "soybeans": ("Soybeans", "Soybean"),
    "soybean": ("Soybeans", "Soybean"),
    "winter wheat": ("WinterWheat", "WinterWheat"),
    "winterwheat": ("WinterWheat", "WinterWheat"),
    "winter_wheat": ("WinterWheat", "WinterWheat"),
}


def build_argument_parser() -> ArgumentParser:
    """Create the command-line interface."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Local CropNet dataset root, for example /mnt/data/CropNet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where modality-specific split folders will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Temporary cache directory for matplotlib and intermediate files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo to download from.",
    )
    parser.add_argument(
        "--target",
        choices=("yield", "production"),
        default=DEFAULT_TARGET,
        help="Which USDA label to extract from the source rows.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Single year to process, for example 2022.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit year list. Overrides --year when provided.",
    )
    parser.add_argument(
        "--fips-codes",
        nargs="+",
        default=DEFAULT_FIPS_CODES,
        help="County FIPS codes to keep the local read bounded.",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=DEFAULT_CROPS,
        help="Crop filters, for example soybeans corn cotton 'winter wheat'.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=DEFAULT_MODALITIES,
        default=DEFAULT_MODALITIES,
        help="Which modality outputs to build.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep intermediate cache directories instead of deleting them.",
    )
    return parser


def _years_from_args(args: Namespace) -> list[int]:
    if args.years is not None:
        return [int(year) for year in args.years]
    return [int(args.year)]


def _normalize_fips_codes(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalized = {str(value).strip().zfill(5) for value in values if str(value).strip()}
    return sorted(normalized)


def _cropnet_crop_names(crops: Sequence[str]) -> list[tuple[str, str]]:
    crop_names: list[tuple[str, str]] = []
    for crop in crops:
        normalized = str(crop).strip().lower()
        try:
            crop_names.append(_CROPNET_CROP_NAMES[normalized])
        except KeyError as exc:
            msg = f"Unsupported CropNet crop type: {crop!r}"
            raise SystemExit(msg) from exc
    return crop_names


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _write_root_manifest(
    output_dir: Path,
    *,
    base_dir: Path,
    repo_id: str,
    years: Sequence[int],
    crops: Sequence[str],
    modalities: Sequence[str],
    selected_fips: Sequence[str],
    modality_stats: dict[str, dict[str, object]],
) -> None:
    manifest = {
        "source": "huggingface_hub.snapshot_download",
        "repo_id": repo_id,
        "base_dir": str(base_dir),
        "years": list(years),
        "crops": list(crops),
        "modalities": list(modalities),
        "selected_fips": list(selected_fips),
        "modality_stats": modality_stats,
    }
    _manifest_path(output_dir).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _ensure_cropnet_data(
    base_dir: Path,
    cache_dir: Path,
    *,
    repo_id: str,
    years: Sequence[int],
    fips_codes: Sequence[str],
    crops: Sequence[str],
) -> None:
    """Download the requested CropNet subset into the local base directory."""
    cache_root = cache_dir / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)

    allow_patterns: list[str] = []
    crop_names = _cropnet_crop_names(crops)
    state_abbrs = sorted({get_state_abbr(code[:2]) for code in fips_codes})

    for folder_name, file_stem in crop_names:
        for year in years:
            allow_patterns.append(
                f"USDA Crop Dataset/{folder_name}/{year}/USDA_{file_stem}_County_{year}.csv"
            )

    for modality in ("ag", "ndvi"):
        image_type = "AG" if modality == "ag" else "NDVI"
        for year in years:
            for state_abbr in state_abbrs:
                allow_patterns.append(
                    f"Sentinel-2 Imagery/data/{image_type}/{year}/{state_abbr}/*.h5"
                )

    for year in years:
        for state_abbr in state_abbrs:
            allow_patterns.append(
                f"WRF-HRRR Computed Dataset/data/{year}/{state_abbr}/*.csv"
            )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(base_dir),
        cache_dir=str(cache_root),
        allow_patterns=allow_patterns,
        ignore_patterns=["**/.DS_Store"],
    )


def run(args: Namespace) -> int:
    """Run the CropNet local-data export workflow."""
    years = _years_from_args(args)
    selected_fips = _normalize_fips_codes(args.fips_codes)
    if not selected_fips:
        raise SystemExit("Please provide at least one FIPS code.")

    base_dir = args.base_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _ensure_cropnet_data(
        base_dir,
        cache_dir,
        repo_id=args.repo_id,
        years=years,
        fips_codes=selected_fips,
        crops=args.crops,
    )

    usda_paths = sorted(
        str(path.relative_to(base_dir))
        for path in base_dir.glob("USDA Crop Dataset/*/*/USDA_*_County_*.csv")
    )
    usda_paths = build_usda_selection(usda_paths, years=years, crops=args.crops)
    if not usda_paths:
        raise SystemExit(
            f"No USDA CropNet files were found under {base_dir}. "
            "Check the repo id, years, and crop filters."
        )

    usda_records = build_usda_records_from_snapshot(
        base_dir,
        usda_paths,
        target_kind=args.target,
    )
    usda_records = [
        record
        for record in usda_records
        if record.get("county_id") is not None
        and str(record["county_id"]).zfill(5) in selected_fips
    ]
    if not usda_records:
        raise SystemExit(
            "No USDA rows matched the requested FIPS codes after download."
        )

    modality_records: dict[str, dict[str, list[dict[str, object]]]] = {
        modality: {"train": [], "validation": [], "test": []}
        for modality in args.modalities
    }
    for modality in args.modalities:
        modality_records[modality] = build_modality_records_by_split(
            base_dir,
            usda_records,
            modality=modality,
        )

    modality_stats: dict[str, dict[str, object]] = {}
    for modality in args.modalities:
        counts = write_modality_jsonl_splits(
            modality_records[modality],
            output_dir / modality,
        )
        modality_stats[modality] = {
            "split_counts": counts,
            "rows_written": sum(counts.values()),
        }

    _write_root_manifest(
        output_dir,
        base_dir=base_dir,
        repo_id=args.repo_id,
        years=years,
        crops=args.crops,
        modalities=args.modalities,
        selected_fips=selected_fips,
        modality_stats=modality_stats,
    )

    print(
        "Wrote modality JSONL splits to",
        output_dir,
        "for",
        ", ".join(args.modalities),
    )
    return 0


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
