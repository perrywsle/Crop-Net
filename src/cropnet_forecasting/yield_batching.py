from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from cropnet_forecasting.yield_regression import (
        load_usda_yield_tables,
        normalize_county_id,
        save_json,
    )
else:  # pragma: no cover - exercised when run as a module
    from .yield_regression import load_usda_yield_tables, normalize_county_id, save_json

DEFAULT_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)
DEFAULT_QUARTERS = ("Q2", "Q3")


@dataclass(frozen=True)
class BatchManifest:
    frame: pd.DataFrame
    path: Path | None = None


def derive_label_fips(
    usda_paths: list[Path],
    crop_type: str = "corn",
    *,
    state_fips: str = "19",
) -> list[str]:
    labels = load_usda_yield_tables(usda_paths, crop_type=crop_type)
    if state_fips:
        prefix = str(state_fips).zfill(2)
        labels = labels[normalize_county_id(labels["county_id"]).str.startswith(prefix)]
    fips = normalize_county_id(labels["county_id"]).dropna().unique().tolist()
    return sorted(str(value).zfill(5) for value in fips)


def build_batch_manifest(
    fips_codes: list[str],
    *,
    batch_size: int = 10,
    years: tuple[int, ...] = DEFAULT_YEARS,
    quarters: tuple[str, ...] = DEFAULT_QUARTERS,
    run_prefix: str = "corn_ia_gs_2017_2022",
) -> pd.DataFrame:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    clean_fips = sorted({str(code).zfill(5) for code in fips_codes})
    rows: list[dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(clean_fips), batch_size), start=1):
        batch_fips = clean_fips[start : start + batch_size]
        rows.append(
            {
                "batch_id": idx,
                "fips_codes": " ".join(batch_fips),
                "years": " ".join(str(year) for year in years),
                "quarters": " ".join(quarters),
                "run_name": f"{run_prefix}_batch_{idx:03d}",
            }
        )
    return pd.DataFrame(rows)


def write_batch_manifest(
    usda_paths: list[Path],
    output_path: Path,
    *,
    crop_type: str = "corn",
    state_fips: str = "19",
    batch_size: int = 10,
    years: tuple[int, ...] = DEFAULT_YEARS,
    quarters: tuple[str, ...] = DEFAULT_QUARTERS,
    run_prefix: str = "corn_ia_gs_2017_2022",
) -> BatchManifest:
    fips_codes = derive_label_fips(usda_paths, crop_type=crop_type, state_fips=state_fips)
    manifest = build_batch_manifest(
        fips_codes,
        batch_size=batch_size,
        years=years,
        quarters=quarters,
        run_prefix=run_prefix,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return BatchManifest(frame=manifest, path=output_path)


def load_monthly_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"county_id", "crop_type", "year", "month"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly table is missing required columns: {missing}")
    out = frame.copy()
    out["county_id"] = normalize_county_id(out["county_id"])
    out["crop_type"] = out["crop_type"].astype(str).str.lower()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    return out


def write_monthly_table(frame: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        frame.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output table suffix: {suffix}")


def merge_monthly_tables(
    monthly_paths: list[Path],
    output_path: Path,
    *,
    diagnostic_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not monthly_paths:
        raise ValueError("At least one monthly table path is required.")
    frames = [load_monthly_table(path) for path in monthly_paths]
    combined = pd.concat(frames, ignore_index=True)
    key_cols = ["county_id", "crop_type", "year", "month"]
    duplicate_mask = combined.duplicated(subset=key_cols, keep=False)
    duplicate_rows = combined.loc[duplicate_mask].copy()
    if not duplicate_rows.empty:
        conflicts: list[dict[str, Any]] = []
        value_cols = [col for col in combined.columns if col not in key_cols]
        for keys, group in duplicate_rows.groupby(key_cols, dropna=False):
            unique_rows = group[value_cols].drop_duplicates()
            if len(unique_rows) > 1:
                conflicts.append(dict(zip(key_cols, keys, strict=False)))
        if conflicts:
            raise ValueError(
                "Conflicting duplicate county-year-month rows found while merging "
                f"batch outputs: {conflicts[:5]}"
            )
    merged = combined.drop_duplicates(subset=key_cols, keep="first")
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_monthly_table(merged, output_path)
    diagnostic = {
        "input_paths": [str(path) for path in monthly_paths],
        "input_rows": int(sum(len(frame) for frame in frames)),
        "output_rows": int(len(merged)),
        "duplicate_rows_removed": int(len(combined) - len(merged)),
        "county_count": int(merged["county_id"].nunique()),
        "year_count": int(merged["year"].nunique()),
        "month_count": int(merged["month"].nunique()),
    }
    if diagnostic_path is not None:
        save_json(diagnostic, diagnostic_path)
    return merged, diagnostic


def build_extraction_command(
    row: pd.Series,
    *,
    state_codes: str = "IA",
    crop: str = "Corn",
) -> str:
    return (
        "python scripts/research/cropnet_feature_forecasting_v12_server.py "
        "--full-run --extract-only "
        f"--state-codes {state_codes} --crop {crop} "
        f"--years {row['years']} --quarters {row['quarters']} "
        f"--fips-codes {row['fips_codes']} "
        f"--run-name {row['run_name']} --resume --delete-raw-after-extract"
    )


def _parse_int_tuple(values: list[str]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _parse_str_tuple(values: list[str]) -> tuple[str, ...]:
    return tuple(value.upper() for value in values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and merge Corn IA yield extraction batches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Build a label-driven FIPS batch manifest.")
    manifest.add_argument("--usda-path", type=Path, nargs="+", required=True)
    manifest.add_argument("--output-path", type=Path, required=True)
    manifest.add_argument("--crop-type", default="corn")
    manifest.add_argument("--state-fips", default="19")
    manifest.add_argument("--batch-size", type=int, default=10)
    manifest.add_argument("--years", nargs="+", default=[str(year) for year in DEFAULT_YEARS])
    manifest.add_argument("--quarters", nargs="+", default=list(DEFAULT_QUARTERS))
    manifest.add_argument("--run-prefix", default="corn_ia_gs_2017_2022")
    manifest.add_argument("--write-commands", type=Path, default=None)

    merge = subparsers.add_parser("merge", help="Merge batch monthly feature tables.")
    merge.add_argument("--monthly-path", type=Path, nargs="+", required=True)
    merge.add_argument("--output-path", type=Path, required=True)
    merge.add_argument("--diagnostic-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "manifest":
            manifest = write_batch_manifest(
                args.usda_path,
                args.output_path,
                crop_type=args.crop_type,
                state_fips=args.state_fips,
                batch_size=args.batch_size,
                years=_parse_int_tuple(args.years),
                quarters=_parse_str_tuple(args.quarters),
                run_prefix=args.run_prefix,
            )
            if args.write_commands is not None:
                commands = [
                    build_extraction_command(row)
                    for _, row in manifest.frame.iterrows()
                ]
                args.write_commands.parent.mkdir(parents=True, exist_ok=True)
                args.write_commands.write_text("\n".join(commands) + "\n", encoding="utf-8")
            print(f"Wrote manifest: {manifest.path}")
            print(f"Batches: {len(manifest.frame)}")
            return 0
        if args.command == "merge":
            _, diagnostic = merge_monthly_tables(
                args.monthly_path,
                args.output_path,
                diagnostic_path=args.diagnostic_path,
            )
            print(json.dumps(diagnostic, indent=2))
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
