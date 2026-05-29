"""List CropNet counties available in the USDA records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crop_fusion_ai.preprocessing.usda_dataset import (  # noqa: E402
    build_usda_records_from_frame,
    infer_usda_split,
    parse_usda_remote_path,
    select_usda_remote_files,
)


DEFAULT_REPO_ID = "CropNet/CropNet"
DEFAULT_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
DEFAULT_CROPS = ["corn", "cotton", "soybeans", "winter wheat"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Years to inspect. Default: 2017-2022.")
    parser.add_argument("--crops", nargs="+", default=DEFAULT_CROPS, help="Crop names to inspect.")
    parser.add_argument(
        "--target",
        choices=("yield", "production"),
        default="yield",
        help="USDA target column to use when building county records.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    return parser


def _normalize_years(values: list[int] | None) -> list[int]:
    if values:
        return sorted(dict.fromkeys(int(value) for value in values))
    return list(DEFAULT_YEARS)


def _build_summary(remote_paths: list[str], *, root: Path, target_kind: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: dict[tuple[str, str], dict[str, object]] = {}

    for remote_path in remote_paths:
        year, crop_type = parse_usda_remote_path(remote_path)
        local_path = root / remote_path
        frame = pd.read_csv(local_path)
        records = build_usda_records_from_frame(
            frame,
            crop_type=crop_type,
            year=year,
            split=infer_usda_split(year),
            target_kind=target_kind,  # type: ignore[arg-type]
            source_path=remote_path,
        )
        for record in records:
            county_id = str(record.get("county_id") or "").zfill(5)
            if not county_id or county_id == "00000":
                continue
            key = (county_id, crop_type)
            entry = seen.setdefault(
                key,
                {
                    "county_id": county_id,
                    "state_name": record.get("state_name"),
                    "county_name": record.get("county_name"),
                    "crop_type": crop_type,
                    "years": set(),
                    "source_paths": set(),
                },
            )
            if record.get("state_name") and not entry["state_name"]:
                entry["state_name"] = record.get("state_name")
            if record.get("county_name") and not entry["county_name"]:
                entry["county_name"] = record.get("county_name")
            entry["years"].add(int(year))  # type: ignore[union-attr]
            entry["source_paths"].add(remote_path)  # type: ignore[union-attr]

    for entry in seen.values():
        rows.append(
            {
                "county_id": entry["county_id"],
                "state_name": entry["state_name"],
                "county_name": entry["county_name"],
                "crop_type": entry["crop_type"],
                "years": ",".join(str(year) for year in sorted(entry["years"])),
                "source_count": len(entry["source_paths"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["county_id", "crop_type"]).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    years = _normalize_years(args.years)
    api = HfApi()
    remote_paths = api.list_repo_files(args.repo_id, repo_type="dataset")
    selected = select_usda_remote_files(remote_paths, years=years, crops=args.crops)
    if not selected:
        raise SystemExit("No USDA files matched the requested years/crops.")

    with TemporaryDirectory() as tmpdir:
        root = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=tmpdir,
                allow_patterns=selected,
                ignore_patterns=["**/.DS_Store"],
            )
        )
        summary = _build_summary(selected, root=root, target_kind=args.target)

    if summary.empty:
        raise SystemExit("No counties were found in the selected USDA files.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Wrote {len(summary)} counties to {args.output}")
    else:
        with pd.option_context("display.max_rows", 200, "display.max_columns", 20, "display.width", 160):
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
