"""Download a small CropNet subset for one county and a 2017-2022 window."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_REPO_ID = "CropNet/CropNet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sample_data"
DEFAULT_CACHE_DIR = ROOT / "data" / "cache" / "cropnet_fetch"
_STATE_ABBR_BY_CODE = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


def _get_state_abbr(state_code: str) -> str:
    try:
        return _STATE_ABBR_BY_CODE[str(state_code).zfill(2)]
    except KeyError as exc:
        raise SystemExit(f"Unsupported county FIPS state code: {state_code!r}") from exc


def _parse_years(values: list[int] | None) -> list[int]:
    if values:
        return sorted(dict.fromkeys(int(value) for value in values))
    return [2017, 2018, 2019, 2020, 2021, 2022]


def _build_allow_patterns(*, crop: str, years: list[int], state_abbr: str) -> list[str]:
    crop_folder, crop_file = {
        "corn": ("Corn", "Corn"),
        "cotton": ("Cotton", "Cotton"),
        "soybeans": ("Soybeans", "Soybean"),
        "soybean": ("Soybeans", "Soybean"),
        "winter wheat": ("WinterWheat", "WinterWheat"),
        "winterwheat": ("WinterWheat", "WinterWheat"),
        "winter_wheat": ("WinterWheat", "WinterWheat"),
    }[crop.strip().lower()]

    patterns = [
        f"USDA Crop Dataset/{crop_folder}/{year}/USDA_{crop_file}_County_{year}.csv"
        for year in years
    ]
    for year in years:
        patterns.append(f"Sentinel-2 Imagery/data/AG/{year}/{state_abbr}/*.h5")
        patterns.append(f"Sentinel-2 Imagery/data/NDVI/{year}/{state_abbr}/*.h5")
        patterns.append(f"WRF-HRRR Computed Dataset/data/{year}/{state_abbr}/*.csv")
    return patterns


def _matches_any_pattern(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--county-id", default="01003", help="Five-digit county FIPS code.")
    parser.add_argument("--crop", default="corn", help="Crop name used by the CropNet dataset.")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="Years to download. Default: 2017-2022.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory for the download.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory.")
    parser.add_argument("--keep-cache", action="store_true", help="Keep the temporary cache directory.")
    parser.add_argument("--list-only", action="store_true", help="Print the matched files and exit without downloading.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    county_id = str(args.county_id).strip().zfill(5)
    if len(county_id) != 5 or not county_id.isdigit():
        raise SystemExit("--county-id must be a five-digit FIPS code.")

    years = _parse_years(args.years)
    state_abbr = _get_state_abbr(county_id[:2])
    allow_patterns = _build_allow_patterns(crop=args.crop, years=years, state_abbr=state_abbr)

    api = HfApi()
    repo_files = api.list_repo_files(args.repo_id, repo_type="dataset")
    matched_files = sorted(path for path in repo_files if _matches_any_pattern(path, allow_patterns))
    if not matched_files:
        raise SystemExit("No repository files matched the requested county/year filter.")

    print("Matched files:")
    for path in matched_files:
        print(f"  {path}")
    print(f"Total matched files: {len(matched_files)}")
    print(
        "Note: CropNet stores AG/NDVI as large state-level Sentinel-2 HDF5 files, "
        "so a single county can still require very large downloads."
    )

    if args.list_only:
        return

    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    staging_path: Path
    if args.keep_cache:
        staging_path = cache_dir / "staging"
        staging_path.mkdir(parents=True, exist_ok=True)
        downloaded = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=staging_path,
            cache_dir=str(cache_dir / "huggingface"),
            allow_patterns=allow_patterns,
            ignore_patterns=["**/.DS_Store"],
        )
        staging_path = Path(downloaded)
    else:
        with TemporaryDirectory(dir=cache_dir) as staging_dir:
            downloaded = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=staging_dir,
                cache_dir=str(cache_dir / "huggingface"),
                allow_patterns=allow_patterns,
                ignore_patterns=["**/.DS_Store"],
            )
            staging_path = Path(downloaded)
            for source in staging_path.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(staging_path)
                target = output_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    if args.keep_cache:
        for source in staging_path.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(staging_path)
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    manifest = {
        "repo_id": args.repo_id,
        "county_id": county_id,
        "crop": args.crop,
        "years": years,
        "state_abbr": state_abbr,
        "allow_patterns": allow_patterns,
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Downloaded CropNet subset for county {county_id} into {output_dir}")
    if args.keep_cache:
        print(f"Cache retained at {cache_dir}")


if __name__ == "__main__":
    main()
