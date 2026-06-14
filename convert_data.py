"""Convert downloaded CropNet data into a GUI-ready ``test_data`` tree."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # pragma: no cover - optional dependency
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - surfaced at runtime
    h5py = None
    _H5PY_IMPORT_ERROR = exc
else:  # pragma: no cover - import availability is environment-specific
    _H5PY_IMPORT_ERROR = None


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".feather"}
DEFAULT_SOURCE_DIR = ROOT / "data" / "sample_data"
DEFAULT_OUTPUT_DIR = ROOT / "test_data"
DEFAULT_DEMO_ROOT = ROOT / "data" / "raw"
DATE_PATTERNS = (
    re.compile(r"(?P<year>20\d{2})[._-](?P<month>\d{1,2})[._-](?P<day>\d{1,2})"),
    re.compile(r"(?P<year>20\d{2})[._-](?P<month>\d{1,2})"),
    re.compile(r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})"),
)


@dataclass(frozen=True, slots=True)
class PlannedArtifact:
    """One output file planned by the converter."""

    source: str
    destination: str
    modality: str
    origin: str
    transform: str
    source_key: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Downloaded CropNet folder to convert.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination GUI sample folder.",
    )
    parser.add_argument(
        "--demo-root",
        type=Path,
        default=DEFAULT_DEMO_ROOT,
        help="Fallback root for bundled demo assets.",
    )
    parser.add_argument(
        "--no-demo-fallback",
        action="store_true",
        help="Require all GUI modalities to be present in the source tree.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned conversion without writing any files.",
    )
    return parser


def _infer_modality(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "ag" in parts:
        return "ag"
    if "ndvi" in parts:
        return "ndvi"
    if "wrf-hrrr computed dataset" in parts or "hrrr" in parts:
        return "weather"
    if "weather" in parts:
        return "weather"
    joined = " / ".join(part.lower() for part in path.parts)
    if "usda crop dataset" in joined or any(part.lower() == "usda" for part in path.parts):
        return "usda"
    return None


def _is_cache_junk(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return ".cache" in parts or path.suffix.lower() in {".lock", ".metadata"}


def _infer_date_tokens(path: Path) -> tuple[int | None, int | None, int | None]:
    for candidate in (path.stem, path.name, *path.parts):
        for pattern in DATE_PATTERNS:
            match = pattern.search(candidate)
            if match is None:
                continue
            year = int(match.group("year"))
            month = int(match.group("month"))
            day = match.groupdict().get("day")
            return year, month, int(day) if day else None
    return None, None, None


def _format_date_name(path: Path, modality: str, *, year: int | None, month: int | None, day: int | None) -> str:
    suffix = path.suffix.lower()
    if modality in {"ag", "ndvi"}:
        if year is not None and month is not None and day is not None:
            return f"{year:04d}_{month:02d}_{day:02d}{suffix}"
        if year is not None and month is not None:
            return f"{year:04d}_{month:02d}{suffix}"
    if modality == "weather" and year is not None and month is not None:
        return f"{year:04d}_{month:02d}{suffix or '.csv'}"
    return path.name


def _destination_for_source(
    source_root: Path,
    file_path: Path,
    *,
    modality: str,
    output_root: Path,
) -> Path:
    if modality == "usda":
        return output_root / "usda" / file_path.relative_to(source_root)

    year, month, day = _infer_date_tokens(file_path)
    filename = _format_date_name(file_path, modality, year=year, month=month, day=day)
    return output_root / modality / filename


def _destination_from_h5_key(output_root: Path, modality: str, county_id: str, date_key: str) -> Path:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", date_key).strip("_")
    return output_root / modality / f"{county_id}_{normalized}.png"


def _ensure_unique_destination(destination: Path, occupied: set[Path]) -> Path:
    if destination not in occupied and not destination.exists():
        occupied.add(destination)
        return destination

    suffix = destination.suffix
    stem = destination.stem
    parent = destination.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if candidate not in occupied and not candidate.exists():
            occupied.add(candidate)
            return candidate
        index += 1


def _collect_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Source directory not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and not _is_cache_junk(path))


def _demo_file_for_modality(modality: str, demo_root: Path) -> Path:
    if modality == "ag":
        return demo_root / "images" / "demo_ag.png"
    if modality == "ndvi":
        return demo_root / "images" / "demo_ndvi.png"
    if modality == "weather":
        return demo_root / "tabular" / "demo_weather.csv"
    raise KeyError(f"Unsupported modality: {modality}")


def _normalize_ag_preview(array: np.ndarray) -> Image.Image:
    working = np.asarray(array)
    while working.ndim > 3:
        working = working.mean(axis=0)
    if working.ndim == 3 and working.shape[-1] not in {1, 3}:
        working = working.mean(axis=-1)
    if working.ndim == 3 and working.shape[-1] >= 3:
        working = working[..., :3].astype(np.float64)
        if np.nanmax(working) <= 1.0:
            working = working * 255.0
        working = np.clip(working, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(working, mode="RGB")
    if working.ndim == 2:
        working = np.clip(working.astype(np.float64), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(working, mode="L").convert("RGB")
    if working.ndim == 1:
        working = np.tile(working, (working.shape[0], 1)).astype(np.uint8)
        return Image.fromarray(working, mode="L").convert("RGB")
    raise ValueError("Unsupported AG HDF5 array shape")


def _normalize_ndvi_preview(array: np.ndarray) -> Image.Image:
    working = np.asarray(array)
    while working.ndim > 2:
        if working.ndim == 3 and working.shape[-1] in {1, 3}:
            working = working.mean(axis=-1)
        else:
            working = working.mean(axis=0)
    working = working.astype(np.float64)
    if np.nanmin(working) < -1.5 or np.nanmax(working) > 1.5:
        working = np.clip(working, 0.0, 255.0) / 255.0 * 2.0 - 1.0
    working = np.nan_to_num(working, nan=0.0)
    working = np.clip((working + 1.0) / 2.0, 0.0, 1.0) * 255.0
    return Image.fromarray(working.astype(np.uint8), mode="L")


def plan_conversion(
    source_root: Path,
    output_root: Path,
    *,
    demo_root: Path = DEFAULT_DEMO_ROOT,
    allow_demo_fallback: bool = True,
) -> list[PlannedArtifact]:
    """Build a file-by-file plan for the GUI-ready tree."""
    files = _collect_files(source_root)
    modality_groups: dict[str, list[Path]] = {"ag": [], "ndvi": [], "weather": [], "usda": []}

    for path in files:
        modality = _infer_modality(path)
        if modality is None:
            continue

        suffix = path.suffix.lower()
        if modality in {"ag", "ndvi"} and suffix in SUPPORTED_IMAGE_SUFFIXES:
            modality_groups[modality].append(path)
        elif modality in {"ag", "ndvi"} and suffix == ".h5":
            modality_groups[modality].append(path)
        elif modality == "weather" and suffix in SUPPORTED_TABLE_SUFFIXES:
            modality_groups[modality].append(path)
        elif modality == "usda":
            modality_groups["usda"].append(path)

    if allow_demo_fallback:
        for modality in ("ag", "ndvi", "weather"):
            if modality_groups[modality]:
                continue
            demo_path = _demo_file_for_modality(modality, demo_root)
            if demo_path.exists():
                modality_groups[modality].append(demo_path)

    demo_paths = {modality: _demo_file_for_modality(modality, demo_root) for modality in ("ag", "ndvi", "weather")}
    missing_required = [modality for modality in ("ag", "ndvi", "weather") if not modality_groups[modality]]
    if missing_required:
        raise SystemExit(
            "No GUI-ready files were found for: "
            + ", ".join(missing_required)
            + ". Re-run with demo fallback enabled or provide those modalities in the source tree."
        )

    planned: list[PlannedArtifact] = []
    occupied: set[Path] = set()

    for modality, paths in modality_groups.items():
        for source_path in paths:
            if modality == "usda":
                destination = _destination_for_source(source_root, source_path, modality=modality, output_root=output_root)
                destination = _ensure_unique_destination(destination, occupied)
                planned.append(
                    PlannedArtifact(
                        source=str(source_path),
                        destination=str(destination),
                        modality=modality,
                        origin="source",
                        transform="copy",
                    )
                )
                continue

            if source_path in demo_paths.values():
                destination = output_root / modality / source_path.name
                destination = _ensure_unique_destination(destination, occupied)
                planned.append(
                    PlannedArtifact(
                        source=str(source_path),
                        destination=str(destination),
                        modality=modality,
                        origin="demo",
                        transform="copy",
                    )
                )
                continue

            if source_path.suffix.lower() == ".h5":
                if h5py is None:
                    raise ModuleNotFoundError("h5py is required to convert AG/NDVI HDF5 files") from _H5PY_IMPORT_ERROR

                with h5py.File(source_path, "r") as handle:
                    for county_id in sorted(handle.keys()):
                        county_group = handle[county_id]
                        for date_key in sorted(county_group.keys()):
                            destination = _destination_from_h5_key(output_root, modality, county_id, date_key)
                            destination = _ensure_unique_destination(destination, occupied)
                            planned.append(
                                PlannedArtifact(
                                    source=str(source_path),
                                    destination=str(destination),
                                    modality=modality,
                                    origin="source",
                                    transform="h5_preview",
                                    source_key=f"{county_id}/{date_key}",
                                )
                            )
                continue

            destination = _destination_for_source(source_root, source_path, modality=modality, output_root=output_root)
            destination = _ensure_unique_destination(destination, occupied)
            planned.append(
                PlannedArtifact(
                    source=str(source_path),
                    destination=str(destination),
                    modality=modality,
                    origin="source",
                    transform="copy",
                )
            )

    return planned


def execute_plan(planned: list[PlannedArtifact], output_root: Path) -> dict[str, object]:
    """Write the planned files and return manifest metadata."""
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {"ag": 0, "ndvi": 0, "weather": 0, "usda": 0}
    entries: list[dict[str, object]] = []

    for item in planned:
        source = Path(item.source)
        destination = Path(item.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if item.transform == "copy":
            shutil.copy2(source, destination)
        elif item.transform == "h5_preview":
            if h5py is None:
                raise ModuleNotFoundError("h5py is required to convert AG/NDVI HDF5 files") from _H5PY_IMPORT_ERROR
            if not item.source_key:
                raise ValueError("Missing HDF5 source key for preview conversion")
            county_id, date_key = item.source_key.split("/", 1)
            with h5py.File(source, "r") as handle:
                grid = np.asarray(handle[county_id][date_key]["data"])
            preview = _normalize_ag_preview(grid) if item.modality == "ag" else _normalize_ndvi_preview(grid)
            preview.save(destination)
        else:
            raise ValueError(f"Unsupported transform: {item.transform}")

        counts[item.modality] = counts.get(item.modality, 0) + 1
        entries.append(asdict(item))

    manifest = {
        "output_dir": str(output_root),
        "counts": counts,
        "files": entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    demo_root = args.demo_root.expanduser().resolve()

    planned = plan_conversion(
        source_root,
        output_root,
        demo_root=demo_root,
        allow_demo_fallback=not args.no_demo_fallback,
    )

    print("Planned files:")
    for item in planned:
        print(f"  [{item.modality:7}] {item.origin:5} {item.transform:11} {item.source} -> {item.destination}")
    print(f"Total files: {len(planned)}")

    if args.dry_run:
        return

    manifest = execute_plan(planned, output_root)
    print(f"Wrote GUI-ready data into {output_root}")
    print("Counts: " + ", ".join(f"{name}={count}" for name, count in manifest["counts"].items()))


if __name__ == "__main__":
    main()
