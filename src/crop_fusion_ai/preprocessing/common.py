"""Shared preprocessing helpers for CropNet modality extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

PathLike: TypeAlias = str | Path
ImageLike: TypeAlias = np.ndarray | Image.Image | PathLike
FrameLike: TypeAlias = pd.DataFrame | PathLike


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Common metadata attached to per-month feature rows."""

    county_id: str | None = None
    crop_type: str | None = None
    year: int | None = None
    month: int | None = None


def _as_uint8_image(array: np.ndarray) -> np.ndarray:
    """Convert numeric image data to a uint8 RGB-compatible array."""
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 0.0
        min_value = float(np.nanmin(array)) if array.size else 0.0
        if min_value >= 0.0 and max_value <= 1.0:
            scaled = np.clip(array, 0.0, 1.0) * 255.0
        else:
            scaled = np.clip(array, 0.0, 255.0)
        return scaled.astype(np.uint8)

    return np.clip(array, 0, 255).astype(np.uint8)


def load_rgb_image(image_input: ImageLike) -> np.ndarray:
    """Load an RGB image as a ``(height, width, 3)`` uint8 array."""
    if isinstance(image_input, np.ndarray):
        array = image_input
        if array.ndim == 2:
            array = np.repeat(array[..., np.newaxis], 3, axis=2)
        if array.ndim != 3 or array.shape[-1] < 3:
            msg = "RGB image input must be a 2D image or a 3D RGB array"
            raise ValueError(msg)
        return _as_uint8_image(array[..., :3])

    if isinstance(image_input, Image.Image):
        return np.asarray(image_input.convert("RGB"), dtype=np.uint8)

    with Image.open(image_input) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_ndvi_array(image_input: ImageLike) -> np.ndarray:
    """Load an NDVI-style array as a floating-point 2D array."""
    if isinstance(image_input, np.ndarray):
        array = image_input
        if array.ndim == 3:
            array = array.mean(axis=2)
        if array.ndim != 2:
            msg = "NDVI input must be a 2D array or an image path"
            raise ValueError(msg)
        if np.issubdtype(array.dtype, np.integer):
            return array.astype(np.float64) / 255.0 * 2.0 - 1.0
        if float(np.nanmax(array)) > 1.5 or float(np.nanmin(array)) < -1.5:
            return array.astype(np.float64) / 255.0 * 2.0 - 1.0
        return array.astype(np.float64)

    if isinstance(image_input, Image.Image):
        return np.asarray(image_input.convert("L"), dtype=np.float64) / 255.0 * 2.0 - 1.0

    with Image.open(image_input) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0 * 2.0 - 1.0


def load_weather_frame(frame_input: FrameLike) -> pd.DataFrame:
    """Load weather records from a CSV path or return a copy of an input frame."""
    if isinstance(frame_input, pd.DataFrame):
        return frame_input.copy()
    return pd.read_csv(frame_input)


def safe_divide(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Divide two values while guarding against zero denominators."""
    if denominator == 0.0:
        return default
    return numerator / denominator


def shannon_entropy(values: np.ndarray) -> float:
    """Compute the Shannon entropy of a histogrammed signal."""
    histogram, _ = np.histogram(values.ravel(), bins=256, range=(0, 255), density=True)
    histogram = histogram[histogram > 0]
    if histogram.size == 0:
        return 0.0
    return float(-(histogram * np.log2(histogram)).sum())


def longest_true_streak(values: Sequence[bool]) -> int:
    """Return the longest consecutive run of ``True`` values."""
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def connected_component_sizes(mask: np.ndarray) -> list[int]:
    """Return connected component sizes for a binary mask using 8-neighbourhoods."""
    if mask.ndim != 2:
        msg = "Connected components require a 2D mask"
        raise ValueError(msg)

    binary = mask.astype(bool)
    visited = np.zeros_like(binary, dtype=bool)
    sizes: list[int] = []
    rows, cols = binary.shape

    for start_row, start_col in np.argwhere(binary):
        if visited[start_row, start_col]:
            continue

        stack: list[tuple[int, int]] = [(int(start_row), int(start_col))]
        visited[start_row, start_col] = True
        area = 0

        while stack:
            row, col = stack.pop()
            area += 1
            for next_row in range(max(0, row - 1), min(rows, row + 2)):
                for next_col in range(max(0, col - 1), min(cols, col + 2)):
                    if next_row == row and next_col == col:
                        continue
                    if binary[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

        sizes.append(area)

    return sizes


def boundary_pixel_count(mask: np.ndarray) -> int:
    """Approximate the boundary pixel count of a binary region."""
    if mask.ndim != 2:
        msg = "Boundary extraction requires a 2D mask"
        raise ValueError(msg)

    binary = mask.astype(bool)
    if not binary.any():
        return 0

    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    boundary = np.zeros_like(binary, dtype=bool)
    rows, cols = binary.shape

    for row in range(rows):
        for col in range(cols):
            if not binary[row, col]:
                continue
            window = padded[row : row + 3, col : col + 3]
            boundary[row, col] = not bool(window.all())

    return int(boundary.sum())


def edge_density(gray: np.ndarray) -> float:
    """Estimate edge density with a simple edge filter."""
    image = Image.fromarray(gray.astype(np.uint8), mode="L")
    edges = np.asarray(image.filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
    return float((edges > 12).mean())


def grayscale_contrast(gray: np.ndarray) -> float:
    """Return the normalized standard deviation of a grayscale image."""
    return float(np.std(gray.astype(np.float64)) / 255.0)


def combine_modality_feature_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Outer-join modality feature frames on the shared CropNet metadata keys."""
    valid_frames = [frame.copy() for frame in frames if not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    for frame in valid_frames:
        for column in ("county_id", "crop_type", "year", "month"):
            if column not in frame.columns:
                frame[column] = pd.NA

    metadata_columns = ["county_id", "crop_type", "year", "month"]
    return reduce(
        lambda left, right: pd.merge(left, right, on=metadata_columns, how="outer"),
        valid_frames,
    )


def aggregate_monthly_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate monthly rows by averaging numeric feature columns."""
    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    metadata_columns = ["county_id", "crop_type", "year", "month"]
    for column in metadata_columns:
        if column not in working.columns:
            working[column] = pd.NA

    numeric_columns = [
        column
        for column in working.columns
        if column not in metadata_columns and pd.api.types.is_numeric_dtype(working[column])
    ]
    if not numeric_columns:
        return working.loc[:, metadata_columns].drop_duplicates().reset_index(drop=True)

    aggregated = (
        working.groupby(metadata_columns, dropna=False, as_index=False)[numeric_columns]
        .mean(numeric_only=True)
        .sort_values(metadata_columns)
        .reset_index(drop=True)
    )
    return aggregated
