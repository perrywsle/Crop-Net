"""Tests for CropNet download planning helpers."""

from __future__ import annotations

from fetch_data import build_usda_allow_patterns


def test_build_usda_allow_patterns_keeps_labels_decoupled_from_modalities() -> None:
    """USDA-only fetches should not include AG, NDVI, or weather chunks."""
    patterns = build_usda_allow_patterns(crop="corn", years=[2017, 2022])

    assert patterns == [
        "USDA Crop Dataset/Corn/2017/USDA_Corn_County_2017.csv",
        "USDA Crop Dataset/Corn/2022/USDA_Corn_County_2022.csv",
    ]
    assert not any("Sentinel-2 Imagery" in pattern for pattern in patterns)
    assert not any("WRF-HRRR" in pattern for pattern in patterns)
