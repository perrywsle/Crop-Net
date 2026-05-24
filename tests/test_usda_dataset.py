"""Tests for USDA tutorial dataset selection and JSONL assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crop_fusion_ai.preprocessing import (
    build_usda_records_from_frame,
    infer_usda_split,
    normalize_crop_type,
    parse_usda_remote_path,
    select_usda_remote_files,
    write_jsonl_records,
)


@pytest.mark.parametrize(
    ("year", "expected_split"),
    [
        (2017, "train"),
        (2020, "train"),
        (2021, "validation"),
        (2022, "test"),
    ],
)
def test_infer_usda_split_maps_the_requested_year_buckets(
    year: int,
    expected_split: str,
) -> None:
    """The split mapping should match the requested 2017-2022 window."""
    assert infer_usda_split(year) == expected_split


def test_select_usda_remote_files_keeps_only_requested_usda_slices() -> None:
    """The file filter should ignore stats files and irrelevant years/crops."""
    remote_paths = [
        "USDA Crop Dataset/Corn/2017/USDA_Corn_County_2017.csv",
        "USDA Crop Dataset/Soybeans/2021/USDA_Soybean_County_2021.csv",
        "USDA Crop Dataset/WinterWheat/2022/USDA_WinterWheat_County_2022.csv",
        "USDA/stats/corn_counties.csv",
        "README.md",
    ]

    selected = select_usda_remote_files(
        remote_paths,
        years=[2017, 2021],
        crops=["corn", "soybeans"],
    )

    assert selected == [
        "USDA Crop Dataset/Corn/2017/USDA_Corn_County_2017.csv",
        "USDA Crop Dataset/Soybeans/2021/USDA_Soybean_County_2021.csv",
    ]


def test_parse_usda_remote_path_normalizes_winter_wheat_label() -> None:
    """Winter wheat paths should normalize to the canonical crop label."""
    year, crop_type = parse_usda_remote_path(
        "USDA Crop Dataset/WinterWheat/2022/USDA_WinterWheat_County_2022.csv"
    )

    assert year == 2022
    assert crop_type == "winter wheat"
    assert normalize_crop_type("Winter_Wheat") == "winter wheat"


def test_build_usda_records_from_frame_emits_jsonl_ready_rows(tmp_path: Path) -> None:
    """County-total USDA rows should be converted into compact training records."""
    frame = pd.DataFrame(
        [
            {
                "state_ansi": 1,
                "county_ansi": 3,
                "state_name": "ALABAMA",
                "county_name": "BALDWIN",
                "commodity_desc": "CORN",
                "agg_level_desc": "COUNTY",
                "domain_desc": "TOTAL",
                "source_desc": "SURVEY",
                "YIELD, MEASURED IN BU / ACRE": 162.5,
                "extra_signal": "keep me",
            },
            {
                "state_ansi": 1,
                "county_ansi": 5,
                "state_name": "ALABAMA",
                "county_name": "OTHER",
                "commodity_desc": "CORN",
                "agg_level_desc": "STATE",
                "domain_desc": "TOTAL",
                "source_desc": "SURVEY",
                "YIELD, MEASURED IN BU / ACRE": 155.0,
                "extra_signal": "drop me",
            },
        ]
    )

    records = build_usda_records_from_frame(
        frame,
        crop_type="corn",
        year=2017,
        split="train",
        target_kind="yield",
        source_path="USDA/data/2017/Corn/USDA_Corn_County_2017.csv",
    )

    assert len(records) == 1
    record = records[0]
    assert record["split"] == "train"
    assert record["county_id"] == "01003"
    assert record["crop_type"] == "corn"
    assert record["year"] == 2017
    assert record["target_kind"] == "yield"
    assert record["target_value"] == pytest.approx(162.5)
    assert record["target_unit"] == "BU / ACRE"
    assert record["state_ansi"] == "01"
    assert record["county_ansi"] == "003"
    assert record["source_desc"] == "SURVEY"
    assert record["features"]["extra_signal"] == "keep me"

    output_path = tmp_path / "records.jsonl"
    written = write_jsonl_records(records, output_path)
    assert written == 1

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["county_id"] == "01003"
    assert payload["features"]["extra_signal"] == "keep me"
