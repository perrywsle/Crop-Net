"""Tests for label-driven Corn IA extraction batching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cropnet_forecasting.yield_batching import (
    build_batch_manifest,
    derive_label_fips,
    merge_monthly_tables,
)


def _write_usda(path: Path) -> None:
    rows = [
        {
            "state_ansi": 19,
            "county_ansi": 1,
            "year": 2021,
            "YIELD, MEASURED IN BU / ACRE": 180.0,
        },
        {
            "state_ansi": 19,
            "county_ansi": 3,
            "year": 2022,
            "YIELD, MEASURED IN BU / ACRE": 190.0,
        },
        {
            "state_ansi": 55,
            "county_ansi": 1,
            "year": 2022,
            "YIELD, MEASURED IN BU / ACRE": 120.0,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _monthly_row(county_id: str, value: float) -> dict[str, object]:
    return {
        "county_id": county_id,
        "crop_type": "corn",
        "year": 2022,
        "month": 4,
        "ag_green_pixel_ratio": value,
    }


def test_derive_label_fips_builds_valid_iowa_codes(tmp_path: Path) -> None:
    usda_path = tmp_path / "USDA_Corn_County_2022.csv"
    _write_usda(usda_path)

    assert derive_label_fips([usda_path], crop_type="corn") == ["19001", "19003"]


def test_build_batch_manifest_chunks_fips_and_commands() -> None:
    manifest = build_batch_manifest(
        ["19009", "19001", "19003"],
        batch_size=2,
        years=(2021, 2022),
        quarters=("Q2", "Q3"),
        run_prefix="corn_ia_test",
    )

    assert manifest["fips_codes"].tolist() == ["19001 19003", "19009"]
    assert manifest["years"].tolist() == ["2021 2022", "2021 2022"]
    assert manifest["quarters"].tolist() == ["Q2 Q3", "Q2 Q3"]
    assert manifest["run_name"].tolist() == [
        "corn_ia_test_batch_001",
        "corn_ia_test_batch_002",
    ]


def test_merge_monthly_tables_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([_monthly_row("19001", 0.1)]).to_csv(first, index=False)
    pd.DataFrame([_monthly_row("19001", 0.2)]).to_csv(second, index=False)

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        merge_monthly_tables([first, second], tmp_path / "merged.parquet")


def test_merge_monthly_tables_deduplicates_identical_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "merged.parquet"
    diagnostic = tmp_path / "merge_diagnostic.json"
    pd.DataFrame([_monthly_row("19001", 0.1)]).to_csv(first, index=False)
    pd.DataFrame([_monthly_row("19001", 0.1), _monthly_row("19003", 0.3)]).to_csv(
        second,
        index=False,
    )

    merged, report = merge_monthly_tables([first, second], output, diagnostic_path=diagnostic)

    assert output.exists()
    assert diagnostic.exists()
    assert len(merged) == 2
    assert report["duplicate_rows_removed"] == 1
