"""Tests for year-based CropNet experiment splits."""

import pandas as pd
import pytest

from crop_fusion_ai.training.data_splits import build_year_based_split
from crop_fusion_ai.training.train_yield_model import create_synthetic_yield_dataframe


def test_build_year_based_split_uses_expected_years() -> None:
    """The helper should expose explicit 2017-2019/2020/2021 partitions."""
    dataframe = create_synthetic_yield_dataframe(96)

    split = build_year_based_split(dataframe, history_years=3)

    assert set(split.train["year"].unique().tolist()) == {2017, 2018, 2019}
    assert set(split.validation["year"].unique().tolist()) == {2020}
    assert set(split.test["year"].unique().tolist()) == {2021}


def test_build_year_based_split_rejects_missing_history_years() -> None:
    """The helper should fail when required preceding years are absent."""
    dataframe = pd.DataFrame(
        {
            "year": [2017, 2018, 2019, 2020, 2021],
            "yield": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )

    with pytest.raises(ValueError, match="missing required history years"):
        build_year_based_split(dataframe, history_years=3)
