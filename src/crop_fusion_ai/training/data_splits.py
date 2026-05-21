"""Year-based dataset split helpers for CropNet-style experiments."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class YearSplit:
    """Container for explicit train, validation, and test dataframes."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def build_year_based_split(
    dataframe: pd.DataFrame,
    *,
    year_column: str = "year",
    train_years: tuple[int, ...] = (2017, 2018, 2019),
    validation_year: int = 2020,
    test_year: int = 2021,
    history_years: int = 3,
) -> YearSplit:
    """Create train/validation/test splits with an optional history constraint."""
    if year_column not in dataframe.columns:
        msg = f"Year column '{year_column}' is missing from dataframe"
        raise ValueError(msg)

    year_series = pd.to_numeric(dataframe[year_column], errors="coerce")
    if year_series.isna().any():
        msg = f"Year column '{year_column}' must contain valid integer years"
        raise ValueError(msg)

    normalized_df = dataframe.copy()
    normalized_df[year_column] = year_series.astype(int)
    available_years = {
        int(year) for year in normalized_df[year_column].unique().tolist()
    }
    _validate_history_requirements(
        available_years=available_years,
        target_years=(*train_years, validation_year, test_year),
        history_years=history_years,
    )

    train_df = normalized_df[normalized_df[year_column].isin(train_years)].copy()
    validation_df = normalized_df[normalized_df[year_column] == validation_year].copy()
    test_df = normalized_df[normalized_df[year_column] == test_year].copy()
    _validate_split_non_empty(train_df, "train")
    _validate_split_non_empty(validation_df, "validation")
    _validate_split_non_empty(test_df, "test")

    return YearSplit(train=train_df, validation=validation_df, test=test_df)


def _validate_history_requirements(
    *,
    available_years: set[int],
    target_years: tuple[int, ...],
    history_years: int,
) -> None:
    """Ensure each target year has the required number of previous years present."""
    if history_years < 0:
        msg = "history_years must be zero or greater"
        raise ValueError(msg)

    for target_year in target_years:
        required_years = {
            target_year - offset for offset in range(1, history_years + 1)
        }
        if not required_years.issubset(available_years):
            missing_years = sorted(required_years - available_years)
            msg = (
                f"Target year {target_year} is missing required history years: "
                f"{missing_years}"
            )
            raise ValueError(msg)


def _validate_split_non_empty(dataframe: pd.DataFrame, split_name: str) -> None:
    """Fail clearly when a requested split has no rows."""
    if dataframe.empty:
        msg = f"The {split_name} split is empty after year filtering"
        raise ValueError(msg)
