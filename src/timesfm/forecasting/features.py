"""Feature preparation helpers for calendar and low-frequency covariates."""

from collections.abc import Sequence

import pandas as pd


def add_known_calendar_covariates(
  frame: pd.DataFrame,
  *,
  malaysia_holidays: pd.DataFrame | None = None,
  winter_months: Sequence[int] = (11, 12, 1, 2),
) -> pd.DataFrame:
  """Add known-in-advance calendar covariates to a dated frame."""
  output = frame.copy()
  output["date"] = pd.to_datetime(output["date"])
  output["is_jeddah_winter"] = output["date"].dt.month.isin(winter_months)
  output["is_malaysia_holiday"] = False
  output["holiday_name"] = None

  if malaysia_holidays is not None and not malaysia_holidays.empty:
    holiday_frame = malaysia_holidays.copy()
    holiday_frame["date"] = pd.to_datetime(holiday_frame["date"])
    holiday_frame = holiday_frame.rename(columns={"holiday_name": "holiday_name"})
    holiday_frame["is_malaysia_holiday"] = True
    output = output.merge(
      holiday_frame[["date", "is_malaysia_holiday", "holiday_name"]],
      on="date",
      how="left",
      suffixes=("", "_holiday"),
    )
    output["is_malaysia_holiday"] = output["is_malaysia_holiday_holiday"].fillna(
      output["is_malaysia_holiday"]
    )
    output["holiday_name"] = output["holiday_name_holiday"].combine_first(
      output["holiday_name"]
    )
    output = output.drop(
      columns=["is_malaysia_holiday_holiday", "holiday_name_holiday"]
    )

  return output


def align_low_frequency_covariates(
  daily_frame: pd.DataFrame,
  low_frequency_frame: pd.DataFrame,
  *,
  value_columns: Sequence[str],
) -> pd.DataFrame:
  """Align lower-frequency covariates onto a daily calendar by forward fill."""
  daily = daily_frame.copy()
  daily["date"] = pd.to_datetime(daily["date"])
  low_frequency = low_frequency_frame.copy()
  low_frequency["date"] = pd.to_datetime(low_frequency["date"])

  merged = daily.merge(
    low_frequency[["date", *value_columns]],
    on="date",
    how="left",
  ).sort_values("date")
  merged[list(value_columns)] = merged[list(value_columns)].ffill()
  return merged


def align_daily_covariates(
  daily_frame: pd.DataFrame,
  daily_covariate_frame: pd.DataFrame,
  *,
  value_columns: Sequence[str],
) -> pd.DataFrame:
  """Align daily covariates onto a daily frame by date and forward fill."""
  daily = daily_frame.copy()
  daily["date"] = pd.to_datetime(daily["date"])
  covariates = daily_covariate_frame.copy()
  covariates["date"] = pd.to_datetime(covariates["date"])

  merged = daily.merge(
    covariates[["date", *value_columns]],
    on="date",
    how="left",
  ).sort_values("date")
  merged[list(value_columns)] = merged[list(value_columns)].ffill()
  return merged
