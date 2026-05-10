"""Canonical daily table helpers."""

import pandas as pd

from .schema import ForecastDataSpec


def build_canonical_daily_table(
  df: pd.DataFrame,
  spec: ForecastDataSpec,
) -> pd.DataFrame:
  """Add a daily calendar spine and explicit missingness flags."""
  if df.empty:
    return df.copy()

  working = df.copy()
  working["date"] = pd.to_datetime(working["date"])
  working = working.sort_values("date").reset_index(drop=True)

  spine = pd.DataFrame(
    {"date": pd.date_range(working["date"].min(), working["date"].max(), freq="D")}
  )
  merged = spine.merge(working, on="date", how="left")

  tracked_columns = list(spec.target_columns) + list(spec.covariate_columns)
  for column in tracked_columns:
    if column in merged.columns:
      merged[f"{column}_is_missing"] = merged[column].isna()

  return merged
