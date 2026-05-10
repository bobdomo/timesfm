"""Data-quality reporting helpers for forecasting datasets."""

from collections.abc import Sequence

import pandas as pd


def build_data_quality_report(
  frame: pd.DataFrame,
  *,
  tracked_columns: Sequence[str],
) -> pd.DataFrame:
  """Summarize per-column missingness for tracked forecast inputs."""
  rows = []
  total_rows = len(frame.index)
  for column in tracked_columns:
    missing_count = int(frame[column].isna().sum()) if column in frame.columns else total_rows
    rows.append(
      {
        "column": column,
        "missing_count": missing_count,
        "missing_ratio": (missing_count / total_rows) if total_rows else 0.0,
      }
    )
  return pd.DataFrame(rows)
