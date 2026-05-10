"""Data-quality reporting helpers for forecasting datasets."""

from collections.abc import Sequence
from typing import Any

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


def build_quality_warnings(
  quality_report: pd.DataFrame,
  *,
  warning_threshold: float = 0.2,
) -> list[dict[str, Any]]:
  """Convert missingness metrics into explicit warnings."""
  warnings: list[dict[str, Any]] = []
  for _, row in quality_report.iterrows():
    if float(row["missing_ratio"]) >= warning_threshold:
      warnings.append(
        {
          "column": row["column"],
          "severity": "warning",
          "message": (
            f"Missing ratio {float(row['missing_ratio']):.2f} exceeds threshold "
            f"{warning_threshold:.2f}."
          ),
        }
      )
  return warnings
