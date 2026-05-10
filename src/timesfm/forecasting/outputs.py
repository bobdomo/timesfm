"""Serialization helpers for forecasting outputs."""

import dataclasses
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .quality import build_quality_warnings


@dataclasses.dataclass(frozen=True)
class ForecastResult:
  """One target-frequency-model forecast."""

  target: str
  frequency: str
  model_type: str
  input_mode: str
  forecast_dates: pd.DatetimeIndex
  point_forecast: np.ndarray
  quantile_forecast: np.ndarray


def forecast_results_to_dataframe(results: Iterable[ForecastResult]) -> pd.DataFrame:
  """Flatten forecast results into a row-based DataFrame."""
  rows = []
  for result in results:
    for idx, forecast_date in enumerate(result.forecast_dates):
      rows.append(
        {
          "forecast_target": result.target,
          "frequency": result.frequency,
          "forecast_date": forecast_date,
          "predicted_value": float(result.point_forecast[idx]),
          "lower_90": float(result.quantile_forecast[idx, 1]),
          "median": float(result.quantile_forecast[idx, 5]),
          "upper_90": float(result.quantile_forecast[idx, 9]),
          "model_type": result.model_type,
          "input_mode": result.input_mode,
        }
      )
  return pd.DataFrame(rows)


def build_run_summary(
  *,
  dataset: pd.DataFrame,
  results: Iterable[ForecastResult],
  quality_report: pd.DataFrame,
  input_mode: str,
  frequencies: Iterable[str],
  horizons: dict[str, int],
  warning_threshold: float = 0.2,
) -> dict[str, Any]:
  """Build a machine-readable summary for one forecasting run."""
  result_list = list(results)
  forecast_rows = sum(len(result.forecast_dates) for result in result_list)
  quality = {
    row["column"]: {
      "missing_count": int(row["missing_count"]),
      "missing_ratio": float(row["missing_ratio"]),
    }
    for _, row in quality_report.iterrows()
  }
  warnings = build_quality_warnings(
    quality_report,
    warning_threshold=warning_threshold,
  )
  return {
    "input_mode": input_mode,
    "row_count": int(len(dataset.index)),
    "date_range": {
      "start": pd.to_datetime(dataset["date"]).min().strftime("%Y-%m-%d"),
      "end": pd.to_datetime(dataset["date"]).max().strftime("%Y-%m-%d"),
    },
    "frequencies": list(frequencies),
    "horizons": horizons,
    "forecast_rows": forecast_rows,
    "quality": quality,
    "warnings": warnings,
  }
