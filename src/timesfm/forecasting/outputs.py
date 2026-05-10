"""Serialization helpers for forecasting outputs."""

import dataclasses
from collections.abc import Iterable

import numpy as np
import pandas as pd


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
