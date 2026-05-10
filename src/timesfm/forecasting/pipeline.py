"""Orchestrate TimesFM forecasting runs over canonical tables."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

from .aggregation import aggregate_by_frequency
from .canonical import build_canonical_daily_table
from .outputs import ForecastResult
from .schema import ForecastDataSpec


@dataclasses.dataclass(frozen=True)
class PipelineRunConfig:
  """Configuration for a forecasting run."""

  horizons: Mapping[str, int]
  frequencies: Sequence[str] = ("daily", "weekly", "monthly", "yearly")


class ForecastBackend(Protocol):
  """Backend protocol for TimesFM-like forecast engines."""

  def forecast(
    self,
    horizon: int,
    inputs: list[np.ndarray],
  ) -> tuple[np.ndarray, np.ndarray]:
    ...

  def forecast_with_covariates(
    self,
    horizon: int,
    inputs: list[np.ndarray],
    dynamic_numerical_covariates: Mapping[str, list[np.ndarray]] | None = None,
    dynamic_categorical_covariates: Mapping[str, list[np.ndarray]] | None = None,
    static_numerical_covariates: Mapping[str, Sequence[float]] | None = None,
    static_categorical_covariates: Mapping[str, Sequence[str]] | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    ...


_FREQUENCY_TO_DATE_RANGE = {
  "daily": "D",
  "weekly": "W",
  "monthly": "MS",
  "yearly": "YS",
}


class TimesFMForecastingPipeline:
  """Run baseline and covariate forecasts for all targets and frequencies."""

  def __init__(
    self,
    backend: ForecastBackend,
    spec: ForecastDataSpec,
    config: PipelineRunConfig,
  ) -> None:
    self.backend = backend
    self.spec = spec
    self.config = config

  def run(
    self,
    dataset: pd.DataFrame,
    input_mode: str = "single_csv",
  ) -> list[ForecastResult]:
    canonical = build_canonical_daily_table(dataset, self.spec)
    results: list[ForecastResult] = []

    for frequency in self.config.frequencies:
      horizon = self.config.horizons[frequency]
      table = (
        canonical
        if frequency == "daily"
        else aggregate_by_frequency(canonical, frequency, self.spec)
      )
      if table.empty:
        continue
      forecast_dates = self._future_dates(table["date"], frequency, horizon)
      covariates = self._future_covariates(table, horizon)

      for target in self.spec.target_columns:
        target_values = table[target].dropna().to_numpy(dtype=np.float32)
        baseline_point, baseline_quantiles = self.backend.forecast(
          horizon=horizon,
          inputs=[target_values],
        )
        results.append(
          ForecastResult(
            target=target,
            frequency=frequency,
            model_type="baseline",
            input_mode=input_mode,
            forecast_dates=forecast_dates,
            point_forecast=baseline_point[0],
            quantile_forecast=baseline_quantiles[0],
          )
        )

        cov_point, cov_quantiles = self.backend.forecast_with_covariates(
          horizon=horizon,
          inputs=[target_values],
          dynamic_numerical_covariates=covariates["numerical"],
          dynamic_categorical_covariates=covariates["categorical"],
        )
        results.append(
          ForecastResult(
            target=target,
            frequency=frequency,
            model_type="covariates",
            input_mode=input_mode,
            forecast_dates=forecast_dates,
            point_forecast=cov_point[0],
            quantile_forecast=cov_quantiles[0],
          )
        )

    return results

  def _future_dates(
    self,
    dates: pd.Series,
    frequency: str,
    horizon: int,
  ) -> pd.DatetimeIndex:
    last_date = pd.to_datetime(dates.iloc[-1])
    freq = _FREQUENCY_TO_DATE_RANGE[frequency]
    return pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]

  def _future_covariates(
    self,
    table: pd.DataFrame,
    horizon: int,
  ) -> dict[str, dict[str, list[np.ndarray]]]:
    numerical: dict[str, list[np.ndarray]] = {}
    categorical: dict[str, list[np.ndarray]] = {}
    for column in self.spec.covariate_columns:
      if column not in table.columns:
        continue
      history = table[column].ffill().bfill()
      if history.isna().all():
        continue
      future = pd.Series([history.iloc[-1]] * horizon)
      values = np.concatenate(
        [
          history.to_numpy(),
          future.to_numpy(),
        ]
      )
      if pd_types.is_bool_dtype(history) or pd_types.is_object_dtype(history):
        categorical[column] = [values]
      else:
        numerical[column] = [values.astype(np.float32)]
    return {"numerical": numerical, "categorical": categorical}
