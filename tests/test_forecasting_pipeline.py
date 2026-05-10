import numpy as np
import pandas as pd

from timesfm.forecasting.outputs import forecast_results_to_dataframe
from timesfm.forecasting.pipeline import (
  ForecastResult,
  PipelineRunConfig,
  TimesFMForecastingPipeline,
)
from timesfm.forecasting.schema import ForecastDataSpec


class FakeBackend:

  def __init__(self):
    self.calls = []
    self.dynamic_numerical_covariates = []
    self.dynamic_categorical_covariates = []

  def forecast(self, horizon, inputs):
    self.calls.append(("forecast", horizon, len(inputs)))
    point = np.full((len(inputs), horizon), 11.0, dtype=np.float32)
    quantiles = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
    quantiles[:, :, 1] = 10.0
    quantiles[:, :, 5] = 11.0
    quantiles[:, :, 9] = 12.0
    return point, quantiles

  def forecast_with_covariates(
    self,
    horizon,
    inputs,
    dynamic_numerical_covariates=None,
    dynamic_categorical_covariates=None,
    static_numerical_covariates=None,
    static_categorical_covariates=None,
  ):
    self.calls.append(("forecast_with_covariates", horizon, len(inputs)))
    self.dynamic_numerical_covariates.append(dynamic_numerical_covariates or {})
    self.dynamic_categorical_covariates.append(dynamic_categorical_covariates or {})
    point = np.full((len(inputs), horizon), 21.0, dtype=np.float32)
    quantiles = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
    quantiles[:, :, 1] = 20.0
    quantiles[:, :, 5] = 21.0
    quantiles[:, :, 9] = 22.0
    return point, quantiles


def test_pipeline_runs_all_targets_and_frequencies_with_fake_backend():
  dataset = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=5, freq="D"),
      "bookings_sold": [10, 12, 13, 15, 16],
      "actual_departures": [8, 9, 10, 11, 12],
      "actual_arrivals": [7, 8, 9, 10, 11],
      "flight_frequency": [2, 2, 3, 3, 4],
      "avg_flight_price": [100.0, 105.0, 110.0, 115.0, 120.0],
      "is_malaysia_holiday": [False, False, True, False, False],
    }
  )
  spec = ForecastDataSpec(
    target_columns={
      "bookings_sold": "bookings_sold",
      "actual_departures": "actual_departures",
      "actual_arrivals": "actual_arrivals",
    },
    covariate_columns={
      "flight_frequency": "flight_frequency",
      "avg_flight_price": "avg_flight_price",
      "is_malaysia_holiday": "is_malaysia_holiday",
    },
    aggregation_rules={
      "bookings_sold": "sum",
      "actual_departures": "sum",
      "actual_arrivals": "sum",
      "flight_frequency": "sum",
      "avg_flight_price": "mean",
      "is_malaysia_holiday": "max",
    },
  )
  backend = FakeBackend()
  pipeline = TimesFMForecastingPipeline(
    backend=backend,
    spec=spec,
    config=PipelineRunConfig(
      horizons={"daily": 2, "monthly": 1},
      frequencies=("daily", "monthly"),
    ),
  )

  results = pipeline.run(dataset, input_mode="multi_csv")

  assert len(results) == 12
  assert set(result.model_type for result in results) == {"baseline", "covariates"}
  assert set(result.frequency for result in results) == {"daily", "monthly"}
  assert set(result.target for result in results) == {
    "bookings_sold",
    "actual_departures",
    "actual_arrivals",
  }
  assert backend.calls.count(("forecast", 2, 1)) == 3
  assert backend.calls.count(("forecast_with_covariates", 2, 1)) == 3


def test_output_rows_include_forecast_metadata():
  result = ForecastResult(
    target="bookings_sold",
    frequency="daily",
    model_type="baseline",
    input_mode="single_csv",
    scenario_label="baseline",
    forecast_dates=pd.date_range("2024-02-01", periods=2, freq="D"),
    point_forecast=np.array([11.0, 12.0], dtype=np.float32),
    quantile_forecast=np.array(
      [[0.0, 10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0],
       [0.0, 11.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 13.0]],
      dtype=np.float32,
    ),
  )

  actual = forecast_results_to_dataframe([result])

  assert list(actual.columns) == [
    "forecast_target",
    "frequency",
    "forecast_date",
    "predicted_value",
    "lower_90",
    "median",
    "upper_90",
    "model_type",
    "input_mode",
    "scenario_label",
  ]
  assert actual["forecast_target"].tolist() == ["bookings_sold", "bookings_sold"]
  assert actual["predicted_value"].tolist() == [11.0, 12.0]
  assert actual["lower_90"].tolist() == [10.0, 11.0]


def test_pipeline_uses_supplied_future_covariates_when_present():
  dataset = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10, 11, 12],
      "actual_departures": [8, 9, 10],
      "actual_arrivals": [7, 8, 9],
      "flight_frequency": [2, 2, 3],
      "is_malaysia_holiday": [False, False, False],
    }
  )
  future_covariates = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-04", periods=2, freq="D"),
      "flight_frequency": [7, 8],
      "is_malaysia_holiday": [True, False],
    }
  )
  spec = ForecastDataSpec(
    target_columns={
      "bookings_sold": "bookings_sold",
      "actual_departures": "actual_departures",
      "actual_arrivals": "actual_arrivals",
    },
    covariate_columns={
      "flight_frequency": "flight_frequency",
      "is_malaysia_holiday": "is_malaysia_holiday",
    },
    aggregation_rules={
      "bookings_sold": "sum",
      "actual_departures": "sum",
      "actual_arrivals": "sum",
      "flight_frequency": "sum",
      "is_malaysia_holiday": "max",
    },
  )
  backend = FakeBackend()
  pipeline = TimesFMForecastingPipeline(
    backend=backend,
    spec=spec,
    config=PipelineRunConfig(horizons={"daily": 2}, frequencies=("daily",)),
  )

  pipeline.run(dataset, input_mode="single_csv", future_covariates=future_covariates)

  first_numeric = backend.dynamic_numerical_covariates[0]["flight_frequency"][0]
  first_categorical = backend.dynamic_categorical_covariates[0]["is_malaysia_holiday"][0]
  assert first_numeric.tolist() == [2.0, 2.0, 3.0, 7.0, 8.0]
  assert first_categorical.tolist() == [False, False, False, True, False]
