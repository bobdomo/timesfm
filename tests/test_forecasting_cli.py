import json

import numpy as np
import pandas as pd

from timesfm.forecasting.cli import main


class FakeBackend:

  def forecast(self, horizon, inputs):
    point = np.full((len(inputs), horizon), 5.0, dtype=np.float32)
    quantiles = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
    quantiles[:, :, 1] = 4.0
    quantiles[:, :, 5] = 5.0
    quantiles[:, :, 9] = 6.0
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
    point = np.full((len(inputs), horizon), 7.0, dtype=np.float32)
    quantiles = np.zeros((len(inputs), horizon, 10), dtype=np.float32)
    quantiles[:, :, 1] = 6.0
    quantiles[:, :, 5] = 7.0
    quantiles[:, :, 9] = 8.0
    return point, quantiles


def test_cli_single_mode_writes_csv_outputs(tmp_path):
  merged = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=4, freq="D"),
      "bookings_sold": [10, 12, 13, 15],
      "actual_departures": [8, 9, 10, 11],
      "actual_arrivals": [7, 8, 9, 10],
      "flight_frequency": [2, 2, 3, 3],
      "avg_flight_price": [100.0, 110.0, 120.0, 130.0],
      "is_malaysia_holiday": [False, False, True, False],
    }
  )
  input_path = tmp_path / "merged.csv"
  output_dir = tmp_path / "out"
  merged.to_csv(input_path, index=False)

  exit_code = main(
    [
      "--mode",
      "single",
      "--input",
      str(input_path),
      "--output-dir",
      str(output_dir),
      "--frequency",
      "daily",
      "--horizon",
      "daily=2",
    ],
    backend_factory=lambda: FakeBackend(),
  )

  assert exit_code == 0
  forecasts = pd.read_csv(output_dir / "forecasts.csv")
  assert sorted(forecasts["model_type"].unique().tolist()) == [
    "baseline",
    "covariates",
  ]
  assert sorted(forecasts["forecast_target"].unique().tolist()) == [
    "actual_arrivals",
    "actual_departures",
    "bookings_sold",
  ]


def test_cli_multi_mode_requires_named_sources(tmp_path):
  source_spec = tmp_path / "source_spec.json"
  source_spec.write_text(json.dumps({}))

  exit_code = main(
    [
      "--mode",
      "multi",
      "--source",
      str(tmp_path / "bookings.csv"),
      "--source-spec",
      str(source_spec),
      "--output-dir",
      str(tmp_path / "out"),
      "--frequency",
      "daily",
      "--horizon",
      "daily=2",
    ],
    backend_factory=lambda: FakeBackend(),
  )

  assert exit_code == 1
