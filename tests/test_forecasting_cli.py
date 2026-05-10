import json

import numpy as np
import pandas as pd

from timesfm.forecasting.cli import main


class FakeBackend:

  def __init__(self):
    self.dynamic_numerical_covariates = []
    self.dynamic_categorical_covariates = []

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
    self.dynamic_numerical_covariates.append(dynamic_numerical_covariates or {})
    self.dynamic_categorical_covariates.append(dynamic_categorical_covariates or {})
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


def test_cli_can_enrich_with_holidays_and_economy_inputs(tmp_path):
  merged = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10, 12, 13],
      "actual_departures": [8, 9, 10],
      "actual_arrivals": [7, 8, 9],
      "flight_frequency": [2, 2, 3],
    }
  )
  holidays = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-04"]),
      "holiday_name": ["School Holiday"],
    }
  )
  economy = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
      "malaysia_cpi": [120.0, 121.0],
    }
  )
  input_path = tmp_path / "merged.csv"
  holidays_path = tmp_path / "holidays.csv"
  economy_path = tmp_path / "economy.csv"
  output_dir = tmp_path / "out"
  merged.to_csv(input_path, index=False)
  holidays.to_csv(holidays_path, index=False)
  economy.to_csv(economy_path, index=False)

  backend = FakeBackend()
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
      "--malaysia-holidays",
      str(holidays_path),
      "--economy-csv",
      str(economy_path),
    ],
    backend_factory=lambda: backend,
  )

  assert exit_code == 0
  first_categorical = backend.dynamic_categorical_covariates[0]["is_malaysia_holiday"][0]
  first_numerical = backend.dynamic_numerical_covariates[0]["malaysia_cpi"][0]
  assert first_categorical.tolist() == [False, False, False, True, False]
  assert first_numerical.tolist() == [120.0, 120.0, 121.0, 121.0, 121.0]


def test_cli_writes_quality_report_and_operational_covariates(tmp_path):
  merged = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10, 12, 13],
      "actual_departures": [8, 9, 10],
      "actual_arrivals": [7, 8, 9],
    }
  )
  flights = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
      "flight_frequency": [2, 7],
      "avg_flight_price": [2500.0, 2800.0],
    }
  )
  hotels = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
      "hotel_price_makkah": [400.0, 450.0],
      "hotel_price_madinah": [300.0, 350.0],
    }
  )
  input_path = tmp_path / "merged.csv"
  flights_path = tmp_path / "flights.csv"
  hotels_path = tmp_path / "hotels.csv"
  output_dir = tmp_path / "out"
  merged.to_csv(input_path, index=False)
  flights.to_csv(flights_path, index=False)
  hotels.to_csv(hotels_path, index=False)

  backend = FakeBackend()
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
      "daily=1",
      "--flights-csv",
      str(flights_path),
      "--hotels-csv",
      str(hotels_path),
    ],
    backend_factory=lambda: backend,
  )

  assert exit_code == 0
  quality = pd.read_csv(output_dir / "quality_report.csv")
  assert "bookings_sold" in quality["column"].tolist()
  first_numeric = backend.dynamic_numerical_covariates[0]
  assert first_numeric["flight_frequency"][0].tolist() == [2.0, 2.0, 2.0, 7.0]
  assert first_numeric["hotel_price_makkah"][0].tolist() == [400.0, 400.0, 400.0, 450.0]


def test_cli_writes_run_summary_json(tmp_path):
  merged = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10, None, 13],
      "actual_departures": [8, 9, 10],
      "actual_arrivals": [7, 8, 9],
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
      "--frequency",
      "monthly",
      "--horizon",
      "daily=2",
      "--horizon",
      "monthly=1",
    ],
    backend_factory=lambda: FakeBackend(),
  )

  assert exit_code == 0
  summary = json.loads((output_dir / "run_summary.json").read_text())
  assert summary["input_mode"] == "single"
  assert summary["date_range"] == {
    "start": "2024-01-01",
    "end": "2024-01-03",
  }
  assert summary["frequencies"] == ["daily", "monthly"]
  assert summary["horizons"] == {"daily": 2, "monthly": 1}
  assert summary["forecast_rows"] > 0
  assert summary["quality"]["bookings_sold"]["missing_count"] == 1


def test_cli_prefers_explicit_future_covariates_csv(tmp_path):
  merged = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10, 12, 13],
      "actual_departures": [8, 9, 10],
      "actual_arrivals": [7, 8, 9],
      "flight_frequency": [2, 2, 3],
    }
  )
  future_covariates = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-04", periods=2, freq="D"),
      "flight_frequency": [9, 10],
      "is_malaysia_holiday": [True, False],
    }
  )
  input_path = tmp_path / "merged.csv"
  future_path = tmp_path / "future_covariates.csv"
  output_dir = tmp_path / "out"
  merged.to_csv(input_path, index=False)
  future_covariates.to_csv(future_path, index=False)

  backend = FakeBackend()
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
      "--future-covariates-csv",
      str(future_path),
    ],
    backend_factory=lambda: backend,
  )

  assert exit_code == 0
  first_numeric = backend.dynamic_numerical_covariates[0]["flight_frequency"][0]
  first_categorical = backend.dynamic_categorical_covariates[0]["is_malaysia_holiday"][0]
  assert first_numeric.tolist() == [2.0, 2.0, 3.0, 9.0, 10.0]
  assert first_categorical.tolist() == [False, False, False, True, False]


def test_cli_rejects_short_future_covariates_csv(tmp_path):
  merged = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10, 12, 13],
      "actual_departures": [8, 9, 10],
      "actual_arrivals": [7, 8, 9],
    }
  )
  future_covariates = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-04", periods=1, freq="D"),
      "flight_frequency": [9],
    }
  )
  input_path = tmp_path / "merged.csv"
  future_path = tmp_path / "future_covariates.csv"
  output_dir = tmp_path / "out"
  merged.to_csv(input_path, index=False)
  future_covariates.to_csv(future_path, index=False)

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
      "--future-covariates-csv",
      str(future_path),
    ],
    backend_factory=lambda: FakeBackend(),
  )

  assert exit_code == 1
