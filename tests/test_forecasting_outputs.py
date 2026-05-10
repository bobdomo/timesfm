import json

import numpy as np
import pandas as pd

from timesfm.forecasting.outputs import (
  ForecastResult,
  build_run_summary,
  forecast_results_to_dataframe,
)


def test_build_run_summary_collects_run_metadata():
  results = [
    ForecastResult(
      target="bookings_sold",
      frequency="daily",
      model_type="baseline",
      input_mode="single",
      scenario_label="baseline-plan",
      run_id="run-123",
      run_timestamp="2026-05-10T12:00:00Z",
      forecast_dates=pd.date_range("2024-02-01", periods=2, freq="D"),
      point_forecast=np.array([1.0, 2.0], dtype=np.float32),
      quantile_forecast=np.zeros((2, 10), dtype=np.float32),
    )
  ]
  quality = pd.DataFrame(
    [
      {"column": "bookings_sold", "missing_count": 1, "missing_ratio": 0.25},
      {"column": "flight_frequency", "missing_count": 0, "missing_ratio": 0.0},
    ]
  )
  dataset = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4, freq="D")})

  actual = build_run_summary(
    dataset=dataset,
    results=results,
    quality_report=quality,
    input_mode="single",
    frequencies=("daily",),
    horizons={"daily": 2},
    scenario_label="baseline-plan",
    run_id="run-123",
    run_timestamp="2026-05-10T12:00:00Z",
  )

  assert actual["input_mode"] == "single"
  assert actual["scenario_label"] == "baseline-plan"
  assert actual["run_id"] == "run-123"
  assert actual["run_timestamp"] == "2026-05-10T12:00:00Z"
  assert actual["row_count"] == 4
  assert actual["forecast_rows"] == 2
  assert actual["quality"]["bookings_sold"]["missing_ratio"] == 0.25
  assert actual["warnings"] == [
    {
      "column": "bookings_sold",
      "severity": "warning",
      "message": "Missing ratio 0.25 exceeds threshold 0.20.",
    }
  ]
  json.dumps(actual)


def test_forecast_results_to_dataframe_includes_scenario_label():
  result = ForecastResult(
    target="bookings_sold",
    frequency="daily",
    model_type="baseline",
    input_mode="single",
    scenario_label="high-demand",
    run_id="run-123",
    run_timestamp="2026-05-10T12:00:00Z",
    forecast_dates=pd.date_range("2024-02-01", periods=1, freq="D"),
    point_forecast=np.array([1.0], dtype=np.float32),
    quantile_forecast=np.zeros((1, 10), dtype=np.float32),
  )

  actual = forecast_results_to_dataframe([result])

  assert actual["scenario_label"].tolist() == ["high-demand"]
  assert actual["run_id"].tolist() == ["run-123"]
  assert actual["run_timestamp"].tolist() == ["2026-05-10T12:00:00Z"]
