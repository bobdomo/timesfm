import pandas as pd

from timesfm.forecasting.quality import build_data_quality_report


def test_build_data_quality_report_summarizes_missingness():
  frame = pd.DataFrame(
    {
      "date": pd.date_range("2024-01-01", periods=3, freq="D"),
      "bookings_sold": [10.0, None, 12.0],
      "actual_departures": [8.0, 9.0, 10.0],
      "flight_frequency": [2.0, None, 3.0],
    }
  )

  actual = build_data_quality_report(
    frame,
    tracked_columns=["bookings_sold", "actual_departures", "flight_frequency"],
  )

  assert actual["column"].tolist() == [
    "bookings_sold",
    "actual_departures",
    "flight_frequency",
  ]
  assert actual["missing_count"].tolist() == [1, 0, 1]
  assert actual["missing_ratio"].tolist() == [1 / 3, 0.0, 1 / 3]
