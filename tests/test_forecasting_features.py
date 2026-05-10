import pandas as pd

from timesfm.forecasting.features import (
  align_daily_covariates,
  add_known_calendar_covariates,
  align_low_frequency_covariates,
)


def test_add_known_calendar_covariates_marks_holidays_and_jeddah_winter():
  frame = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-10", "2024-06-17", "2024-07-01"]),
    }
  )
  malaysia_holidays = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-06-17"]),
      "holiday_name": ["Hari Raya Haji"],
    }
  )

  actual = add_known_calendar_covariates(frame, malaysia_holidays=malaysia_holidays)

  assert actual["is_jeddah_winter"].tolist() == [True, False, False]
  assert actual["is_malaysia_holiday"].tolist() == [False, True, False]
  assert actual["holiday_name"].tolist() == [None, "Hari Raya Haji", None]


def test_align_low_frequency_covariates_forward_fills_monthly_values():
  daily = pd.DataFrame(
    {"date": pd.date_range("2024-01-01", periods=4, freq="D")}
  )
  monthly = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
      "malaysia_cpi": [120.5, 121.0],
    }
  )

  actual = align_low_frequency_covariates(daily, monthly, value_columns=["malaysia_cpi"])

  assert actual["malaysia_cpi"].tolist() == [120.5, 120.5, 121.0, 121.0]


def test_align_daily_covariates_aligns_history_and_future_values():
  daily = pd.DataFrame(
    {"date": pd.date_range("2024-01-01", periods=5, freq="D")}
  )
  flights = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"]),
      "flight_frequency": [2, 4, 5],
      "avg_flight_price": [2500.0, 2700.0, 2800.0],
    }
  )

  actual = align_daily_covariates(
    daily,
    flights,
    value_columns=["flight_frequency", "avg_flight_price"],
  )

  assert pd.isna(actual.loc[0, "flight_frequency"])
  assert actual["flight_frequency"].tolist()[1:] == [2, 2, 4, 5]
  assert pd.isna(actual.loc[0, "avg_flight_price"])
  assert actual["avg_flight_price"].tolist()[1:] == [2500.0, 2500.0, 2700.0, 2800.0]
