import pandas as pd

from timesfm.forecasting.aggregation import aggregate_by_frequency
from timesfm.forecasting.canonical import build_canonical_daily_table
from timesfm.forecasting.schema import ForecastDataSpec


def test_build_canonical_daily_table_adds_calendar_spine_and_flags():
  raw = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
      "bookings_sold": [10, 12],
      "flight_frequency": [3, 4],
    }
  )
  spec = ForecastDataSpec(
    target_columns={"bookings_sold": "bookings_sold"},
    covariate_columns={"flight_frequency": "flight_frequency"},
  )

  actual = build_canonical_daily_table(raw, spec)

  assert actual["date"].dt.strftime("%Y-%m-%d").tolist() == [
    "2024-01-01",
    "2024-01-02",
    "2024-01-03",
  ]
  assert actual["bookings_sold_is_missing"].tolist() == [False, True, False]
  assert actual["flight_frequency_is_missing"].tolist() == [False, True, False]


def test_aggregate_table_uses_metric_specific_rules():
  raw = pd.DataFrame(
    {
      "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
      "bookings_sold": [10, 20, 30],
      "flight_frequency": [1, 2, 3],
      "avg_flight_price": [100.0, 200.0, 300.0],
      "is_malaysia_holiday": [False, True, False],
    }
  )
  spec = ForecastDataSpec(
    target_columns={"bookings_sold": "bookings_sold"},
    covariate_columns={
      "flight_frequency": "flight_frequency",
      "avg_flight_price": "avg_flight_price",
      "is_malaysia_holiday": "is_malaysia_holiday",
    },
    aggregation_rules={
      "bookings_sold": "sum",
      "flight_frequency": "sum",
      "avg_flight_price": "mean",
      "is_malaysia_holiday": "max",
    },
  )

  actual = aggregate_by_frequency(raw, "monthly", spec)

  assert actual["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01"]
  assert actual.loc[0, "bookings_sold"] == 60
  assert actual.loc[0, "flight_frequency"] == 6
  assert actual.loc[0, "avg_flight_price"] == 200.0
  assert bool(actual.loc[0, "is_malaysia_holiday"]) is True
