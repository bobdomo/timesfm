import pandas as pd

from timesfm.forecasting.ingestion import (
  SourceCSVSpec,
  load_multi_csv,
  load_single_csv,
)
from timesfm.forecasting.schema import ForecastDataSpec


def test_single_csv_mode_renames_targets_and_covariates(tmp_path):
  raw = pd.DataFrame(
    {
      "ds": ["2024-01-02", "2024-01-01"],
      "bookings": [12, 10],
      "departures": [8, 7],
      "arrivals": [7, 6],
      "flights": [4, 3],
    }
  )
  path = tmp_path / "merged.csv"
  raw.to_csv(path, index=False)

  spec = ForecastDataSpec(
    date_column="ds",
    target_columns={
      "bookings_sold": "bookings",
      "actual_departures": "departures",
      "actual_arrivals": "arrivals",
    },
    covariate_columns={"flight_frequency": "flights"},
  )

  actual = load_single_csv(path, spec)

  assert list(actual.columns) == [
    "date",
    "bookings_sold",
    "actual_departures",
    "actual_arrivals",
    "flight_frequency",
  ]
  assert actual["date"].dt.strftime("%Y-%m-%d").tolist() == [
    "2024-01-01",
    "2024-01-02",
  ]
  assert actual["bookings_sold"].tolist() == [10, 12]


def test_multi_csv_mode_merges_sources_on_date(tmp_path):
  bookings = pd.DataFrame(
    {
      "travel_date": ["2024-01-01", "2024-01-02"],
      "bookings": [10, 12],
      "departures": [7, 8],
    }
  )
  flights = pd.DataFrame(
    {
      "ds": ["2024-01-01", "2024-01-02"],
      "flight_count": [3, 4],
      "avg_price": [2500.0, 2600.0],
    }
  )

  bookings_path = tmp_path / "bookings.csv"
  flights_path = tmp_path / "flights.csv"
  bookings.to_csv(bookings_path, index=False)
  flights.to_csv(flights_path, index=False)

  actual = load_multi_csv(
    {
      "bookings": bookings_path,
      "flights": flights_path,
    },
    {
      "bookings": SourceCSVSpec(
        date_column="travel_date",
        column_map={
          "bookings_sold": "bookings",
          "actual_departures": "departures",
        },
      ),
      "flights": SourceCSVSpec(
        date_column="ds",
        column_map={
          "flight_frequency": "flight_count",
          "avg_flight_price": "avg_price",
        },
      ),
    },
  )

  assert actual["date"].dt.strftime("%Y-%m-%d").tolist() == [
    "2024-01-01",
    "2024-01-02",
  ]
  assert actual["bookings_sold"].tolist() == [10, 12]
  assert actual["actual_departures"].tolist() == [7, 8]
  assert actual["flight_frequency"].tolist() == [3, 4]
  assert actual["avg_flight_price"].tolist() == [2500.0, 2600.0]
