"""Frequency aggregation helpers for forecasting tables."""

import pandas as pd

from .schema import ForecastDataSpec


_FREQUENCY_TO_PERIOD = {
  "daily": "D",
  "weekly": "W",
  "monthly": "M",
  "yearly": "Y",
}


def aggregate_by_frequency(
  df: pd.DataFrame,
  frequency: str,
  spec: ForecastDataSpec,
) -> pd.DataFrame:
  """Aggregate a canonical table to a lower frequency."""
  if frequency not in _FREQUENCY_TO_PERIOD:
    raise ValueError(f"Unsupported frequency: {frequency}")

  working = df.copy()
  working["date"] = pd.to_datetime(working["date"])
  period = working["date"].dt.to_period(_FREQUENCY_TO_PERIOD[frequency])
  working = working.assign(_period=period)

  grouped = working.groupby("_period", sort=True)
  pieces = []
  for column, rule in spec.aggregation_rules.items():
    if column not in working.columns:
      continue
    if rule == "sum":
      series = grouped[column].sum(min_count=1)
    elif rule == "mean":
      series = grouped[column].mean()
    elif rule == "max":
      series = grouped[column].max()
    elif rule == "end":
      series = grouped[column].last()
    else:
      raise ValueError(f"Unsupported aggregation rule for {column}: {rule}")
    pieces.append(series.rename(column))

  if not pieces:
    return pd.DataFrame(columns=["date"])

  aggregated = pd.concat(pieces, axis=1).reset_index()
  aggregated["date"] = aggregated["_period"].dt.start_time
  return aggregated.drop(columns=["_period"])[["date"] + [c for c in aggregated.columns if c not in {"_period", "date"}]]
