"""CSV ingestion helpers for forecasting datasets."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .schema import ForecastDataSpec


@dataclasses.dataclass(frozen=True)
class SourceCSVSpec:
  """Column mapping for one CSV source in multi-file mode."""

  date_column: str
  column_map: Mapping[str, str]


def _read_csv(path: str | Path) -> pd.DataFrame:
  return pd.read_csv(path)


def _rename_to_canonical(
  df: pd.DataFrame,
  date_column: str,
  column_map: Mapping[str, str],
) -> pd.DataFrame:
  rename_map = {date_column: "date"} | {source: canonical for canonical, source in column_map.items()}
  missing = [source for source in rename_map if source not in df.columns]
  if missing:
    raise ValueError(f"Missing required columns: {missing}")
  output = df.rename(columns=rename_map)[list(rename_map.values())].copy()
  output["date"] = pd.to_datetime(output["date"])
  return output.sort_values("date").reset_index(drop=True)


def load_single_csv(path: str | Path, spec: ForecastDataSpec) -> pd.DataFrame:
  """Load one already-merged CSV into canonical column names."""
  raw = _read_csv(path)
  column_map = dict(spec.target_columns) | dict(spec.covariate_columns)
  return _rename_to_canonical(raw, spec.date_column, column_map)


def load_multi_csv(
  paths: Mapping[str, str | Path],
  source_specs: Mapping[str, SourceCSVSpec],
) -> pd.DataFrame:
  """Load and merge multiple source CSVs on the canonical date column."""
  missing_specs = set(paths) - set(source_specs)
  if missing_specs:
    raise ValueError(f"Missing source specs for: {sorted(missing_specs)}")

  merged: pd.DataFrame | None = None
  for source_name, path in paths.items():
    spec = source_specs[source_name]
    source_df = _rename_to_canonical(_read_csv(path), spec.date_column, spec.column_map)
    if merged is None:
      merged = source_df
    else:
      merged = merged.merge(source_df, on="date", how="outer")

  if merged is None:
    return pd.DataFrame(columns=["date"])
  return merged.sort_values("date").reset_index(drop=True)
