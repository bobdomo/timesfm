"""Shared schema definitions for forecasting pipelines."""

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True)
class ForecastDataSpec:
  """Canonical column mapping for a forecasting dataset."""

  date_column: str = "date"
  target_columns: Mapping[str, str] = dataclasses.field(default_factory=dict)
  covariate_columns: Mapping[str, str] = dataclasses.field(default_factory=dict)
  aggregation_rules: Mapping[str, str] = dataclasses.field(default_factory=dict)
