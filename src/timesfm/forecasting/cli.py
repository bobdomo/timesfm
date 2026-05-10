"""Command-line entry point for the forecasting pipeline."""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from .ingestion import SourceCSVSpec, load_multi_csv, load_single_csv
from .outputs import forecast_results_to_dataframe
from .pipeline import PipelineRunConfig, TimesFMForecastingPipeline
from .schema import ForecastDataSpec


def _default_spec() -> ForecastDataSpec:
  return ForecastDataSpec(
    target_columns={
      "bookings_sold": "bookings_sold",
      "actual_departures": "actual_departures",
      "actual_arrivals": "actual_arrivals",
    },
    covariate_columns={
      "flight_frequency": "flight_frequency",
      "avg_flight_price": "avg_flight_price",
      "is_malaysia_holiday": "is_malaysia_holiday",
      "is_jeddah_winter": "is_jeddah_winter",
      "hotel_price_makkah": "hotel_price_makkah",
      "hotel_price_madinah": "hotel_price_madinah",
      "malaysia_cpi": "malaysia_cpi",
    },
    aggregation_rules={
      "bookings_sold": "sum",
      "actual_departures": "sum",
      "actual_arrivals": "sum",
      "flight_frequency": "sum",
      "avg_flight_price": "mean",
      "is_malaysia_holiday": "max",
      "is_jeddah_winter": "max",
      "hotel_price_makkah": "mean",
      "hotel_price_madinah": "mean",
      "malaysia_cpi": "mean",
    },
  )


def _parse_horizons(values: Sequence[str]) -> dict[str, int]:
  horizons = {}
  for value in values:
    name, separator, raw_horizon = value.partition("=")
    if not separator:
      raise ValueError(f"Invalid horizon mapping: {value}")
    horizons[name] = int(raw_horizon)
  return horizons


def _parse_named_sources(values: Sequence[str]) -> dict[str, str]:
  sources = {}
  for value in values:
    name, separator, path = value.partition("=")
    if not separator:
      raise ValueError(f"Invalid source mapping: {value}")
    sources[name] = path
  return sources


def _load_source_specs(path: str | Path) -> Mapping[str, SourceCSVSpec]:
  raw = json.loads(Path(path).read_text())
  return {
    name: SourceCSVSpec(
      date_column=value["date_column"],
      column_map=value["column_map"],
    )
    for name, value in raw.items()
  }


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run the TimesFM forecasting pipeline.")
  parser.add_argument("--mode", choices=["single", "multi"], required=True)
  parser.add_argument("--input")
  parser.add_argument("--source", action="append", default=[])
  parser.add_argument("--source-spec")
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--frequency", action="append", default=[])
  parser.add_argument("--horizon", action="append", default=[])
  return parser


def main(
  argv: Sequence[str] | None = None,
  backend_factory: Callable[[], object] | None = None,
) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)

  frequencies = tuple(args.frequency or ["daily", "weekly", "monthly", "yearly"])
  horizons = _parse_horizons(args.horizon or ["daily=30", "weekly=12", "monthly=6", "yearly=2"])
  spec = _default_spec()

  try:
    if args.mode == "single":
      if not args.input:
        raise ValueError("--input is required for single mode")
      dataset = load_single_csv(args.input, spec)
    else:
      if not args.source_spec:
        raise ValueError("--source-spec is required for multi mode")
      source_paths = _parse_named_sources(args.source)
      source_specs = _load_source_specs(args.source_spec)
      dataset = load_multi_csv(source_paths, source_specs)
  except ValueError:
    return 1

  if backend_factory is None:
    raise ValueError("A backend_factory is required to run the forecasting CLI.")

  pipeline = TimesFMForecastingPipeline(
    backend=backend_factory(),
    spec=spec,
    config=PipelineRunConfig(
      horizons=horizons,
      frequencies=frequencies,
    ),
  )
  results = pipeline.run(dataset, input_mode=args.mode)
  output = forecast_results_to_dataframe(results)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  output.to_csv(output_dir / "forecasts.csv", index=False)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
