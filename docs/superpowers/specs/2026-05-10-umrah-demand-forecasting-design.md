# Umrah Demand Forecasting System Design

Date: 2026-05-10
Repo: `/Users/yencusor/Documents/experiment/timesfm`
Status: Approved design, ready for user review

## Goal

Build a production-style forecasting system that uses TimesFM 2.5 to predict Umrah demand from Malaysia using both historical target data and exogenous drivers. The system must support two business situations:

- package bookings sold
- actual Umrah travel, represented as departures and arrivals when available

The system must support both ingestion modes:

- a single merged CSV
- multiple source CSVs that are standardized and merged by the pipeline

The system must also produce forecasts at four business frequencies:

- daily
- weekly
- monthly
- yearly

## Business Scope

The forecasting system will model three targets:

- `bookings_sold`
- `actual_departures`
- `actual_arrivals`

These targets are related but distinct. Bookings measure demand intent, while departures and arrivals measure realized travel and are affected by operational and market constraints.

The system must factor in the following covariates for both historical and future periods when available:

- winter season in Jeddah
- holiday season in Malaysia
- flight frequency
- flight price
- Malaysia economic conditions
- hotel prices in Makkah
- hotel prices in Madinah

## Recommended Architecture

The recommended approach is a unified-table-first architecture built around a canonical daily modeling table. All ingestion paths normalize into the same daily table, and higher-level forecast tables are derived from it.

This approach is preferred because it:

- supports both single-CSV and multi-CSV ingestion naturally
- avoids maintaining separate preprocessing systems per forecast frequency
- aligns well with TimesFM's covariate forecasting interface
- creates a clear separation between source ingestion, feature assembly, and forecasting

## System Components

The system will contain five main layers.

### 1. Ingestion Layer

Accepts either:

- `mode=single_csv`: one already-merged CSV
- `mode=multi_csv`: multiple domain-specific CSVs

Expected source families in multi-CSV mode may include:

- bookings
- actual travel
- flights
- Malaysia holidays
- Jeddah seasonal or weather proxy data
- hotel pricing
- Malaysia macroeconomic indicators

### 2. Standardization Layer

Normalizes all raw sources into a common schema and consistent units.

Responsibilities:

- parse and normalize dates
- normalize currencies
- standardize numeric types
- standardize categorical labels
- validate unique keys
- detect duplicate dates
- preserve source lineage
- create missingness and imputation flags

### 3. Feature Assembly Layer

Builds a canonical daily table using a calendar spine keyed by date. Every business date should appear once, even when some sources are missing, so missingness is explicit rather than hidden.

This layer is responsible for:

- joining standardized sources onto a daily calendar
- aligning targets and covariates to the same daily grain
- preserving source freshness metadata
- separating known future covariates from estimated future covariates

### 4. Forecasting Layer

Runs TimesFM for each target and frequency.

Model families:

- `baseline`: TimesFM `forecast()`
- `covariate`: TimesFM `forecast_with_covariates()`

Targets:

- bookings sold
- actual departures
- actual arrivals

Frequencies:

- daily
- weekly
- monthly
- yearly

### 5. Serving and Output Layer

Publishes forecast outputs in machine-friendly and analyst-friendly formats.

Supported output types:

- CSV
- JSON

Each output must include point forecasts, prediction bands, metadata, and quality warnings.

## Canonical Daily Data Model

The heart of the system is a single canonical daily table.

Each row represents one date and includes the following fields or field groups.

### Target Fields

- `bookings_sold`
- `actual_departures`
- `actual_arrivals`

### Operational Covariates

- `flight_frequency`
- `avg_flight_price`
- `hotel_price_makkah`
- `hotel_price_madinah`

### Calendar and Seasonality Covariates

- `is_malaysia_holiday`
- `holiday_type`
- `is_jeddah_winter`
- optional `ramadan_flag`
- optional `school_holiday_flag`
- optional `long_weekend_flag`

### Economic Covariates

The final metric list may vary by available data source, but the design should support:

- `fx_rate`
- `cpi`
- `consumer_sentiment`
- `unemployment_rate`
- `policy_rate`
- `gdp_proxy`

### Metadata and Quality Fields

- `source_*` lineage fields
- coverage flags
- imputation flags
- freshness timestamps
- forecastability flags

## Frequency-Specific Tables

The system should derive weekly, monthly, and yearly modeling tables from the canonical daily table.

This is not a blind aggregation. Different variables require different aggregation rules.

### Aggregation Rules

Use `sum` for:

- bookings sold
- departures
- arrivals
- flight counts

Use `average` for:

- flight price
- hotel price Makkah
- hotel price Madinah
- exchange rate
- continuous economy indicators

Use `max` or `any` for:

- holiday flags
- winter flags
- event indicators

Use `end-of-period` when appropriate for:

- indicators published as monthly or quarterly snapshots

The aggregation rules must be explicit and tested so that business totals remain interpretable across daily, weekly, monthly, and yearly views.

## Modeling Strategy

The system will not use one universal forecast for every business question. Instead, it will run a small, structured family of forecasts.

### Per-Target Models

Forecast these separately:

- `bookings_sold`
- `actual_departures`
- `actual_arrivals`

### Per-Frequency Models

Forecast these separately:

- daily
- weekly
- monthly
- yearly

Weekly, monthly, and yearly outputs should come from separately aggregated modeling tables rather than only summing daily predictions. This is recommended because the noise structure and seasonality patterns differ by frequency.

### TimesFM Strategy

Use TimesFM 2.5 as the primary model backend.

Default covariate strategy:

- use `forecast_with_covariates()`
- default to `xreg + timesfm`
- preserve `timesfm + xreg` as an experiment switch
- keep a plain `forecast()` baseline for comparison

For covariate runs, `ForecastConfig` must enable:

- `return_backcast=True`

Recommended baseline configuration:

- `normalize_inputs=True`
- `use_continuous_quantile_head=True`
- `force_flip_invariance=True`
- `fix_quantile_crossing=True`

The `infer_is_positive` setting should be enabled for strictly nonnegative targets such as bookings and travel counts.

## Future Covariate Strategy

The system must distinguish between future covariates that are known in advance and future covariates that must be estimated or supplied externally.

### Known Future Covariates

- Malaysia holiday calendar
- Jeddah winter season flag
- fixed calendar events

### Partly Known Future Covariates

- scheduled flight frequency

### Estimated or User-Supplied Future Covariates

- future flight price
- future hotel prices
- future economic indicators

The system should support two production modes:

- `user_supplied_future_covariates`
- `system_estimated_future_covariates`

For better forecast quality, user-supplied future covariates should be preferred when available.

## Data Flow

The end-to-end production flow should be:

1. Ingest one merged CSV or multiple source CSVs.
2. Standardize schema, units, and dates.
3. Validate date coverage and uniqueness.
4. Build the canonical daily table on a daily calendar spine.
5. Derive weekly, monthly, and yearly modeling tables.
6. Run baseline and covariate TimesFM forecasts for each target and frequency.
7. Publish forecast outputs and quality reports.

## Output Schema

Each forecast record should include at least:

- `forecast_target`
- `frequency`
- `forecast_date`
- `predicted_value`
- `lower_90`
- `median`
- `upper_90`
- `model_type`
- `input_mode`
- `run_timestamp`

Optional but recommended metadata:

- source coverage summary
- covariate completeness summary
- warnings
- scenario label

## Guardrails and Validation

To avoid silent data failures, the system must validate both training inputs and future covariates.

Required validations:

- reject duplicate dates after standardization unless an aggregation rule is explicitly configured
- reject unsupported or missing target columns
- reject covariate series that do not extend through the requested forecast horizon
- warn when target history is too short for a requested frequency
- warn when large shares of a feature are imputed
- warn when important sources are stale

The system should also clearly report whether each future covariate is:

- observed
- externally supplied
- internally estimated
- missing

## MVP Boundary

The first implementation should include:

- support for both single-CSV and multi-CSV ingestion
- canonical daily table build
- aggregation into daily, weekly, monthly, and yearly modeling tables
- forecast generation for bookings, departures, and arrivals
- TimesFM baseline and TimesFM-with-covariates runs
- CSV and JSON outputs
- data-quality reporting

The first implementation should not require:

- dashboard UI
- automatic API integrations
- model ensembling
- scheduling and orchestration
- scenario-planning interface

These can be added in a later phase once the data contracts and forecast quality are stable.

## Testing Strategy

Testing should cover:

- schema validation and column mapping
- date alignment across multiple CSV inputs
- canonical table generation
- aggregation correctness by frequency
- covariate horizon validation
- successful TimesFM inference on a representative sample dataset
- regression coverage for output schema and warning behavior

Recommended test grouping:

- unit tests for schema mapping and aggregation
- integration tests for ingestion-to-canonical-table assembly
- smoke tests for end-to-end TimesFM forecasting

## Risks and Open Assumptions

This design assumes:

- historical target data exists at least at a date-aligned grain that can be normalized to daily records
- future holiday and seasonal calendars can be prepared deterministically
- future flight, hotel, and macro covariates can be either supplied by users or estimated through separate logic

Primary implementation risks:

- inconsistent source frequency across vendors
- delayed availability of macro or pricing data
- weak future covariate quality reducing model lift over baseline
- sparse yearly data limiting reliability for annual forecasts

These risks are acceptable for the MVP as long as they are made visible in validation output and model comparison reporting.

## Recommended Implementation Direction

Proceed with a plan for a production-style forecasting pipeline that:

- accepts both single and multiple CSV input modes
- builds one canonical daily modeling table
- derives four forecast frequencies from that table
- runs TimesFM baseline and covariate forecasts for three business targets
- emits structured forecast outputs and validation diagnostics

This design is intentionally scoped to create a dependable core pipeline first, with UI, automation, and advanced modeling deferred until the data contracts are proven.
