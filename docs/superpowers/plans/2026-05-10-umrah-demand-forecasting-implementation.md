# Umrah Demand Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-style TimesFM forecasting pipeline that accepts either a merged CSV or multiple source CSVs, creates a canonical daily table, derives weekly/monthly/yearly tables, and emits forecast outputs for bookings, departures, and arrivals.

**Architecture:** Add a focused `timesfm.forecasting` package that separates ingestion, canonical-table assembly, aggregation, forecasting orchestration, and serialization. Keep the model backend behind a thin protocol so the pipeline can be tested with fakes while still supporting real TimesFM runs in production.

**Tech Stack:** Python, pandas, numpy, pytest, TimesFM package APIs

---

### Task 1: Add Canonical Schema and Ingestion Building Blocks

**Files:**
- Create: `src/timesfm/forecasting/__init__.py`
- Create: `src/timesfm/forecasting/schema.py`
- Create: `src/timesfm/forecasting/ingestion.py`
- Test: `tests/test_forecasting_ingestion.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_single_csv_mode_renames_targets_and_covariates(tmp_path):
  ...

def test_multi_csv_mode_merges_sources_on_date(tmp_path):
  ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_forecasting_ingestion.py -v`
Expected: FAIL because `timesfm.forecasting.ingestion` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclasses.dataclass(frozen=True)
class ForecastDataSpec:
  date_column: str = "date"
  target_columns: Mapping[str, str] = ...
  covariate_columns: Mapping[str, str] = ...

def load_single_csv(path: str | Path, spec: ForecastDataSpec) -> pd.DataFrame:
  ...

def load_multi_csv(paths: Mapping[str, str | Path], spec: ForecastDataSpec) -> pd.DataFrame:
  ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_forecasting_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_forecasting_ingestion.py src/timesfm/forecasting/__init__.py src/timesfm/forecasting/schema.py src/timesfm/forecasting/ingestion.py
git commit -m "Add forecasting ingestion primitives"
```

### Task 2: Build Canonical Daily Table and Frequency Aggregation

**Files:**
- Create: `src/timesfm/forecasting/canonical.py`
- Create: `src/timesfm/forecasting/aggregation.py`
- Test: `tests/test_forecasting_canonical.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_canonical_daily_table_adds_calendar_spine_and_flags():
  ...

def test_aggregate_table_uses_metric_specific_rules():
  ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_forecasting_canonical.py -v`
Expected: FAIL because canonical and aggregation helpers do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def build_canonical_daily_table(df: pd.DataFrame, spec: ForecastDataSpec) -> pd.DataFrame:
  ...

def aggregate_by_frequency(df: pd.DataFrame, frequency: str, spec: ForecastDataSpec) -> pd.DataFrame:
  ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_forecasting_canonical.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_forecasting_canonical.py src/timesfm/forecasting/canonical.py src/timesfm/forecasting/aggregation.py
git commit -m "Add canonical table and aggregation helpers"
```

### Task 3: Add Forecast Orchestration and Output Serialization

**Files:**
- Create: `src/timesfm/forecasting/pipeline.py`
- Create: `src/timesfm/forecasting/outputs.py`
- Test: `tests/test_forecasting_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_pipeline_runs_all_targets_and_frequencies_with_fake_backend():
  ...

def test_output_rows_include_forecast_metadata():
  ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_forecasting_pipeline.py -v`
Expected: FAIL because the orchestration layer does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
class ForecastBackend(Protocol):
  def forecast(...): ...
  def forecast_with_covariates(...): ...

class TimesFMForecastingPipeline:
  def run(self, dataset: pd.DataFrame, future_covariates: Mapping[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_forecasting_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_forecasting_pipeline.py src/timesfm/forecasting/pipeline.py src/timesfm/forecasting/outputs.py
git commit -m "Add forecasting pipeline orchestration"
```

### Task 4: Add CLI Entry Point and End-to-End Smoke Coverage

**Files:**
- Create: `src/timesfm/forecasting/cli.py`
- Modify: `src/timesfm/__init__.py`
- Test: `tests/test_forecasting_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_cli_single_mode_writes_csv_outputs(tmp_path):
  ...

def test_cli_multi_mode_requires_named_sources(tmp_path):
  ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_forecasting_cli.py -v`
Expected: FAIL because the CLI module and exports do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def main(argv: Sequence[str] | None = None) -> int:
  ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_forecasting_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_forecasting_cli.py src/timesfm/forecasting/cli.py src/timesfm/__init__.py
git commit -m "Add forecasting CLI entry point"
```

### Task 5: Full Verification and Documentation Touch-Up

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-10-umrah-demand-forecasting-design.md`

- [ ] **Step 1: Add README usage section for the new pipeline**

```markdown
## Forecasting pipeline

python3 -m timesfm.forecasting.cli --mode single --input merged.csv --output-dir out
```

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest tests -v`
Expected: PASS

- [ ] **Step 3: Run a focused smoke command**

Run: `python3 -m timesfm.forecasting.cli --help`
Expected: exit 0 and show usage text

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-05-10-umrah-demand-forecasting-design.md
git commit -m "Document forecasting pipeline usage"
```
