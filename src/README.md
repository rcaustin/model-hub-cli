# `src/` — Application Core

This directory contains the CLI entry point, core domain objects, metric interfaces/implementations, and utility functions for URL grouping and metadata retrieval.

## Purpose

- Parse an input file of 1–3 URLs per line (model/code/dataset)
- Normalize and fetch metadata for those URLs (Hugging Face, GitHub, dataset endpoints)
- Evaluate a suite of metrics and compute a composite **NetScore**
- Emit one **NDJSON** record per input line for downstream tooling

## Contents

- **`main.py`** — CLI entry point: parses args, reads input file, delegates to `ModelCatalogue`.
- **`Interfaces.py`** — Lightweight protocols/typed dicts describing model/code/dataset shapes used across the app.
- **`Metric.py`** — Metric base interface/abstract class; defines shared structure (e.g., `name`, `weight`, `evaluate()`).
- **`Model.py`** — Wraps URLs & fetched metadata; coordinates metric evaluations and computes NetScore.
- **`ModelCatalogue.py`** — Orchestrates batch evaluation across many input lines and serializes **NDJSON** output.
- **`metrics/`** — Individual metric implementations (Availability, License, DatasetQuality, CodeQuality, Size, RampUp, PerformanceClaims).
- **`util/`** — URL utilities and metadata fetchers (Hugging Face, GitHub, dataset helpers).

## High-level Execution Flow

1. **Input**: `main.py` ingests a path to a text file; each line contains 1–3 URLs (order agnostic).
2. **URL Grouping**: `util/url_utils.py` classifies URLs into `{model, code, dataset}` buckets.
3. **Metadata Fetch**: `util/metadata_fetchers.py` retrieves repo/model/dataset info (uses `GITHUB_TOKEN` if set).
4. **Model Setup**: `Model.py` stores grouped URLs + metadata and prepares metric contexts.
5. **Metrics**: Classes in `metrics/` implement `Metric` and return numeric scores and latencies.
6. **Aggregation**: `Model.py` computes the weighted **NetScore** and per-metric timing.
7. **Output**: `ModelCatalogue.py` emits one NDJSON object per input line.

```
main.py → ModelCatalogue → (url_utils, metadata_fetchers) → Model → Metric(s) → NDJSON
```

## CLI (expected behavior)

- **Positional argument**: path to input file (absolute path recommended)
- **Input format**: Each line has 1–3 URLs (model/code/dataset). Order doesn’t matter.
- **Output**: NDJSON to stdout; one object per input line with fields such as:
  - `net_score`, `net_score_latency`
  - `ramp_up`, `license`, `size_score`, `dataset_and_code_score`, `dataset_quality`, `code_quality`, `performance_claims`
  - `*_latency` fields in milliseconds

## Environment & Config

- `GITHUB_TOKEN` *(REQUIRED)* — increases GitHub API rate limit and completeness of repo metadata.
- `LOG_FILE` *(optional, file must exist)* — specifies the location of the log file to be used.
- `LOG_LEVEL` *(optional)* — one of \[0 - silent, 1 - warnings, 2 - debug\]; defaults to 0.

```bash
export GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Extension Points

- **Add a new metric**: create `src/metrics/MyNewMetric.py` implementing the `Metric` interface; register it where metrics are composed.
- **New metadata source**: extend `util/metadata_fetchers.py` with a fetch function and integrate it in `Model` construction.
- **URL types**: update `util/url_utils.py` if you introduce new URL categories or detection rules.

## Testing Notes

- Unit tests live under `tests/` with per-metric tests in `tests/metric_tests/`.
- Run with `./run test` or `pytest -q`.

