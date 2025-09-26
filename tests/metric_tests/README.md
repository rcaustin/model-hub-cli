# `tests/metric_tests/` — Per-Metric Tests

Focused tests for each metric implementation in `src/metrics/`.

## Files
- **base_metric_test.py** — shared helpers/assertions for metrics.
- **test_availability_metric.py**
- **test_bus_factor_metric.py**
- **test_code_quality_metric.py**
- **test_dataset_quality.py**
- **test_license_metric.py**
- **test_metadata_fetchers.py**  # utility-specific, but lives here for convenience
- **test_performance_claims_metric.py**
- **test_size_metric.py**

## What to test for each metric
- **Score range**: `0.0 ≤ score ≤ 1.0` (unless documented otherwise).
- **Latency**: non-negative int; reasonable upper bound for unit context.
- **Happy path**: strong positive example returns high score.
- **Partial evidence**: returns mid-range score (≈0.5).
- **Negative/missing evidence**: returns low score (≈0.0).
- **Resilience**: metric returns a score even with missing context fields.

## Fixtures
- Minimal context dict that mirrors what `Model` provides (URLs + metadata).
- Small, static text blobs to emulate README/model cards.
- Repo metadata examples with varying licenses, contributors, and commit recency.

## Adding a new metric test
1. Create `test_<metric_name>_metric.py`.
2. Import the metric class from `src/metrics/...`.
3. Build fixture contexts for positive/partial/negative cases.
4. Reuse helpers from `base_metric_test.py` when possible.
5. Keep tests deterministic and offline.

## Example structure
```python
import pytest
from src.metrics.MyNewMetric import MyNewMetric

def test_my_new_metric_positive(minimal_context_positive):
    m = MyNewMetric()
    score, latency = m.evaluate(minimal_context_positive)
    assert 0.9 <= score <= 1.0
    assert isinstance(latency, int) and latency >= 0
```
