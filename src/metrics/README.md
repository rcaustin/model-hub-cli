# `src/metrics/` — Metric Implementations

This package contains the individual metric classes that score different aspects of a model/repo/dataset. Each module implements the `Metric` interface defined in `src/Metric.py` and returns:
- a numeric **score** (normalized to 0–1 unless otherwise stated), and
- a **latency** (ms) indicating runtime spent computing the metric.

## Files

- **AvailabilityMetric.py** — checks presence/reachability of model, code, and dataset URLs.
- **BusFactorMetric.py** — approximates “bus factor” risk from repository ownership/contributor signals.
- **CodeQualityMetric.py** — uses repo hygiene/structure/static signals to infer basic code quality.
- **DatasetQualityMetric.py** — evaluates dataset availability and quality indicators.
- **LicenseMetric.py** — extracts and classifies repository license compatibility/permissiveness.
- **PerformanceClaimsMetric.py** — detects stated performance/evaluation claims and traceability.
- **RampUpMetric.py** — estimates how quickly a newcomer can become productive.
- **SizeMetric.py** — computes size-related signals (repo/model size buckets, sanity checks).

> Exact features depend on what metadata was fetched in `util/metadata_fetchers.py`. Metrics should degrade gracefully (returning a conservative score) when inputs are missing.

---

## Common Interface

All metrics follow the contract in `src/Metric.py`:

- `name: str` — human-readable metric name
- `weight: float` — contribution to NetScore (set where metrics are composed)
- `evaluate(context) -> tuple[float, int]`  
  Returns `(score, latency_ms)`

The `context` object (or dict) is populated by `Model.py` and typically includes:
- grouped URLs `{model, code, dataset}`
- fetched metadata (GitHub repo info, files, README text, HF model card, etc.)
- any cached/network clients/tokens

---

## Scoring Guidance (high level)

- **0.0** = missing, blocked, or strongly negative signal  
- **0.5** = partial signal or uncertain/insufficient evidence  
- **1.0** = strong, positive, verifiable signal

Each metric module documents its specific rubric in its top-of-file docstring.

---

## Edge Cases & Resilience

- **No network / rate limits**: Prefer cached metadata from `metadata_fetchers`; otherwise return a conservative score and record why in logs.
- **Ambiguous URLs**: `url_utils` makes a best-effort classification; metrics must handle `None` for any of `{model, code, dataset}`.
- **Private repos**: treat as unavailable unless authenticated access is explicitly provided and succeeds.
- **Non-English READMEs/model cards**: attempt minimal heuristics; do not fail hard.

---

## Adding a New Metric

1. Create a new file: `src/metrics/MyNewMetric.py`
2. Implement the `Metric` interface.
3. Document:
   - inputs required from `context`
   - scoring rubric (0–1) and thresholds
   - known limitations
4. Register the metric where metrics are composed (typically in `Model.py` or a factory).

### Skeleton

```python
"""MyNewMetric.py

Evaluates <signal> to estimate <what this measures>.

Inputs expected in context:
- context["code_repo"] (dict): GitHub repo metadata
- context["model_card"] (str|None): model card text, if available
...

Scoring rubric (0–1):
- 1.0: <strong positive condition>
- 0.5: <partial condition>
- 0.0: <negative/missing condition>

Limitations:
- <notes about assumptions, rate limits, languages, etc.>
"""
from typing import Tuple
from src.Metric import Metric

class MyNewMetric(Metric):
    name = "my_new_metric"
    weight = 0.0  # set actual weight when composing metrics

    def evaluate(self, context) -> Tuple[float, int]:
        import time
        start = time.perf_counter()
        # --- compute score safely, handle missing inputs ---
        score = 0.5
        latency_ms = int((time.perf_counter() - start) * 1000)
        return score, latency_ms
```

---

## Testing

- Unit tests for each metric live in `tests/metric_tests/`  
- Run: `./run test` or `pytest -q`

Ensure:
- deterministic behavior with fixed metadata fixtures
- safe handling of `None`/missing fields
- realistic thresholds that match repository fixtures
