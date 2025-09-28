"""
ModelCatalogue.py
=================
Batch orchestrator that evaluates one or more input lines of URLs and emits
one NDJSON record per line.

Responsibilities
----------------
- Accept an iterable of raw lines, where each line contains 1–3 URLs that may
  correspond to a model, code repository, and/or dataset.
- Use ``util/url_utils.py`` to normalize and classify URLs into
  ``{model, code, dataset}`` buckets (order-agnostic).
- Call ``util/metadata_fetchers.py`` to retrieve metadata required by metrics
  (e.g., GitHub repo info, model cards, dataset descriptors).
- Construct a ``Model`` instance for each grouped line and execute the metric
  suite defined under ``src/metrics/`` via the ``Metric`` interface.
- Aggregate per-metric results (score + latency) and compute a composite
  **NetScore**.
- Serialize the outcome as a single NDJSON object and stream to an output sink
  (typically STDOUT).

Key Concepts
------------
- **Context**: A dict prepared for each line that includes grouped URLs,
  fetched metadata, and any clients/tokens needed by metrics.
- **Metrics**: Classes implementing ``Metric`` with ``evaluate(context)``
  returning ``(score: float, latency_ms: int)``.
- **NetScore**: A weighted aggregation of metric scores computed by ``Model``.

Typical Flow
------------
1) Parse a raw line into candidate URLs.
2) Classify into {model, code, dataset}.
3) Fetch metadata (with optional ``GITHUB_TOKEN`` for GitHub rate limits).
4) Build a ``Model`` and evaluate metrics.
5) Write one NDJSON object with:
   - per-metric scores and latencies
   - ``net_score`` and ``net_score_latency``
   - any identifiers extracted from URLs/metadata

Inputs & Outputs
----------------
Input:
- Iterable[str]: lines of whitespace-delimited URLs (blank/comment lines skipped).

Output:
- Iterable[str] or streaming writes of NDJSON lines. The exact mechanism is
  implementation-defined but should be easily testable.

Error Handling
--------------
- Malformed lines: log a warning and continue to next line.
- Network/rate-limit issues: degrade gracefully by returning partial metadata;
  metrics should handle missing fields conservatively.
- Exceptions in a single line should not crash the entire batch; handle and
  continue when practical.

Testing Notes
-------------
- Avoid real network calls in unit tests; use fixtures for metadata.
- Ensure deterministic ordering of metrics in output for stable assertions.
- Verify latency fields are non-negative integers.

Environment
-----------
- ``GITHUB_TOKEN`` (optional): improves GitHub API limits and metadata completeness.
"""



import json

from loguru import logger

from src.Metric import Metric
from src.metrics.AvailabilityMetric import AvailabilityMetric
from src.metrics.BusFactorMetric import BusFactorMetric
from src.metrics.CodeQualityMetric import CodeQualityMetric
from src.metrics.DatasetQualityMetric import DatasetQualityMetric
from src.metrics.LicenseMetric import LicenseMetric
from src.metrics.PerformanceClaimsMetric import PerformanceClaimsMetric
from src.metrics.RampUpMetric import RampUpMetric
from src.metrics.SizeMetric import SizeMetric
from src.Model import Model


class ModelCatalogue:

    # models holds all Model instances in the catalogue.
    # metrics holds all Metric instances to be applied to models.

    def __init__(self) -> None:
        self.models: list[Model] = []
        self.metrics: list[Metric] = [
            LicenseMetric(),
            AvailabilityMetric(),
            PerformanceClaimsMetric(),
            BusFactorMetric(),
            SizeMetric(),
            CodeQualityMetric(),
            DatasetQualityMetric(),
            RampUpMetric()
        ]

    def addModel(self, model: Model) -> None:
        self.models.append(model)

        logger.debug(
            """Model added to the catalogue:
            Model URL = '{}',
            Dataset URL = '{}',
            Code URL = '{}'""",
            model.modelLink,
            model.datasetLink,
            model.codeLink
        )

    def evaluateModels(self) -> None:
        for model in self.models:
            model.evaluate_all(self.metrics)

    def generateReport(self) -> str:
        ndjson_report = []
        for model in self.models:
            ndjson_report.append(self.getModelNDJSON(model))

        logger.debug("Report generated for {} models.", len(self.models))
        return "\n".join(ndjson_report)

    def getModelNDJSON(self, model: Model) -> str:
        ndjson_obj = {
            "name": model.name,
            "category": model.getCategory(),
            "net_score": model.getScore("NetScore"),
            "net_score_latency": model.getLatency("NetScore"),
            "ramp_up_time": model.getScore("RampUpMetric"),
            "ramp_up_time_latency": model.getLatency("RampUpMetric"),
            "bus_factor": model.getScore("BusFactorMetric"),
            "bus_factor_latency": model.getLatency("BusFactorMetric"),
            "performance_claims": model.getScore("PerformanceClaimsMetric"),
            "performance_claims_latency": model.getLatency("PerformanceClaimsMetric"),
            "license": model.getScore("LicenseMetric"),
            "license_latency": model.getLatency("LicenseMetric"),
            "size_score": model.getScore("SizeMetric", {}),  # may be dict
            "size_score_latency": model.getLatency("SizeMetric"),
            "dataset_and_code_score": model.getScore("AvailabilityMetric"),
            "dataset_and_code_score_latency": model.getLatency("AvailabilityMetric"),
            "dataset_quality": model.getScore("DatasetQualityMetric"),
            "dataset_quality_latency": model.getLatency("DatasetQualityMetric"),
            "code_quality": model.getScore("CodeQualityMetric"),
            "code_quality_latency": model.getLatency("CodeQualityMetric"),
        }

        # Convert dictionary to NDJSON (one key-value pair per line)
        return json.dumps(ndjson_obj, separators=(",", ":"))
