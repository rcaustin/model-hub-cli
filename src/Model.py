"""
Model.py
========
Core domain object representing a single evaluated "model bundle" composed of up
to three URLs: model, code repository, and dataset. The Model is responsible for
holding grouped URLs plus fetched metadata, executing the metric suite, and
producing a composite **NetScore** alongside per-metric results.

Responsibilities
----------------
- Store normalized inputs:
  - urls: dict with keys in { "model", "code", "dataset" } (values may be None)
  - metadata: dict populated by util/metadata_fetchers.py
- Prepare a reusable **context** dict passed to all metrics.
- Execute each ``Metric`` (see src/Metric.py, src/metrics/*) and collect:
  ``(score: float, latency_ms: int)``
- Aggregate scores using configured weights to compute ``net_score`` and record
  total latency for the aggregation path.

Attributes (typical)
--------------------
- self.urls: dict[str, str|None]
- self.metadata: dict[str, Any]
- self.metrics: list[Metric]      # concrete metric objects
- self.results: dict[str, float]  # per-metric scores by metric.name
- self.latencies: dict[str, int]  # per-metric latencies (ms)
- self.net_score: float
- self.net_score_latency: int

Workflow
--------
1) Initialize with grouped URLs and fetched metadata.
2) Build ``context`` with everything metrics may need (URLs, metadata, tokens).
3) For each metric in ``self.metrics``:
     score, latency = metric.evaluate(context)
     record in ``self.results`` and ``self.latencies``
4) Compute weighted NetScore and ``net_score_latency``.
5) Provide a serializable summary (dict) for NDJSON output.

Scoring
-------
- Each metric exposes ``name`` and ``weight``.
- NetScore = sum(weight_i * score_i) / sum(weights_present)
- Missing metrics (or missing inputs) must not crash evaluation; gracefully use
  defaults and/or skip with clear logging.

Error Handling & Resilience
---------------------------
- Metrics should never raise uncaught exceptions; catch and return conservative
  scores (e.g., 0.0) with warnings where appropriate.
- Handle absent fields in ``metadata`` and URLs = None.
- Keep evaluation deterministic for a given context.

Testing Notes
-------------
- Prefer fixture-based metadata (offline) for unit tests.
- Assert 0 ≤ scores ≤ 1, int latencies ≥ 0, and stable metric ordering.
- Verify that NetScore respects metric weights and missing metrics are handled.

Environment
-----------
- ``GITHUB_TOKEN`` (REQUIRED) must be present so underlying fetchers can improve 
  rate limits and completeness.

Thread-Safety
-------------
- Instances are not inherently thread-safe; if parallelizing evaluation, ensure
  each Model is used by a single worker or guard shared state appropriately.
"""

import time
import os
import time
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric
from src.util.metadata_fetchers import (DatasetFetcher, GitHubFetcher,
                                        HuggingFaceFetcher)


class Model(ModelData):
    def __init__(
        self,
        urls: List[str]
    ) -> None:
        # Extract URLs
        self.codeLink: Optional[str] = urls[0] if urls[0] else None
        self.datasetLink: Optional[str] = urls[1] if urls[1] else None
        self.modelLink: str = urls[2]

        # Validate Model URL Exists
        if not self.modelLink:
            raise ValueError("Model URL is required")

        # Metadata Caching
        self._hf_metadata: Optional[Dict[str, Any]] = None
        self._github_metadata: Optional[Dict[str, Any]] = None
        self._dataset_metadata: Optional[Dict[str, Any]] = None

        # Get GitHub token from environment (validated at startup)
        self._github_token: Optional[str] = os.getenv("GITHUB_TOKEN")

        # evaluations: maps metric names to their scores
        #   scores: a float or a dictionary of floats for complex metrics (SizeMetric)
        # evaluationsLatency: maps metric names to the time taken to compute them
        self.evaluations: dict[str, Union[float, dict[str, float]]] = {}
        self.evaluationsLatency: dict[str, float] = {}

    @property
    def name(self) -> str:
        try:
            if self.hf_metadata:
                return self.hf_metadata.get("id", "").split("/")[1]
        except (AttributeError, IndexError):
            pass
        return "UNKNOWN_MODEL"

    @property
    def hf_metadata(self) -> Optional[Dict[str, Any]]:
        if self._hf_metadata is None:
            fetcher = HuggingFaceFetcher()
            self._hf_metadata = fetcher.fetch_metadata(self.modelLink)
        return self._hf_metadata

    @property
    def github_metadata(self) -> Optional[Dict[str, Any]]:
        if self._github_metadata is None:
            fetcher = GitHubFetcher(token=self._github_token)
            self._github_metadata = fetcher.fetch_metadata(self.codeLink)
        return self._github_metadata

    @property
    def dataset_metadata(self) -> Optional[Dict[str, Any]]:
        if self._dataset_metadata is None:
            fetcher = DatasetFetcher()
            self._dataset_metadata = fetcher.fetch_metadata(self.datasetLink)
        return self._dataset_metadata

    def getScore(
        self, metric_name: str, default: Union[float, dict[str, float]] = 0.0
    ) -> Union[float, dict[str, float]]:
        value = self.evaluations.get(metric_name, default)
        if isinstance(value, dict):
            return {k: round(v, 2) for k, v in value.items()}
        return round(value, 2)

    def getLatency(self, metric_name: str) -> int:
        latency = self.evaluationsLatency.get(metric_name, 0.0)
        return int(latency * 1000)

    def evaluate_all(self, metrics: List[Metric]) -> None:
        for metric in metrics:
            self.evaluate(metric)
        self.computeNetScore()

    def evaluate(self, metric: Metric) -> None:
        start: float = time.time()
        score: Union[float, dict[str, float]] = metric.evaluate(self)
        end: float = time.time()

        metric_name: str = type(metric).__name__
        self.evaluations[metric_name] = score
        self.evaluationsLatency[metric_name] = end - start

    def getCategory(self) -> str:
        return "MODEL"

    def computeNetScore(self) -> None:
        def safe_score(key: str) -> float:
            val = self.evaluations.get(key)
            if key == "SizeMetric":
                # Only accept dict for SizeMetric, else 0.0
                if isinstance(val, dict) and val:
                    return sum(val.values()) / len(val)
                else:
                    logger.warning(f"SizeMetric score is not a valid dict: {val}")
                    return 0.0
            else:
                if isinstance(val, dict):
                    return sum(val.values()) / len(val) if val else 0.0
                return val if val is not None else 0.0

        # Compute Net Score and Net Latency
        license_score = safe_score("LicenseMetric")
        weighted_sum = (
            0.2 * safe_score("SizeMetric")
            + 0.3 * safe_score("RampUpMetric")
            + 0.1 * safe_score("BusFactorMetric")
            + 0.1 * safe_score("AvailabilityMetric")
            + 0.1 * safe_score("DatasetQualityMetric")
            + 0.1 * safe_score("CodeQualityMetric")
            + 0.1 * safe_score("PerformanceClaimsMetric")
        )
        self.evaluations["NetScore"] = license_score * weighted_sum
        self.evaluationsLatency["NetScore"] = sum(
            latency for key, latency in self.evaluationsLatency.items()
            if key != "NetScore"
        )
