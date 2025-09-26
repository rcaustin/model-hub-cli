import time
import os
from typing import Any, Dict, List, Optional, Union

from src.Interfaces import ModelData
from src.Metric import Metric
from src.util.metadata_fetchers import GitHubFetcher, HuggingFaceFetcher
from src.util.url_utils import URLSet, classify_urls


class Model(ModelData):
    def __init__(
        self,
        urls: List[str]
    ):
        # Extract and Classify URLs
        urlset: URLSet = classify_urls(urls)
        self.modelLink: str = urlset.model
        self.codeLink: Optional[str] = urlset.code
        self.datasetLink: Optional[str] = urlset.dataset

        # Metadata Caching
        self._hf_metadata: Optional[Dict[str, Any]] = None
        self._github_metadata: Optional[Dict[str, Any]] = None

        # Get GitHub token from environment (validated at startup)
        self._github_token: Optional[str] = os.getenv("GITHUB_TOKEN")

        """
        evaluations maps metric names to their scores.
        Scores can be a float or a dictionary of floats for complex metrics.
        evaluationsLatency maps metric names to the time taken to compute them.
        """
        self.evaluations: dict[str, Union[float, dict[str, float]]] = {}
        self.evaluationsLatency: dict[str, float] = {}

    @property
    def name(self) -> str:
        try:
            return self.hf_metadata.get("id", "").split("/")[1]
        except (AttributeError, IndexError):
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
            # Pass the validated GitHub token to the fetcher
            fetcher = GitHubFetcher(token=self._github_token)
            self._github_metadata = fetcher.fetch_metadata(self.codeLink)
        return self._github_metadata

    def getScore(
        self, metric_name: str, default: float = 0.0
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

    def computeNetScore(self) -> float:
        def safe_score(key: str) -> float:
            val = self.evaluations.get(key)
            if key == "SizeMetric":
                # Only accept dict for SizeMetric, else 0.0
                if isinstance(val, dict) and val:
                    return sum(val.values()) / len(val)
                else:
                    return 0.0
            else:
                if isinstance(val, dict):
                    return sum(val.values()) / len(val) if val else 0.0
                return val if val is not None else 0.0

        license_score = safe_score("LicenseMetric")

        weighted_sum = (
            0.2 * safe_score("SizeMetric") +
            0.3 * safe_score("RampUpMetric") +
            0.1 * safe_score("BusFactorMetric") +
            0.1 * safe_score("AvailabilityMetric") +
            0.1 * safe_score("DatasetQualityMetric") +
            0.1 * safe_score("CodeQualityMetric") +
            0.1 * safe_score("PerformanceClaimsMetric")
        )

        net_score = license_score * weighted_sum

        self.evaluations["NetScore"] = net_score
        self.evaluationsLatency["NetScore"] = 0.0  # Derived metric, no latency
        self.evaluationsLatency["NetScore"] = sum(
            latency for key, latency in self.evaluationsLatency.items()
            if key != "NetScore"
        )
        
        return net_score
