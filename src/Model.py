import time
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
        # Extract andClassify URLs
        urlset: URLSet = classify_urls(urls)
        self.modelLink: str = urlset.model
        self.codeLink: Optional[str] = urlset.code
        self.datasetLink: Optional[str] = urlset.dataset

        # Metadata Caching
        self._hf_metadata: Optional[Dict[str, Any]] = None
        self._github_metadata: Optional[Dict[str, Any]] = None

        """
        evaluations maps metric names to their scores.
        Scores can be a float or a dictionary of floats for complex metrics.
        evaluationsLatency maps metric names to the time taken to compute them.
        """
        self.evaluations: dict[str, Union[float, dict[str, float]]] = {}
        self.evaluationsLatency: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self.hf_metadata.get("id", "").split("/")[1]

    @property
    def hf_metadata(self) -> Optional[Dict[str, Any]]:
        if self._hf_metadata is None:
            fetcher = HuggingFaceFetcher()
            self._hf_metadata = fetcher.fetch_metadata(self.modelLink)
        return self._hf_metadata

    @property
    def github_metadata(self) -> Optional[Dict[str, Any]]:
        if self._github_metadata is None:
            fetcher = GitHubFetcher()
            self._github_metadata = fetcher.fetch_metadata(self.codeLink)
        return self._github_metadata

    def getScore(
        self, metric_name: str, default: float = 0.0
    ) -> float | dict[str, float]:
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
        license_score = self.evaluations.get("LicenseMetric")
        size_score = self.evaluations.get("SizeMetric")
        rampup_score = self.evaluations.get("RampUpMetric")
        bus_score = self.evaluations.get("BusFactorMetric")
        avail_score = self.evaluations.get("AvailabilityMetric")
        data_qual_score = self.evaluations.get("DatasetQualityMetric")
        code_qual_score = self.evaluations.get("CodeQualityMetric")
        perf_score = self.evaluations.get("PerformanceClaimsMetric")

        weighted_sum = (
            0.2 * sum(size_score.values()) / len(size_score) if size_score else 0.0 +
            0.3 * rampup_score +
            0.1 * bus_score +
            0.1 * avail_score +
            0.1 * data_qual_score +
            0.1 * code_qual_score +
            0.1 * perf_score
        )

        net_score = license_score * weighted_sum

        self.evaluations["NetScore"] = net_score
        self.evaluationsLatency["NetScore"] = 0.0  # Derived metric

        return round(net_score, 2)
