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

    def get_score(self, metric_name: str, default: float = 0.0) -> float:
        value = self.evaluations.get(metric_name, default)
        if isinstance(value, dict):
            return value.get("average", default)
        return value

    def get_latency(self, metric_name: str) -> int:
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
        categories = []
        if self.modelLink:
            categories.append("MODEL")
        if self.datasetLink:
            categories.append("DATASET")
        if self.codeLink:
            categories.append("CODE")
        return f"[{', '.join(categories)}]"

    def computeNetScore(self) -> float:
        license_score = self.get_score("LicenseMetric")
        size_score = self.get_score("SizeMetric")
        rampup_score = self.get_score("RampUpMetric")
        bus_score = self.get_score("BusFactorMetric")
        avail_score = self.get_score("AvailabilityMetric")
        data_qual_score = self.get_score("DatasetQualityMetric")
        code_qual_score = self.get_score("CodeQualityMetric")
        perf_score = self.get_score("PerformanceClaimsMetric")

        weighted_sum = (
            0.2 * size_score +
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

        return net_score
