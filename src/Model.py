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
        # Extract and Classify URLs
        urls: URLSet = classify_urls(urls)
        self.modelLink: str = urls.model
        self.codeLink: Optional[str] = urls.code
        self.datasetLink: Optional[str] = urls.dataset

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

    def evaluate(self, metric: Metric) -> None:
        # Evaluate the given metric and record its score and evaluation time.
        start: float = time.time()
        score: Union[float, dict[str, float]] = metric.evaluate(self)
        end: float = time.time()
        elapsed: float = end - start

        # Record the evaluation results.
        metric_name: str = type(metric).__name__
        self.evaluations[metric_name] = score
        self.evaluationsLatency[metric_name] = elapsed

    def getEvals(self) -> dict[str, Union[float, dict[str, float]]]:
        return self.evaluations

    def getEvalsLatency(self) -> dict[str, float]:
        return self.evaluationsLatency

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
        """
        Computes the NetScore using the formula:
        NetScore = License * (
            0.2 * Size +
            0.3 * Ramp-Up +
            0.1 * Bus Factor +
            0.1 * Availability +
            0.1 * Dataset Quality +
            0.1 * Cody Quality +
            0.1 * Performance Claims
        )
        """
        def get_score(metric_name: str, default: float = 0.0) -> float:
            score = self.evaluations.get(metric_name, default)
            if isinstance(score, dict):
                # THIS LINE DEPENDS ON HOW SizeMetric IS IMPLEMENTED
                return score.get("average", default)
            return score

        license_score = get_score("LicenseMetric")
        size_score = get_score("SizeMetric")
        rampup_score = get_score("RampUpMetric")
        bus_score = get_score("BusFactorMetric")
        avail_score = get_score("AvailabilityMetric")
        data_qual_score = get_score("DatasetQualityMetric")
        code_qual_score = get_score("CodeQualityMetric")
        perf_score = get_score("PerformanceClaimsMetric")

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
        self.evaluationsLatency["NetScore"] = 0.0  # Derived metric; not timed

        return net_score
