import time
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import requests
from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric
from src.util.URLBundler import URLBundle


class Model(ModelData):

    def __init__(
        self,
        urls: URLBundle
    ):
        self.modelLink: str = urls.model
        self.codeLink: Optional[str] = urls.code
        self.datasetLink: Optional[str] = urls.dataset

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
            self._hf_metadata = self._fetch_hf_metadata()
        return self._hf_metadata

    @property
    def github_metadata(self) -> Optional[Dict[str, Any]]:
        if self._github_metadata is None:
            self._github_metadata = self._fetch_github_metadata()
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

    def _fetch_hf_metadata(self) -> Optional[Dict[str, Any]]:
        if not self.modelLink:
            logger.debug("No modelLink provided, skipping HuggingFace metadata fetch.")
            return None

        try:
            parts = self.modelLink.rstrip("/").split("/")
            if len(parts) < 2:
                logger.warning("Model link is malformed: {}", self.modelLink)
                return None

            org, model_id = parts[-2], parts[-1]
            url = f"https://huggingface.co/api/models/{org}/{model_id}"
            logger.debug("Fetching HuggingFace metadata from: {}", url)

            response = requests.get(url, timeout=5)

            if response.ok:
                logger.debug("HuggingFace metadata retrieved for model '{}'.", model_id)
                return response.json()

            logger.warning(
                "Failed to retrieve HuggingFace metadata (HTTP {}).",
                response.status_code
            )

        except Exception as e:
            logger.exception("Exception while fetching HuggingFace metadata: {}", e)

        return None

    def _fetch_github_metadata(self) -> Optional[Dict[str, Any]]:
        if not self.codeLink:
            logger.debug("No codeLink provided, skipping GitHub metadata fetch.")
            return None

        try:
            parsed = urlparse(self.codeLink)
            if "github.com" not in parsed.netloc:
                logger.debug(f"Code link is not a GitHub URL: {self.codeLink}")
                return None

            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) < 2:
                logger.warning(f"Invalid GitHub repository path: {parsed.path}")
                return None

            owner, repo = path_parts[0], path_parts[1]
            base_url = f"https://api.github.com/repos/{owner}/{repo}"
            headers = {
                "Accept": "application/vnd.github.v3+json",
                # "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",  # optional
            }

            metadata: Dict[str, Any] = {}

            # Fetch contributors
            contributors_url = f"{base_url}/contributors"
            logger.debug(f"Fetching GitHub contributors from: {contributors_url}")
            contributors_resp = requests.get(
                contributors_url,
                headers=headers,
                timeout=5
            )
            if contributors_resp.ok:
                metadata["contributors"] = contributors_resp.json()
                logger.debug("Contributors data retrieved.")
            else:
                logger.warning(
                    "Failed to fetch contributors (HTTP {}).",
                    contributors_resp.status_code
                )

            # Fetch license
            license_url = f"{base_url}/license"
            logger.debug(f"Fetching GitHub license from: {license_url}")
            license_resp = requests.get(license_url, headers=headers, timeout=5)
            if license_resp.ok:
                metadata["license"] = license_resp.json()
                logger.debug("License data retrieved.")
            else:
                logger.warning(
                    "Failed to fetch license (HTTP {}).",
                    license_resp.status_code
                )

            return metadata if metadata else None

        except Exception as e:
            logger.exception("Exception while fetching GitHub metadata: {}", e)
            return None
