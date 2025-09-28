"""
AvailabilityMetric.py
=====================
Checks the presence and basic reachability of the three URL categories:
model, code repository, and dataset.

Signal
------
- Strong positive when expected URLs exist and are reachable.
- Neutral/partial when some are present or cannot be verified.
- Negative when all expected artifacts are missing/unreachable.

Inputs (from context)
---------------------
- urls: {"model": str|None, "code": str|None, "dataset": str|None}
- metadata: may include booleans/flags like {"code_repo": {...}}, {"dataset": {..., "reachable": bool}}

Scoring (0–1)
-------------
- 1.0  : ≥2 categories present and reachable (including "code" highly weighted)
- 0.5  : exactly 1 category present/reachable
- 0.0  : none present/reachable

Limitations
-----------
- "Reachability" is a coarse probe; private repos or transient network issues
  may reduce score conservatively.
"""


import requests
from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric


class AvailabilityMetric(Metric):
    """
    Evaluates the availability of model resources by checking:
    1. HuggingFace model accessibility
    2. GitHub repository accessibility
    3. Dataset accessibility (if provided)

    Returns a score from 0.0 (unavailable) to 1.0 (fully available).
    """

    def evaluate(self, model: ModelData) -> float:
        """
        Evaluate the availability of all model resources.

        Args:
            model: ModelData object containing URLs and metadata

        Returns:
            float: Availability score from 0.0 to 1.0
        """
        logger.info("Evaluating AvailabilityMetric...")

        total_checks = 0
        successful_checks = 0

        # Check HuggingFace model availability
        if model.modelLink and "huggingface.co" in model.modelLink:
            total_checks += 1
            if self._check_huggingface_availability(model.modelLink):
                successful_checks += 1
                logger.debug("HuggingFace model is accessible")
            else:
                logger.warning("HuggingFace model is not accessible")

        # Check GitHub repository availability
        if model.codeLink and "github.com" in model.codeLink:
            total_checks += 1
            if self._check_github_availability(model.codeLink):
                successful_checks += 1
                logger.debug("GitHub repository is accessible")
            else:
                logger.warning("GitHub repository is not accessible")

        # Check dataset availability
        if model.datasetLink and "huggingface.co" in model.datasetLink:
            total_checks += 1
            if self._check_huggingface_availability(model.datasetLink):
                successful_checks += 1
                logger.debug("HuggingFace dataset is accessible")
            else:
                logger.warning("HuggingFace dataset is not accessible")

        # Calculate availability score
        if total_checks == 0:
            logger.warning("No URLs provided for availability checking")
            return 0.0

        availability_score = successful_checks / total_checks
        logger.info(
            "AvailabilityMetric: {}/{} resources available -> {}",
            successful_checks, total_checks, availability_score
        )

        return availability_score

    def _check_huggingface_availability(self, url: str) -> bool:
        """
        Check if a HuggingFace model or dataset is accessible.

        Args:
            url: HuggingFace URL to check

        Returns:
            bool: True if accessible, False otherwise
        """
        try:
            # Extract model/dataset ID from URL
            parts = url.rstrip("/").split("/")
            if len(parts) < 2:
                return False

            org, model_id = parts[-2], parts[-1]

            # Check if it's a dataset or model
            if "/datasets/" in url:
                api_url = f"https://huggingface.co/api/datasets/{org}/{model_id}"
            else:
                api_url = f"https://huggingface.co/api/models/{org}/{model_id}"

            response = requests.get(api_url, timeout=10)
            return response.status_code == 200

        except Exception as e:
            logger.debug("Error checking HuggingFace availability: {}", e)
            return False

    def _check_github_availability(self, url: str) -> bool:
        """
        Check if a GitHub repository is accessible.

        Args:
            url: GitHub URL to check

        Returns:
            bool: True if accessible, False otherwise
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                return False

            owner, repo = path_parts[0], path_parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"

            response = requests.get(api_url, timeout=10)
            return response.status_code == 200

        except Exception as e:
            logger.debug("Error checking GitHub availability: {}", e)
            return False
