"""
metadata_fetchers.py
====================
Thin wrappers around external APIs/HTML endpoints to gather repo/model/dataset
metadata needed by metrics.

Responsibilities
----------------
- Fetch GitHub repository information (license, stars/forks, last commit,
  contributors) for classified code URLs.
- Fetch Hugging Face model card text, tags, and basic model metadata for model URLs.
- Probe dataset/other URLs for basic reachability and descriptive content.
- Apply small caching/retry logic; handle rate limits gracefully.

Typical Functions
-----------------
- fetch_code_repo(url: str, token: str | None = None) -> dict
- fetch_model_card(url: str) -> dict
- fetch_dataset_info(url: str) -> dict
- build_metadata_bundle(urls: dict[str, str | None], token: str | None = None) -> dict
    Returns a dict aggregating the pieces above for consumption by Model/metrics.

Error Handling
--------------
- Never raise uncaught exceptions for network failures; return partial results
  with flags like {"reachable": False, "error": "..."} and log a warning.
- Timeouts should be short and retried a limited number of times.

Testing
-------
- Use fixtures to simulate API responses and rate-limit conditions.
- Ensure deterministic outputs for given inputs (no live network dependency).
"""

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from loguru import logger


class MetadataFetcher:
    def fetch_metadata(self, url: Optional[str]) -> Dict[str, Any]:
        raise NotImplementedError("Must be implemented by subclasses.")


class HuggingFaceFetcher(MetadataFetcher):
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self.BASE_API_URL = "https://huggingface.co/api/models"

    def fetch_metadata(self, url: Optional[str]) -> Dict[str, Any]:
        metadata = {}

        # Verify URL Exists
        # - Should Always Exist for Model URLs
        if not url:
            logger.error("No model URL provided to HuggingFaceFetcher.")
            return metadata

        # Verify URL is a HuggingFace Model URL
        # - Should Always Be a HuggingFace Model URL
        parsed = urlparse(url)
        if "huggingface.co" not in parsed.netloc:
            logger.error(f"Unsupported model URL: {url}")
            return metadata

        # Parse URL to Extract Organization ID and Model ID
        # - Expect URL Format: huggingface.co/{organization}/{model_id}
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            logger.warning(f"Malformed HuggingFace model URL: {url}")
            return metadata

        organization, model_id = parts[0], parts[1]
        api_url = f"{self.BASE_API_URL}/{organization}/{model_id}"

        # Fetch Metadata from Hugging Face API
        try:
            logger.debug(f"Fetching HF metadata from: {api_url}")
            resp = self.session.get(api_url, timeout=5)

            if resp.ok:
                logger.debug(f"HF metadata retrieved for model: {model_id}")
                metadata = resp.json()
            else:
                logger.warning(
                    f"Failed to retrieve HF metadata (HTTP {resp.status_code}) "
                    f"for {url}"
                )
        except Exception as e:
            logger.exception(f"Exception fetching HF metadata: {e}")

        return metadata


class GitHubFetcher(MetadataFetcher):
    def __init__(
        self,
        token: Optional[str] = None,
        session: Optional[requests.Session] = None
    ) -> None:
        self.token = token
        self.session = session or requests.Session()
        self.BASE_API_URL = "https://api.github.com/repos"

    def fetch_metadata(self, url: Optional[str]) -> Dict[str, Any]:
        metadata = {}

        # Verify URL Exists
        # - May Not Exist if No Code Link Provided
        if not url:
            logger.info("No repository URL provided to GitHubFetcher.")
            return metadata

        # Verify URL is a Valid GitHub URL
        # - May Not Be a GitHub URL if Unsupported Code Link Provided
        parsed = urlparse(url)
        if "github.com" not in parsed.netloc:
            logger.info(f"URL is not a GitHub URL: {url}")
            return metadata

        # Parse URL to Extract Owner and Repository Name
        # - Expect URL Format: github.com/{owner}/{repo}
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            logger.warning(f"Malformed GitHub URL: {url}")
            return metadata

        owner, repo = parts[0], parts[1]
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            # Fetch contributors
            contributors_url = f"{self.BASE_API_URL}/{owner}/{repo}/contributors"
            logger.debug(f"Fetching GitHub contributors from: {contributors_url}")
            resp = self.session.get(contributors_url, headers=headers, timeout=5)
            if resp.ok:
                metadata["contributors"] = resp.json()
            else:
                logger.warning(
                    f"Failed to fetch contributors (HTTP {resp.status_code}) for {url}"
                )

            # Fetch license
            license_url = f"{self.BASE_API_URL}/{owner}/{repo}/license"
            logger.debug(f"Fetching GitHub license from: {license_url}")
            resp = self.session.get(license_url, headers=headers, timeout=5)
            if resp.ok:
                license = resp.json().get("license", {}).get("spdx_id")
                metadata["license"] = license
            else:
                logger.warning(
                    f"Failed to fetch license (HTTP {resp.status_code}) for {url}"
                )

            # Fetch repository info
            repo_url = f"{self.BASE_API_URL}/{owner}/{repo}"
            logger.debug(f"Fetching GitHub repository info from: {repo_url}")
            resp = self.session.get(repo_url, headers=headers, timeout=5)
            if resp.ok:
                repo_data = resp.json()
                metadata["clone_url"] = repo_data.get("clone_url")
                metadata["stargazers_count"] = repo_data.get("stargazers_count", 0)
                metadata["forks_count"] = repo_data.get("forks_count", 0)
            else:
                logger.warning(
                    f"Failed to fetch repository info (HTTP {resp.status_code}) "
                    f"for {url}"
                )

            # Fetch recent commit activity
            commits_url = f"{self.BASE_API_URL}/{owner}/{repo}/commits"
            logger.debug(f"Fetching GitHub commits from: {commits_url}")
            params = {"since": "30 days ago", "per_page": 100}
            resp = self.session.get(commits_url, params=params, headers=headers)
            if resp.ok:
                commits = resp.json()
                avg_daily = len(commits) / 30
                metadata["avg_daily_commits_30d"] = avg_daily
            else:
                logger.warning(
                    f"Failed to fetch commits (HTTP {resp.status_code}) for {url}"
                )

        except Exception as e:
            logger.exception(f"Exception fetching GitHub metadata: {e}")

        return metadata


class DatasetFetcher(MetadataFetcher):
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self.BASE_API_URL = "https://huggingface.co/api/datasets"

    def fetch_metadata(self, url: Optional[str]) -> Dict[str, Any]:
        metadata = {}

        # Verify URL Exists
        # - May Not Exist if No Dataset Link Provided
        if not url:
            logger.debug("No dataset URL provided to DatasetFetcher.")
            return metadata

        # Verify URL is a HuggingFace Dataset URL
        # - May Not Be a HuggingFace URL if Unsupported Dataset Link Provided
        parsed = urlparse(url)
        if parsed.netloc != "huggingface.co":
            logger.warning(f"Unsupported dataset URL domain: {url}")
            return metadata

        # Parse URL to Extract Organization and Dataset ID
        # - Expect URL Format: huggingface.co/datasets/{organization}/{dataset_id}
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "datasets":
            logger.warning(f"Malformed dataset URL path: {url}")
            return metadata

        organization, dataset_id = parts[1], parts[2]
        api_url = f"{self.BASE_API_URL}/{organization}/{dataset_id}"

        # Fetch Metadata from HuggingFace Datasets API
        try:
            logger.debug(f"Fetching HF dataset metadata from: {api_url}")
            resp = self.session.get(api_url, timeout=5)
            if resp.ok:
                metadata = resp.json()
            else:
                logger.warning(
                    f"Failed to retrieve HF dataset metadata (HTTP {resp.status_code}) "
                    f"for {url}"
                )
        except Exception as e:
            logger.exception(f"Exception fetching HF dataset metadata: {e}")

        return metadata
