import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class MetadataFetcher:
    """Base class/interface for metadata fetchers."""

    def fetch_metadata(self, url: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("Must be implemented by subclasses.")


class HuggingFaceFetcher(MetadataFetcher):
    """Fetch metadata from the Hugging Face API."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.BASE_API_URL = "https://huggingface.co/api/models"

    def fetch_metadata(self, model_url: str) -> Optional[Dict[str, Any]]:
        if not model_url:
            logger.debug("No model URL provided to HuggingFaceFetcher.")
            return None

        try:
            parsed = urlparse(model_url)
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) < 2:
                logger.warning(f"Malformed Hugging Face model URL: {model_url}")
                return None

            org, model_id = path_parts[0], path_parts[1]

            api_url = f"{self.BASE_API_URL}/{org}/{model_id}"
            logger.debug(f"Fetching Hugging Face metadata from: {api_url}")

            response = self.session.get(api_url, timeout=5)
            if response.ok:
                logger.debug(f"Hugging Face metadata retrieved for model '{model_id}'.")
                return response.json()
            else:
                logger.warning(
                    "Failed to retrieve Hugging Face metadata (HTTP {}) for {}.",
                    response.status_code,
                    model_url
                )
        except Exception as e:
            logger.exception(f"Exception while fetching Hugging Face metadata: {e}")

        return None


class GitHubFetcher(MetadataFetcher):
    """Fetch metadata from the GitHub API."""

    def __init__(
        self, token: Optional[str] = None,
        session: Optional[requests.Session] = None
    ):
        self.token = token
        self.session = session or requests.Session()
        self.BASE_API_URL = "https://api.github.com/repos"

    def fetch_metadata(self, repo_url: str) -> Optional[Dict[str, Any]]:
        if not repo_url:
            logger.debug("No repo URL provided to GitHubFetcher.")
            return None

        try:
            parsed = urlparse(repo_url)
            if "github.com" not in parsed.netloc:
                logger.debug(f"GitHubFetcher received a non-GitHub URL: {repo_url}")
                return None

            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) < 2:
                logger.warning(f"Invalid GitHub repository path: {parsed.path}")
                return None

            owner, repo = path_parts[0], path_parts[1]
            headers = {"Accept": "application/vnd.github.v3+json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            metadata = {}

            # Fetch contributors
            contributors_url = f"{self.BASE_API_URL}/{owner}/{repo}/contributors"
            logger.debug(f"Fetching GitHub contributors from: {contributors_url}")
            contributors_resp = self.session.get(
                contributors_url,
                headers=headers,
                timeout=5
            )
            if contributors_resp.ok:
                metadata["contributors"] = contributors_resp.json()
                logger.debug("GitHub contributors data retrieved.")
            else:
                logger.warning(
                    "Failed to fetch contributors (HTTP %s) for %s.",
                    contributors_resp.status_code,
                    repo_url
                )

            # Fetch license
            license_url = f"{self.BASE_API_URL}/{owner}/{repo}/license"
            logger.debug(f"Fetching GitHub license from: {license_url}")
            license_resp = self.session.get(license_url, headers=headers, timeout=5)
            if license_resp.ok:
                metadata["license"] = license_resp.json().get("license").get("spdx_id")
                logger.debug("GitHub license data retrieved.")
            else:
                logger.warning(
                    "Failed to fetch license (HTTP %s) for %s.",
                    license_resp.status_code,
                    repo_url
                )

            # Fetch main repository info
            repo_url = f"{self.BASE_API_URL}/{owner}/{repo}"
            logger.debug(f"Fetching GitHub repo info from: {repo_url}")
            repo_resp = self.session.get(repo_url, headers=headers, timeout=5)
            if repo_resp.ok:
                repo_data = repo_resp.json()
                metadata["clone_url"] = repo_data.get("clone_url")
                metadata["stargazers_count"] = repo_data.get("stargazers_count", 0)
                metadata["forks_count"] = repo_data.get("forks_count", 0)
                logger.debug("GitHub repository data retrieved.")

            # Fetch commit activity
            commits_url = f"{self.BASE_API_URL}/{owner}/{repo}/commits"
            params: dict[str, Any] = {"since": "30 days ago", "per_page": 100}  # Last 30 days
            commits_resp = self.session.get(commits_url, params=params, headers=headers)
            if commits_resp.ok:
                commits = commits_resp.json()
                avg_daily_commits = len(commits) / 30
                metadata["avg_daily_commits_30d"] = avg_daily_commits
                logger.debug("GitHub commit activity data retrieved.")

            return metadata if metadata else None

        except Exception as e:
            logger.exception(f"Exception while fetching GitHub metadata: {e}")
            return None
