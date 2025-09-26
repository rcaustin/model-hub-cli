import os

import requests
from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric


class RampUpMetric(Metric):
    PARDUE_AI_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"

    def __init__(self):
        self.API_KEY = os.getenv("GEN_AI_STUDIO_API_KEY", "")
        if not self.API_KEY:
            logger.warning(
                "Environment variable GEN_AI_STUDIO_API_KEY is not set. "
                "API calls may fail."
            )

    def evaluate(self, model: ModelData) -> float:
        logger.debug("Evaluating Ramp Up Time Metric...")

        readme_text = self._get_readme_text(model)
        model_index_text = self._get_model_index_text(model)

        if not readme_text and not model_index_text:
            logger.warning(
                "No README.md or model_index.json data found; returning 0.0 score."
            )
            return 0.0

        combined_text = "\n\n".join(filter(None, [readme_text, model_index_text]))
        score = self._query_purdue_ai(combined_text)

        logger.debug(f"Ramp Up Time Metric score: {score}")
        return score

    def _get_readme_text(self, model: ModelData) -> str | None:
        """Get README text from cached metadata or fetch from repo URL."""
        # Try GitHub metadata first
        if model.github_metadata and "readme" in model.github_metadata:
            logger.debug("Using cached README from GitHub metadata")
            return model.github_metadata["readme"]

        # Try HuggingFace metadata next
        if model.hf_metadata and "readme" in model.hf_metadata:
            logger.debug("Using cached README from HuggingFace metadata")
            return model.hf_metadata["readme"]

        # Fallback to fetching README.md from repo URL if available
        if model.modelLink:
            readme_url = self._construct_raw_url(model.modelLink, "README.md")
            if readme_url:
                try:
                    response = requests.get(readme_url, timeout=5)
                    response.raise_for_status()
                    logger.debug("Successfully fetched README.md from repo")
                    return response.text
                except Exception as e:
                    logger.warning(f"Failed to fetch README.md from {readme_url}: {e}")
        return None

    def _get_model_index_text(self, model: ModelData) -> str | None:
        """Get model_index.json text from cached metadata or fetch from repo URL."""
        # Try GitHub metadata first
        if model.github_metadata and "model_index" in model.github_metadata:
            logger.debug("Using cached model_index from GitHub metadata")
            return model.github_metadata["model_index"]

        # Try HuggingFace metadata next
        if model.hf_metadata and "model_index" in model.hf_metadata:
            logger.debug("Using cached model_index from HuggingFace metadata")
            return model.hf_metadata["model_index"]

        # Fallback to fetching model_index.json from repo URL if available
        if model.modelLink:
            model_index_url = self._construct_raw_url(
                model.modelLink, "model_index.json"
            )
            if model_index_url:
                try:
                    response = requests.get(model_index_url, timeout=5)
                    response.raise_for_status()
                    logger.debug("Successfully fetched model_index.json from repo")
                    return response.text
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch model_index.json from {model_index_url}: {e}"
                    )
        return None

    def _construct_raw_url(self, repo_url: str, filename: str) -> str:
        """
        Convert GitHub or Hugging Face repo URL to raw file
        URL for README or model_index.
        """
        if "github.com" in repo_url:
            parts = repo_url.rstrip("/").split("/")
            if len(parts) < 5:
                logger.warning(f"GitHub repo URL malformed: {repo_url}")
                return ""
            owner, repo = parts[3], parts[4]
            return f"https://raw.githubusercontent.com/{owner}/{repo}/main/{filename}"

        if "huggingface.co" in repo_url:
            parts = repo_url.rstrip("/").split("/")
            if len(parts) < 5:
                logger.warning(f"Hugging Face repo URL malformed: {repo_url}")
                return ""
            namespace, repo = parts[3], parts[4]
            return (
                f"https://huggingface.co/{namespace}/{repo}/resolve/main/{filename}"
            )

        logger.warning(f"Unknown repo host for URL: {repo_url}")
        return ""

    def _query_purdue_ai(self, text: str) -> float:
        """Send combined README and model_index text to Purdue AI API and get score."""
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "llama3.1:latest",
            "messages": [
                {
                    "role": "user",
                    "content": text
                }
            ],
            "stream": False
        }

        try:
            response = requests.post(
                self.PARDUE_AI_API_URL, json=body, headers=headers, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            score = data.get("ramp_up_score", 0.0)
            if not (0.0 <= score <= 1.0):
                logger.warning(f"Received out-of-range score: {score}, clamping.")
                score = max(0.0, min(1.0, score))
            return score
        except Exception as e:
            logger.error(f"Failed to get ramp-up score from Purdue AI API: {e}")
            return 0.0
