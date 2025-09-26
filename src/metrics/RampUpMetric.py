import os
import requests

from huggingface_hub import hf_hub_download
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

        combined_text = """Evaluate the following AI model's README and index text
            to determine a ramp-up time metric representing how difficult it would
            be for a new team to learn and apply the model to develop new products
            or services. The metric should consist of a single floating-point value
            between 0.0 and 1.0, where 0.0 is extremely difficult to learn and 1.0
            is extremely easy for a new team to learn.

            Specifically you should add +.20 for each of the following:
              - a clear and helpful README
              - clear installation instructions
              - usage examples
              - a dataset description
              - a training script\n\n"""
        combined_text += "\n\n".join(filter(None, [readme_text, model_index_text]))
        score = self._query_purdue_ai(combined_text)

        logger.debug(f"Ramp Up Time Metric score: {score}")
        return score

    def _get_readme_text(self, model: ModelData) -> str | None:
        if not model.modelLink or "huggingface.co" not in model.modelLink:
            return None

        parts = model.modelLink.rstrip("/").split("/")
        if len(parts) < 5:
            return None

        repo_id = f"{parts[3]}/{parts[4]}"
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md"
            )
            with open(readme_path, "r", encoding="utf-8") as f:
                logger.debug("Successfully fetched README.md from Hugging Face repo")
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to fetch README.md via huggingface_hub: {e}")

        return None

    def _get_model_index_text(self, model: ModelData) -> str | None:
        """Fetch model_index.json text only from Hugging Face repo using
            huggingface_hub."""
        if not model.modelLink or "huggingface.co" not in model.modelLink:
            return None

        parts = model.modelLink.rstrip("/").split("/")
        if len(parts) < 5:
            return None

        repo_id = f"{parts[3]}/{parts[4]}"
        try:
            model_index_path = hf_hub_download(
                repo_id=repo_id,
                filename="model_index.json"
            )
            with open(model_index_path, "r", encoding="utf-8") as f:
                logger.debug(
                    "Successfully fetched model_index.json from Hugging Face"
                )
                return f.read()
        except Exception as e:
            logger.warning(
                f"Failed to fetch model_index.json via huggingface_hub: {e}"
            )

        return None

    def _query_purdue_ai(self, text: str) -> float:
        """Send combined README and model_index text to Purdue AI API and get score."""
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "llama3.1:latest",
            "messages": [{
                "role": "user",
                "content": text
            }],
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
