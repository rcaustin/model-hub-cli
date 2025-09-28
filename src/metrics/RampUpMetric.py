"""
RampUpMetric.py
===============

Evaluates how easy it is for a new developer team to understand and use an AI model,
based on documentation like README.md and model_index.json files from Hugging Face.

Responsibilities
----------------
- Retrieve documentation files for the model.
- Compose a prompt to evaluate documentation clarity and usefulness.
- Use LLMClient to query a language model and extract a ramp-up ease score.

Dependencies
------------
- Requires a valid Hugging Face modelLink in the `ModelData` object.
- Requires GEN_AI_STUDIO_API_KEY environment variable for LLMClient.
"""

from typing import Optional

from huggingface_hub import hf_hub_download
from loguru import logger

from src.Metric import Metric
from src.ModelData import ModelData
from src.util.LLMClient import LLMClient


class RampUpMetric(Metric):
    def __init__(self) -> None:
        self.llm = LLMClient()

    def evaluate(self, model: ModelData) -> float:
        logger.debug("Evaluating Ramp Up Time Metric...")

        # Fetch README.md and model_index.json content
        readme_text = self._get_readme_text(model)
        model_index_text = self._get_model_index_text(model)
        if not readme_text and not model_index_text:
            logger.warning(
                "No README.md or model_index.json data found; returning 0.0 score."
            )
            return 0.0

        # Construct the prompt for the LLM
        prompt = (
            "You are evaluating how easy it is for a new developer team to understand "
            "and use an AI model, based only on the provided README and model index.\n"
            "Score the model's 'ramp-up ease' from 0.0 (extremely difficult to learn) "
            "to 1.0 (extremely easy to learn). Your output must contain only a single "
            "float on the first line, with no additional explanation or commentary.\n"
            "To determine the score, award up to 0.20 points each for:\n"
            "- A clear and helpful README\n"
            "- Clear installation instructions\n"
            "- Usage examples\n"
            "- A dataset description\n"
            "- A training script\n"
            "Again, respond with a single float (e.g., 0.60) on the first line. You "
            "may include justifications *after* the score if needed, but only the "
            "first line will be used as the final metric.\n"
        )
        combined_text = "\n\n".join(filter(None, [readme_text, model_index_text]))
        full_prompt = combined_text + "\n\n" + prompt

        # Query the LLM and extract the score
        response = self.llm.send_prompt(full_prompt)
        score = self.llm.extract_score(response)

        logger.debug(f"Ramp Up Time Metric score: {score}")
        return score

    def _get_readme_text(self, model: ModelData) -> Optional[str]:
        # Validate model link
        if not model.modelLink or "huggingface.co" not in model.modelLink:
            return None

        parts = model.modelLink.rstrip("/").split("/")
        if len(parts) < 5:
            return None

        repo_id = f"{parts[3]}/{parts[4]}"
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md")
            with open(readme_path, "r", encoding="utf-8") as f:
                logger.debug("Successfully fetched README.md from Hugging Face")
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to fetch README.md via huggingface_hub: {e}")
            return None

    def _get_model_index_text(self, model: ModelData) -> Optional[str]:
        # Validate model link
        if not model.modelLink or "huggingface.co" not in model.modelLink:
            return None

        parts = model.modelLink.rstrip("/").split("/")
        if len(parts) < 5:
            return None

        repo_id = f"{parts[3]}/{parts[4]}"
        try:
            index_path = hf_hub_download(repo_id=repo_id, filename="model_index.json")
            with open(index_path, "r", encoding="utf-8") as f:
                logger.debug("Successfully fetched model_index.json from Hugging Face")
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to fetch model_index.json via huggingface_hub: {e}")
            return None
