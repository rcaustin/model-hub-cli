from typing import Optional

from loguru import logger

from src.Metric import Metric
from src.ModelData import ModelData
from src.util.LLMClient import LLMClient


class RampUpMetric(Metric):
    def __init__(self) -> None:
        self.llm: LLMClient = LLMClient()

    def evaluate(self, model: ModelData) -> float:
        logger.debug("Evaluating Ramp Up Time Metric...")

        # Extract README and model_index content
        readme_text: Optional[str] = model.hf_metadata.get("readme")
        model_index_text: Optional[str] = model.hf_metadata.get("model_index")
        if not readme_text and not model_index_text:
            logger.warning(
                "No README.md or model_index.json data found in model metadata; "
                "returning 0.0 score."
            )
            return 0.0

        # Construct the prompt for the LLM
        prompt: str = (
            "You are evaluating how easy it is for a new developer team to "
            "understand and use an AI model, based only on the provided README "
            "and model index.\n"
            "Score the model's 'ramp-up ease' from 0.0 (extremely difficult to "
            "learn) to 1.0 (extremely easy to learn). Your output must contain "
            "only a single float on the first line, with no additional "
            "explanation or commentary.\n"
            "To determine the score, award up to 0.20 points each for:\n"
            "- A clear and helpful README\n"
            "- Clear installation instructions\n"
            "- Usage examples\n"
            "- A dataset description\n"
            "- A training script\n"
            "Again, respond with a single float (e.g., 0.60) on the first line. "
            "You may include justifications *after* the score if needed, but "
            "only the first line will be used as the final metric.\n"
        )
        combined_text: str = "\n\n".join(filter(None, [readme_text, model_index_text]))
        full_prompt: str = combined_text + "\n\n" + prompt

        # Query the LLM and extract the score
        response: str = self.llm.send_prompt(full_prompt)
        score: float = self.llm.extract_score(response)

        logger.debug(f"Ramp Up Time Metric score: {score}")
        return score
