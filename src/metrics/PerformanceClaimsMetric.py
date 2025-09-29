"""
PerformanceClaimsMetric.py
==========================

Evaluates whether the model makes clear, verifiable performance claims using LLMClient.

Overview
--------
Uses LLMClient to analyze the Hugging Face model page (and optionally GitHub README)
for claims, benchmarks, and comparisons. The LLM is prompted to find claims comparing
the model to others, benchmarks showing favorable results, and multiple supporting
claims/benchmarks to determine a score in [0,1].

Scoring (0.0 – 1.0)
-------------------
+1.0: Multiple specific claims with strong benchmarks and comparisons
+0.8: Specific claims with benchmarks or comparisons
+0.6: Vague claims with some benchmarks
+0.4: Vague claims only
+0.2: Benchmark mention only
+0.0: No relevant claims found

Responsibilities
----------------
- Use LLMClient to extract claims and benchmarks from model documentation
- Score based on richness, specificity, and credibility of claims/benchmarks
- Return a score in [0,1]

Limitations
-----------
- Relies on LLMClient's ability to parse and summarize documentation
- May miss claims if documentation is sparse or poorly formatted
"""

from loguru import logger

from src.ModelData import ModelData
from src.Metric import Metric
from src.util.LLMClient import LLMClient


class PerformanceClaimsMetric(Metric):
    """
    Evaluates the quality and verifiability of performance claims made by a model
    using LLMClient to analyze documentation and benchmarks.
    """

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def evaluate(self, model: ModelData) -> float:
        """
        Evaluate the quality of performance claims made by the model using LLMClient.

        Args:
            model: ModelData object containing URLs and metadata

        Returns:
            float: Performance claims score from 0.0 to 1.0
        """
        logger.info("Evaluating PerformanceClaimsMetric with LLMClient...")

        # Gather relevant metadata for the prompt
        metadata = model._hf_metadata if hasattr(model, "_hf_metadata") else {}

        if metadata is None:
            logger.warning("No metadata available for model.")
            return 0.0
        card_data = {}
        readme = ""
        if metadata.get("cardData"):
            card_data = metadata.get("cardData", {})
        if metadata.get("readme"):
            readme = metadata.get("readme", "")

        # Compose a prompt for the LLM using extracted metadata
        prompt = (
            "Given the following Hugging Face model metadata and documentation:\n\n"
            f"Model Card Data: {card_data}\n\n"
            f"README:\n{readme}\n\n"
            "Identify any claims comparing this model to other models, "
            "benchmarks showing favorable results, "
            "and multiple supporting claims or benchmarks."
            "Summarize the claims and benchmarks, "
            "and provide a score from 0.0 to 1.0 on the first line, "
            "where 1.0 means strong, specific, "
            "and well-supported claims and benchmarks, "
            "and 0.0 means no relevant claims or benchmarks are present.\n"
            "Respond with only your score, fully numerical, with no other text."
        )

        # Query the LLM
        response = self.llm_client.send_prompt(prompt)
        score = self.llm_client.extract_score(response)

        logger.info("PerformanceClaimsMetric: LLM-based score -> {}", score)
        return min(score, 1.0)
