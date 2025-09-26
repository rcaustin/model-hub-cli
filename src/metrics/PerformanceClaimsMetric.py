"""
PerformanceClaimsMetric.py
==========================
Looks for stated performance/evaluation claims and their traceability.

Signals (examples)
------------------
- Benchmarks with dataset/task names and numbers
- Links to evaluation scripts or reproducible notebooks
- Comparison tables with baselines and metrics (accuracy, F1, WER, etc.)

Inputs (from context)
---------------------
- model_card.card_text: str | None
- code_repo README/docs text: str | None

Scoring (0–1)
-------------
- 1.0 : claims + datasets + metrics + reproduction pointers
- 0.5 : claims present but weak details (no dataset/metric names or no repro)
- 0.0 : no claims detected

Limitations
-----------
- Pure text search can miss image-only tables or PDFs.
- Be conservative when evidence is thin or unverifiable.
"""


import re
from typing import Any, Dict, List

from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric


class PerformanceClaimsMetric(Metric):
    """
    Evaluates the quality and verifiability of performance claims made by a model.

    Looks for performance metrics in:
    1. HuggingFace model cards and descriptions
    2. GitHub repository README and documentation
    3. Model metadata and tags

    Returns a score from 0.0 (no claims or misleading)
    to 1.0 (clear, quantified claims).
    """

    # Common performance metric patterns
    PERFORMANCE_PATTERNS = [
        r'accuracy[:\s]*(\d+\.?\d*)\s*%?',
        r'f1[:\s]*(\d+\.?\d*)\s*%?',
        r'precision[:\s]*(\d+\.?\d*)\s*%?',
        r'recall[:\s]*(\d+\.?\d*)\s*%?',
        r'auc[:\s]*(\d+\.?\d*)\s*%?',
        r'bleu[:\s]*(\d+\.?\d*)\s*%?',
        r'rouge[:\s]*(\d+\.?\d*)\s*%?',
        r'perplexity[:\s]*(\d+\.?\d*)',
        r'loss[:\s]*(\d+\.?\d*)',
        r'error[:\s]*(\d+\.?\d*)\s*%?',
        r'score[:\s]*(\d+\.?\d*)\s*%?',
        r'(\d+\.?\d*)\s*%?\s*(accuracy|f1|precision|recall|auc|bleu|rouge)',
        # Add more flexible patterns
        r'bleu\s+score\s+of\s+(\d+\.?\d*)',
        r'rouge-l[:\s]*(\d+\.?\d*)',
    ]

    # Benchmark dataset patterns
    BENCHMARK_PATTERNS = [
        r'imagenet',
        r'glue',
        r'squad',
        r'wmt',
        r'coco',
        r'vqa',
        r'ms\s*marco',
        r'common\s*crawl',
        r'wikipedia',
        r'bookcorpus',
    ]

    def evaluate(self, model: ModelData) -> float:
        """
        Evaluate the quality of performance claims made by the model.

        Args:
            model: ModelData object containing URLs and metadata

        Returns:
            float: Performance claims score from 0.0 to 1.0
        """
        logger.info("Evaluating PerformanceClaimsMetric...")

        claims_found = []
        total_score = 0.0

        # Check HuggingFace metadata for performance claims
        if model.modelLink and "huggingface.co" in model.modelLink:
            hf_claims = self._extract_hf_claims(model)
            claims_found.extend(hf_claims)

        # Check GitHub metadata for performance claims
        if model.codeLink and "github.com" in model.codeLink:
            gh_claims = self._extract_github_claims(model)
            claims_found.extend(gh_claims)

        # Calculate score based on claims quality
        if not claims_found:
            logger.info("PerformanceClaimsMetric: No performance claims found -> 0.0")
            return 0.0

        # Score each claim and calculate average
        for claim in claims_found:
            claim_score = self._score_claim(claim)
            total_score += claim_score

        final_score = total_score / len(claims_found)
        logger.info(
            "PerformanceClaimsMetric: {} claims found, average score {} -> {}",
            len(claims_found), total_score / len(claims_found), final_score
        )

        return min(final_score, 1.0)  # Cap at 1.0

    def _extract_hf_claims(self, model: ModelData) -> List[Dict[str, Any]]:
        """Extract performance claims from HuggingFace metadata."""
        claims = []

        try:
            hf_meta = model.hf_metadata
            if not hf_meta:
                return claims

            # Check model card data
            card_data = hf_meta.get("cardData", {})
            if isinstance(card_data, dict):
                # Check various text fields for performance claims
                text_fields = [
                    card_data.get("model_description", ""),
                    card_data.get("model_summary", ""),
                    card_data.get("limitations", ""),
                    card_data.get("training_data", ""),
                ]

                for field_text in text_fields:
                    if field_text:
                        field_claims = self._find_performance_claims(field_text)
                        claims.extend(field_claims)

            # Check tags for performance indicators
            tags = hf_meta.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str) and any(pattern in tag.lower()
                       for pattern in ['accuracy', 'f1', 'bleu', 'rouge']):
                        claims.append({
                            'text': tag,
                            'source': 'hf_tags',
                            'context': 'model tags'
                        })

        except Exception as e:
            logger.debug("Error extracting HuggingFace claims: {}", e)

        return claims

    def _extract_github_claims(self, model: ModelData) -> List[Dict[str, Any]]:
        """Extract performance claims from GitHub metadata."""
        claims = []

        try:
            gh_meta = model.github_metadata
            if not gh_meta:
                return claims

            # Note: GitHub API doesn't provide README content by default
            # In a real implementation, you might want to fetch the README separately
            # For now, we'll focus on repository metadata

            # Check repository description
            description = gh_meta.get("description", "")
            if description:
                desc_claims = self._find_performance_claims(description)
                claims.extend(desc_claims)

        except Exception as e:
            logger.debug("Error extracting GitHub claims: {}", e)

        return claims

    def _find_performance_claims(self, text: str) -> List[Dict[str, Any]]:
        """Find performance claims in text using regex patterns."""
        claims = []

        if not text or not isinstance(text, str):
            return claims

        text_lower = text.lower()

        # Look for performance metrics
        for pattern in self.PERFORMANCE_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                claims.append({
                    'text': match.group(0),
                    'metric': match.group(1) if match.groups() else None,
                    'source': 'text_analysis',
                    'context': 'performance metric'
                })

        # Look for benchmark mentions
        for pattern in self.BENCHMARK_PATTERNS:
            if re.search(pattern, text_lower):
                claims.append({
                    'text': pattern,
                    'source': 'text_analysis',
                    'context': 'benchmark dataset'
                })

        return claims

    def _score_claim(self, claim: Dict[str, Any]) -> float:
        """
        Score a performance claim based on its quality and verifiability.

        Scoring criteria:
        - 1.0: Quantified metric with benchmark context
        - 0.8: Quantified metric without clear benchmark
        - 0.6: Vague metric with benchmark context
        - 0.4: Vague metric without benchmark
        - 0.2: Benchmark mention without specific metrics
        - 0.0: Unclear or misleading claims
        """
        text = claim.get('text', '').lower()

        # Check if it's a quantified metric
        has_number = bool(re.search(r'\d+\.?\d*', text))

        # Check if it mentions a benchmark
        has_benchmark = any(pattern in text for pattern in self.BENCHMARK_PATTERNS)

        # Check if it's a specific performance metric
        is_performance_metric = any(metric in text for metric in [
            'accuracy', 'f1', 'precision', 'recall',
            'auc', 'bleu', 'rouge', 'perplexity'
        ])

        # Score based on criteria
        if has_number and is_performance_metric and has_benchmark:
            return 1.0
        elif has_number and is_performance_metric:
            return 0.8
        elif is_performance_metric and has_benchmark:
            return 0.6
        elif is_performance_metric:
            return 0.4
        elif has_benchmark:
            return 0.2
        else:
            return 0.0
