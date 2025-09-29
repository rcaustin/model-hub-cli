"""
PerformanceClaimsMetric.py
==========================

Evaluates whether the model makes clear, verifiable performance claims.

Overview
--------
Searches Hugging Face and GitHub metadata for references to accuracy, F1, BLEU, etc.
It also detects benchmark mentions like SQuAD or ImageNet. Uses regex to find
quantified metrics in text.

Scoring (0.0 – 1.0)
-------------------
+1.0: Specific metric with benchmark (e.g. "95% accuracy on ImageNet")
+0.8: Specific metric only (e.g. "accuracy: 95%")
+0.6: Vague metric + benchmark (e.g. "strong accuracy on SQuAD")
+0.4: Vague metric only (e.g. "high F1")
+0.2: Benchmark mention only (e.g. "evaluated on GLUE")
+0.0: No relevant claims found

Responsibilities
----------------
- Extract model descriptions from Hugging Face and GitHub metadata
- Detect common metrics and benchmarks using regex
- Score each claim based on specificity and context
- Return average score across claims (top-k average to avoid dilution)

Limitations
-----------
- Does not fetch full README from GitHub (only uses metadata)
- May miss unconventional phrasing or claims outside standard fields
"""

import re
from typing import Any, Dict, List

from loguru import logger

from src.ModelData import ModelData
from src.Metric import Metric


class PerformanceClaimsMetric(Metric):
    """
    Evaluates the quality and verifiability of performance claims made by a model.

    Looks for performance metrics in:
    1. HuggingFace model cards and descriptions (including structured model-index)
    2. GitHub repository description (metadata only)
    3. Model metadata and tags

    Returns a score from 0.0 (no claims or misleading)
    to 1.0 (clear, quantified claims).
    """

    # Common performance metric patterns (tightened to avoid generic/ambiguous matches)
    PERFORMANCE_PATTERNS = [
        # Generic: "<metric> [score] (of|is|=|:) <number>%?"
        r"\b(?:accuracy|f1(?:-score)?|precision|recall|auc|bleu|rouge(?:-l)?|perplexity|loss)(?:\s*score)?(?:\s*(?:of|is|=|:))?\s*(\d+\.?\d*)\s*%?\b",
        r"\baccuracy[:\s]*(\d+\.?\d*)\s*%?\b",
        r"\bf1(?:-score)?[:\s]*(\d+\.?\d*)\s*%?\b",
        r"\bprecision[:\s]*(\d+\.?\d*)\s*%?\b",
        r"\brecall[:\s]*(\d+\.?\d*)\s*%?\b",
        r"\bauc[:\s]*(\d+\.?\d*)\s*%?\b",
        # Allow "score of|is|=|:" for BLEU/ROUGE variations
        r"\bbleu(?:\s*score)?(?:\s*(?:of|is|=|:))?\s*(\d+\.?\d*)\b",
        r"\brouge(?:-l)?(?:\s*score)?(?:\s*(?:of|is|=|:))?\s*(\d+\.?\d*)\b",
        r"\bperplexity[:\s]*(\d+\.?\d*)\b",
        r"\bloss[:\s]*(\d+\.?\d*)\b",
        # Number then metric
        r"(\d+\.?\d*)\s*%?\s*(accuracy|f1(?:-score)?|precision|recall|auc|bleu|rouge)\b",
    ]

    # Benchmark dataset patterns
    BENCHMARK_PATTERNS = [
        r"imagenet",
        r"glue",
        r"squad",
        r"wmt",
        r"coco",
        r"vqa",
        r"ms\s*marco",
        r"common\s*crawl",
        r"wikipedia",
        r"bookcorpus",
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

        claims_found: List[Dict[str, Any]] = []

        # Check HuggingFace metadata for performance claims
        if model.modelLink and "huggingface.co" in model.modelLink:
            hf_claims = self._extract_hf_claims(model)
            claims_found.extend(hf_claims)

        # Check GitHub metadata for performance claims
        if model.codeLink and "github.com" in model.codeLink:
            gh_claims = self._extract_github_claims(model)
            claims_found.extend(gh_claims)

        if not claims_found:
            logger.info("PerformanceClaimsMetric: No performance claims found -> 0.0")
            return self._normalize_score(0.0)

        # Score by top-k average to avoid dilution from many benchmark-only mentions
        try:
            scores = [self._normalize_score(self._score_claim(c)) for c in claims_found]
            scores.sort(reverse=True)
            topk = scores[:3]  # consider the three strongest signals
            avg = sum(topk) / len(topk) if topk else 0.0
        except Exception as e:
            logger.debug("PerformanceClaimsMetric: error while averaging scores: {}", e)
            avg = 0.0

        avg = self._normalize_score(avg)
        logger.info(
            "PerformanceClaimsMetric: {} claims found, top-k avg -> {}",
            len(claims_found), avg
        )
        return avg

    def _normalize_score(self, value) -> float:
        """
        Coerce any input to a float in [0.0, 1.0].
        - Non-numeric -> 0.0
        - NaN/Inf -> 0.0 (then clamp)
        - Always returns a Python float
        """
        try:
            v = float(value)  # Coerce to float first
        except Exception:
            return 0.0

        # Handle NaN / Inf cleanly
        if v != v or v == float("inf") or v == float("-inf"):
            v = 0.0

        # Clamp to [0, 1]
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0

        # Ensure exact float type (not numpy scalar, etc.)
        return float(v)

    def _extract_hf_claims(self, model: ModelData) -> List[Dict[str, Any]]:
        """Extract performance claims from HuggingFace metadata, including model-index."""
        claims: List[Dict[str, Any]] = []
        try:
            hf_meta = model.hf_metadata
            if not hf_meta:
                return claims

            # Accept both snake_case and camelCase metadata keys
            card_data = hf_meta.get("card_data") or hf_meta.get("cardData") or {}
            if isinstance(card_data, dict):
                text_fields = [
                    card_data.get("model_description", ""),
                    card_data.get("model_summary", ""),
                    card_data.get("limitations", ""),
                    card_data.get("training_data", ""),
                    card_data.get("intended_uses", ""),
                    card_data.get("results", ""),
                ]
                for field_text in text_fields:
                    if field_text:
                        field_claims = self._find_performance_claims(field_text)
                        claims.extend(field_claims)

            # Parse structured results in model-index / model_index if available
            model_index = hf_meta.get("model-index") or hf_meta.get("model_index") or []
            if isinstance(model_index, list):
                for entry in model_index:
                    results = (entry.get("results") or [])
                    for res in results:
                        dataset = (res.get("dataset") or {})
                        dataset_name = (
                            dataset.get("name")
                            or dataset.get("type")
                            or dataset.get("id")
                            or ""
                        )
                        metrics = (res.get("metrics") or [])
                        for m in metrics:
                            mname = (m.get("name") or m.get("type") or "").lower()
                            mval = m.get("value")
                            if isinstance(mval, (int, float)) and mname:
                                claims.append({
                                    "text": f"{mname}: {mval} on {dataset_name}",
                                    "metric": str(mval),
                                    "source": "hf_model_index",
                                    "context": dataset_name.lower(),
                                })

            # Tags with obvious metric names
            tags = hf_meta.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str) and any(t in tag.lower()
                        for t in ["accuracy", "f1", "bleu", "rouge"]):
                        claims.append({
                            "text": tag,
                            "source": "hf_tags",
                            "context": "model tags",
                        })

        except Exception as e:
            logger.debug("Error extracting HuggingFace claims: {}", e)
        return claims

    def _extract_github_claims(self, model: ModelData) -> List[Dict[str, Any]]:
        """Extract performance claims from GitHub metadata (description only)."""
        claims: List[Dict[str, Any]] = []
        try:
            gh_meta = model.github_metadata
            if not gh_meta:
                return claims
            description = gh_meta.get("description", "")
            if description:
                claims.extend(self._find_performance_claims(description))
        except Exception as e:
            logger.debug("Error extracting GitHub claims: {}", e)
        return claims

    def _find_performance_claims(self, text: str) -> List[Dict[str, Any]]:
        """Find performance claims in text using tightened regex patterns and noise filters."""
        claims: List[Dict[str, Any]] = []
        if not text or not isinstance(text, str):
            return claims

        text_lower = text.lower()

        def _window(start: int, end: int, pad: int = 30) -> str:
            s = max(0, start - pad)
            e = min(len(text_lower), end + pad)
            return text_lower[s:e]

        # Metric matches
        for pattern in self.PERFORMANCE_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                num = None
                if match.lastindex:
                    # choose the last captured numeric-looking group to be robust across patterns
                    for gi in range(match.lastindex, 0, -1):
                        g = match.group(gi)
                        if g and re.fullmatch(r"\d+\.?\d*", g):
                            num = g
                            break
                context = _window(match.start(), match.end())
                if num and self._looks_like_noise(num, context):
                    continue
                claims.append({
                    "text": match.group(0),
                    "metric": num,
                    "source": "text_analysis",
                    "context": "performance metric",
                })

        # Dedup benchmark mentions (each dataset contributes at most once)
        seen_benchmarks = set()
        for pattern in self.BENCHMARK_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                key = m.group(0)
                if key not in seen_benchmarks:
                    seen_benchmarks.add(key)
                    claims.append({
                        "text": key,
                        "source": "text_analysis",
                        "context": "benchmark dataset",
                    })
        return claims

    def _looks_like_noise(self, num_str: str, context: str) -> bool:
        """Heuristics to skip numbers that are likely not evaluation metrics."""
        try:
            val = float(num_str)
        except Exception:
            return True

        # Years near the match
        if re.search(r"\b(19|20)\d{2}\b", context):
            return True

        # Version-like strings:
        # - Allow single-decimal numbers (e.g., 0.92) which are common for metrics.
        # - Treat ONLY semantic versions with two dots (e.g., 1.2.3) as noise,
        #   or explicit mentions of the word 'version'.
        if re.search(r"\b(?:v(?:er)?\.?\s*)?\d+\.\d+\.\d+\b", context):
            return True
        if re.search(r"\bversion\b", context):
            return True

        # Extremely large numbers (nonsense for typical metric values)
        if val > 1000:
            return True

        return False

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
        text = (claim.get("text") or "").lower()

        # Quantified?
        has_number = bool(re.search(r"\d+\.?\d*", text))

        # Mentions a benchmark?
        has_benchmark = any(pat in text for pat in self.BENCHMARK_PATTERNS)

        # Specific metric?
        is_performance_metric = any(metric in text for metric in [
            "accuracy", "f1", "precision", "recall",
            "auc", "bleu", "rouge", "perplexity", "loss"
        ])

        if has_number and is_performance_metric and has_benchmark:
            return self._normalize_score(1.0)
        elif has_number and is_performance_metric:
            return self._normalize_score(0.8)
        elif is_performance_metric and has_benchmark:
            return self._normalize_score(0.6)
        elif is_performance_metric:
            return self._normalize_score(0.4)
        elif has_benchmark:
            return self._normalize_score(0.2)
        else:
            return self._normalize_score(0.0)
