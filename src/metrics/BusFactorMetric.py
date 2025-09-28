"""
BusFactorMetric.py
==================
Estimates repository "bus factor" risk—the likelihood that project knowledge is
concentrated in very few people.

Signal
------
- Strong positive when there are multiple active maintainers and a healthy
  spread of recent contributors.
- Negative when commits/issues/ownership are dominated by one person or a
  very small group.

Inputs (from context)
---------------------
- code_repo: dict with fields like:
    {
      "contributors": [str, ...] or [{"login": str, "contributions": int}, ...],
      "last_commit_iso": str | None,
      "forks": int,
      "stars": int,
      ...
    }

Heuristics (illustrative)
-------------------------
- n_active = number of recent contributors (e.g., last 3–6 months)
- dominance_ratio = top_contributor_contribs / total_recent_contribs
- Consider org ownership vs. single maintainer, bus-factor-relevant metadata.

Scoring (0–1)
-------------
- 1.0 : n_active >= 3 and dominance_ratio <= 0.5
- 0.5 : n_active in {2,3} or dominance_ratio <= 0.75
- 0.0 : n_active <= 1 or dominance_ratio > 0.9

Limitations
-----------
- API data may omit private contributors or skew recency windows.
- Fork-based signals (stars/forks) are weak proxies; use cautiously.
- Prefer recency windows and normalize for repo size when possible.
"""


from loguru import logger

from src.ModelData import ModelData
from src.Metric import Metric


class BusFactorMetric(Metric):
    LARGE_COMPANIES = {
        "google", "facebook", "microsoft", "openai", "huggingface",
        "amazon", "ibm", "apple", "tencent", "baidu"
    }

    MAX_TOP_CONTRIBS = 10
    SCORE_PER_CONTRIB = 0.1

    def evaluate(self, model: ModelData) -> float:
        logger.debug("Evaluating BusFactorMetric...")

        # Get author/org from HuggingFace metadata
        hf_metadata = model.hf_metadata
        author = ""
        if hf_metadata:
            author = (
                hf_metadata.get("author")
                or hf_metadata.get("id", "").split("/")[0]
                or ""
            )
        logger.debug("Extracted author from HuggingFace metadata: '{}'", author)

        # Return full score if author is a large company
        if author in self.LARGE_COMPANIES:
            logger.debug(
                "Author '{}' is known large company, returning full score 1.0",
                author
            )
            return 1.0

        # Get contributors from Github metadata
        github_metadata = model.github_metadata
        contributors = (
            github_metadata.get("contributors", [])
            if github_metadata else []
        )
        logger.debug("Number of contributors found: {}", len(contributors))

        # No contributor data
        if not contributors:
            logger.debug("No contributors found, returning score 0.0")
            return 0.0

        # Top contributors sorted by contributions descending
        top_contribs = sorted(
            contributors,
            key=lambda c: c.get("contributions", 0),
            reverse=True
        )[:self.MAX_TOP_CONTRIBS]

        # Zero contributors -> minimum score
        num_contribs = len(top_contribs)
        if num_contribs == 0:
            logger.debug("No contributors found, returning score 0.0")
            return 0.0

        contrib_counts = [c.get("contributions", 0) for c in top_contribs]
        max_contrib = max(contrib_counts)
        min_contrib = min(contrib_counts)

        # Avoid division by zero if max_contrib == 0
        if max_contrib == 0:
            distribution_score = 0.0
        else:
            distribution_score = min_contrib / max_contrib  # between 0 and 1

        # Cap max score based on contributor count
        max_score = min(num_contribs, self.MAX_TOP_CONTRIBS) * self.SCORE_PER_CONTRIB

        # Final score: scale distribution_score by max_score
        score = distribution_score * max_score
        logger.debug("Final BusFactor score computed: {}", score)

        return score
