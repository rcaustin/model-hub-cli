"""
BusFactorMetric.py
==================

Estimates the "bus factor" risk — the risk that project knowledge and
maintenance responsibility is concentrated in too few contributors.

Signal
------
- Positive signal when a project has multiple active contributors sharing
  the workload.
- Negative signal when commits and contributions are dominated by one or
  very few individuals.

Inputs (from context)
---------------------
- GitHub repository metadata including:
    - contributors list with contribution counts
    - repository metadata such as stars, forks, and recent activity
- HuggingFace metadata for author or organization information

Heuristics (illustrative)
-------------------------
- Number of active contributors (top 10 considered)
- Distribution of contributions among top contributors (min/max ratio)
- Known large organizations automatically scored as low risk

Scoring (0–1)
-------------
- 1.0 if at least 3 contributors with relatively balanced contributions
  (dominance ratio <= 0.5)
- 0.5 if 2-3 contributors or dominance ratio <= 0.75
- 0.0 if 1 or fewer contributors or dominance ratio > 0.9

Limitations
-----------
- Contributor data may exclude private or minor contributors
- Stars and forks are weak proxies for bus factor risk
- Recency and normalization by project size are not considered but can
  improve accuracy

Note
----
This metric corresponds to the "Bus Factor" metric in the specification,
providing an estimate of risk associated with knowledge centralization
within the maintainers of a project.
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
