from src.Interfaces import ModelData
from src.Metric import Metric


class BusFactorMetric(Metric):
    LARGE_COMPANIES = {
        "google",
        "facebook",
        "microsoft",
        "openai",
        "huggingface",
        "amazon",
        "ibm",
        "apple",
        "tencent",
        "baidu"
    }

    def evaluate(self, model: ModelData) -> float:
        # Get author/org from HuggingFace metadata
        hf_metadata = model.hf_metadata
        author = ""
        if hf_metadata:
            author = (
                hf_metadata.get("author") or
                hf_metadata.get("id", "").split("/")[0] or
                ""
            )

        # Return full score if author is a large company
        if author in self.LARGE_COMPANIES:
            return 1.0

        # Get contributors from Github metadata
        github_metadata = model.github_metadata
        contributors = (
            github_metadata.get("contributors", [])
            if github_metadata else []
        )

        # No contributor data
        if not contributors:
            return 0.0

        # Sort contributors by contributions, descending
        sorted_contribs = sorted(
            contributors,
            key=lambda c: c.get("contributions", 0),
            reverse=True
        )
        top_contribs = sorted_contribs[:10]
        num_contribs = len(top_contribs)
        if num_contribs == 0:
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
        max_score = min(num_contribs, 10) * 0.1

        # Final score: scale distribution_score by max_score
        score = distribution_score * max_score

        return score
