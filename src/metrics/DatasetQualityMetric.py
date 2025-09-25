import math
import re
from typing import Dict, Tuple

from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except Exception:  # keep project import-safe if dependency isn't installed yet
    HfApi = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore


class DatasetQualityMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        """
        Computes dataset quality based on HuggingFace dataset metadata:
        - Documentation completeness (sections present in README)
        - Citations/references
        - Social proof (likes, downloads)
        - Optional: dataset size if discoverable
        Returns a score between 0.0 and 1.0.
        """
        # --- Helper functions (nested so they don’t leak globally) ---
        def _clamp01(x: float) -> float:
            return 0.0 if x < 0 else 1.0 if x > 1 else x

        def _docs_completeness_score(readme_md: str) -> Tuple[float, Dict[str, bool]]:
            sections = [
                "dataset summary", "intended uses", "dataset structure",
                "data instances", "data fields", "bias", "limitations",
                "citation", "references"
            ]
            text = (readme_md or "").lower()
            found = sum(1 for sec in sections if sec in text)
            return (
                _clamp01(found / len(sections)),
                {sec: (sec in text) for sec in sections}
            )

        def _count_citations(md: str) -> int:
            patterns = [
                r"arxiv\.org/abs/\d{4}\.\d{4,5}",
                r"\bdoi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+",
            ]
            return sum(len(re.findall(p, md or "", flags=re.I)) for p in patterns)

        def _logish(x: float, denom: float) -> float:
            return _clamp01(math.log10(x + 1.0) / denom) if x > 0 else 0.0

        # --- Resolve repo id ---
        repo_id = getattr(model, "dataset_id", None) or \
            getattr(model, "dataset_url", None)
        if not repo_id:
            logger.error(
                "DatasetQualityMetric: No dataset id/url provided on ModelData"
            )
            return 0.0

        if HfApi is None:
            logger.error("DatasetQualityMetric: huggingface_hub not installed")
            return 0.0

        api = HfApi()
        try:
            info = api.dataset_info(repo_id)
        except HfHubHTTPError as e:
            logger.error(
                f"DatasetQualityMetric: failed to fetch dataset_info for {repo_id}: {e}"
            )
            return 0.0

        likes = getattr(info, "likes", 0) or 0
        downloads = getattr(info, "downloads", 0) or 0
        try:
            readme_md = api.get_repo_readme(repo_id=repo_id, repo_type="dataset") or ""
        except Exception:
            readme_md = ""

        # --- Component scores ---
        docs_score, _ = _docs_completeness_score(readme_md)
        cites = _count_citations(readme_md)
        citations_score = _clamp01(cites / 5.0)
        likes_score = _logish(likes, denom=3.0)
        downloads_score = _logish(downloads, denom=6.0)

        # --- Weighted blend ---
        score = (
            0.35 * docs_score +
            0.20 * citations_score +
            0.20 * likes_score +
            0.25 * downloads_score
        )
        score = _clamp01(score)

        logger.debug(
            f"DatasetQualityMetric: repo={repo_id}, score={score:.3f}, "
            f"docs={docs_score:.2f}, cites={citations_score:.2f} ({cites}), "
            f"likes={likes} ({likes_score:.2f}),\
                downloads={downloads} ({downloads_score:.2f})"
        )
        return score
