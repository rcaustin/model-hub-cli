"""
LicenseMetric.py
================

Evaluates license compatibility of a model using cached metadata.

Inputs (from context)
---------------------
- HuggingFace model metadata (hf_metadata)
- GitHub repository metadata (github_metadata)

Scoring (0.0 – 1.0)
-------------------
- 1.0: Known license and fully compatible (e.g., MIT, Apache-2.0)
- 0.5: Unknown, ambiguous, or undetermined license
- 0.0: Known but incompatible license (e.g., GPL-3.0, AGPL-3.0, Proprietary)

Process
-------
1. Check license from HuggingFace metadata if available.
2. If not found, fallback to GitHub repository license.
3. Map detected license to compatibility score.

Limitations
-----------
- Only uses cached metadata, no live license detection.
- May misclassify if license metadata is missing or inconsistent.
"""

from typing import Dict

from loguru import logger

from src.Metric import Metric
from src.ModelData import ModelData


class LicenseMetric(Metric):
    # 1.0: License is known and fully compatible
    # 0.5: License is unknown, ambiguous, or cannot be determined
    # 0.0: License is known and incompatible
    LICENSE_COMPATIBILITY : Dict[str, float] = {
        "mit": 1.0,
        "bsd-2-clause": 1.0,
        "bsd-3-clause": 1.0,
        "lgpl-2.1": 1.0,
        "lgpl-2.1-or-later": 1.0,
        "gpl-2.0-or-later": 1.0,
        "apache-2.0": 1.0,

        "gpl-2.0": 0.5,
        "mpl-2.0": 0.5,
        "unlicense": 0.5,

        "gpl-3.0": 0.0,
        "lgpl-3.0": 0.0,
        "agpl-3.0": 0.0,
        "proprietary": 0.0
    }

    def evaluate(self, model: ModelData) -> float:
        """
        Evaluate the license compatibility of the model using cached metadata.

        Priority:
            1. HuggingFace model license (via hf_metadata)
            2. GitHub repo license (via github_metadata)

        Returns:
            float: Score from 0.0 (incompatible) to 1.0 (fully compatible).
        """
        logger.debug("Evaluating LicenseMetric...")
        license_id : str = "unknown"  # Default to unknown

        # Step 1: Try HuggingFace metadata
        if model.modelLink and "huggingface.co" in model.modelLink:
            logger.debug("Checking HuggingFace metadata for license...")
            hf_meta = model.hf_metadata
            if hf_meta:
                license_id = hf_meta.get("cardData", {}).get("license", "unknown")
                if license_id:
                    logger.debug(
                        "License found in HuggingFace metadata: {}",
                        license_id
                    )

        # Step 2: Fallback to GitHub metadata
        if not license_id and model.codeLink and "github.com" in model.codeLink:
            logger.debug("Falling back to GitHub metadata for license...")
            gh_meta = model.github_metadata
            if gh_meta:
                license_info = gh_meta if gh_meta else {}
                license_id = license_info.get("license", "unknown")
                if license_id:
                    logger.debug(
                        "License found in GitHub metadata: {}", license_id
                    )

        # Step 3: Map license to score
        license_score : float = self.LICENSE_COMPATIBILITY.get(license_id.lower(), 0.5)
        logger.debug("LicenseMetric: {} -> {}", license_id or "UNKNOWN", license_score)
        return license_score
