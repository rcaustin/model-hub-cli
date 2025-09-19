from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric


class LicenseMetric(Metric):
    # 1.0: License is known and fully compatible
    # 0.5: License is unknown, ambiguous, or cannot be determined
    # 0.0: License is known and incompatible
    LICENSE_COMPATIBILITY = {
        "MIT": 1.0,
        "BSD-2-Clause": 1.0,
        "BSD-3-Clause": 1.0,
        "LGPL-2.1": 1.0,
        "LGPL-2.1-or-later": 1.0,
        "GPL-2.0-or-later": 1.0,

        "GPL-2.0": 0.5,
        "MPL-2.0": 0.5,
        "Unlicense": 0.5,

        "Apache-2.0": 0.0,
        "GPL-3.0": 0.0,
        "LGPL-3.0": 0.0,
        "AGPL-3.0": 0.0,
        "Proprietary": 0.0
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
        logger.info("Evaluating LicenseMetric...")
        license_id = None

        # Step 1: Try HuggingFace metadata
        if model.modelLink and "huggingface.co" in model.modelLink:
            logger.debug("Checking HuggingFace metadata for license...")
            hf_meta = model.hf_metadata
            if hf_meta:
                license_id = hf_meta.get("cardData", {}).get("license")
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
                license_info = gh_meta if isinstance(gh_meta, dict) else {}
                license_id = license_info.get("license", "")
                if license_id:
                    logger.debug("License found in GitHub metadata: {}", license_id)

        # Step 3: Map license to score
        license_score = self.LICENSE_COMPATIBILITY.get(license_id, 0.5)
        logger.info("LicenseMetric: {} -> {}", license_id or "UNKNOWN", license_score)
        return license_score
