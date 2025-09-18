import requests
from loguru import logger

from src.Interfaces import ModelData
from src.Metric import Metric


class LicenseMetric(Metric):
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
        Evaluate the license compatibility of the provided model, dataset, or code link
        against the LGPL-2.1 license used by this project.

        This method currently focuses on the `codeLink`, checking its license (via
        GitHub API if applicable), and comparing it against a predefined list of
        compatible and incompatible licenses with respect to LGPL-2.1.

        Returns a float score in the range [0.0, 1.0]:
            - 1.0: License is known and fully compatible with LGPL-2.1
            - 0.5: License is unknown, ambiguous, or cannot be determined
            - 0.0: License is known and incompatible with LGPL-2.1

        Parameters:
            modelLink (str): URL to the AI/ML model (not used in current version)
            datasetLink (str): Optional URL to the dataset (not used in current version)
            codeLink (str): URL to the source code repository (preferably GitHub)

        Returns:
            float: A compatibility score between 0.0 and 1.0.
        """
        logger.info("Evaluating LicenseMetric...")

        license_id = None

        # Step 1: Try Hugging Face license from modelLink
        if model.modelLink and "huggingface.co" in model.modelLink:
            logger.debug("Trying Hugging Face license lookup...")
            license_id = self._get_license_from_huggingface(model.modelLink)
            if license_id:
                logger.debug("Model License FOUND on huggingface.co")

        # Step 2: Fallback to GitHub license from codeLink
        if not license_id and model.codeLink and "github.com" in model.codeLink:
            logger.debug("Falling back to GitHub license lookup...")
            license_id = self._get_spdx_license_from_github(model.codeLink)
            if license_id:
                logger.debug("Code license FOUND on github.com")

        # Step 3: Return score based on SPDX ID (or default to unknown)
        license_score = self.LICENSE_COMPATIBILITY.get(license_id, 0.5)
        logger.info("LicenseMetric: {} -> {}", license_id or "UNKNOWN", license_score)
        return license_score

    def _get_license_from_huggingface(self, model_url: str) -> str:
        """
        Extracts license from Hugging Face model metadata using their public API.
        Example URL: https://huggingface.co/facebook/bart-large -> 'facebook/bart-large'
        """
        try:
            # Parse model ID from URL
            parts = model_url.rstrip("/").split("/")
            if len(parts) < 2:
                return ""

            namespace = parts[-2]
            model_id = parts[-1]
            repo_id = f"{namespace}/{model_id}"

            api_url = f"https://huggingface.co/api/models/{repo_id}"
            response = requests.get(api_url)

            if response.status_code == 200:
                data = response.json()
                card_data = data.get("cardData", {})
                return card_data.get("license", "")
            return ""
        except Exception as e:
            logger.error("Failed to get license from Hugging Face: {}", str(e))
            return ""

    def _get_spdx_license_from_github(self, repo_url: str) -> str:
        """
        Extracts the SPDX license ID from a GitHub repo using the GitHub API.
        Assumes repo_url is in the format https://github.com/{owner}/{repo}
        """
        try:
            if "github.com" not in repo_url:
                return ""

            parts = repo_url.rstrip("/").split("/")
            if len(parts) < 5:
                return ""

            owner, repo = parts[3], parts[4]
            api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
            headers = {
                "Accept": "application/vnd.github.v3+json"
            }

            response = requests.get(api_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("license", {}).get("spdx_id", "")
            else:
                return ""
        except Exception:
            return ""
