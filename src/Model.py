import time
import os
from typing import Any, Dict, List, Optional, Union

from src.Interfaces import ModelData
from src.Metric import Metric
from src.util.metadata_fetchers import GitHubFetcher, HuggingFaceFetcher, DatasetFetcher
from src.util.url_utils import URLSet, classify_urls



class Model(ModelData):
    # -----------------------------------------
    # Shared, process-wide dictionary that all Model instances can use.
    # Keys are canonical dataset names (lowercased), values are absolute URLs.
    # This is intentionally a class variable (Python "static" style).
    
    DATASET_NAME_TO_URL: dict[str, str] = {}

    def __init__(
        self,
        urls: List[str]
    ):
        # Extract and Classify URLs
        # urlset: URLSet = classify_urls(urls)
        # Positional parsing (code, dataset, model) -> URLSet(model, code, dataset)
        urlset: URLSet = parse_urls_by_position(urls)
        self.modelLink: str = urlset.model
        self.codeLink: Optional[str] = urlset.code
        self.datasetLink: Optional[str] = urlset.dataset

        # Metadata Caching
        self._hf_metadata: Optional[Dict[str, Any]] = None
        self._github_metadata: Optional[Dict[str, Any]] = None
        self._dataset_metadata: Optional[Dict[str, Any]] = None

        # Get GitHub token from environment (validated at startup)
        self._github_token: Optional[str] = os.getenv("GITHUB_TOKEN")

        """
        evaluations maps metric names to their scores.
        Scores can be a float or a dictionary of floats for complex metrics.
        evaluationsLatency maps metric names to the time taken to compute them.
        """
        self.evaluations: dict[str, Union[float, dict[str, float]]] = {}
        self.evaluationsLatency: dict[str, float] = {}


        # --- NEW: opportunistically infer missing links ---
        # Only touch what is missing; keep explicit user-provided links authoritative.
        if self.datasetLink is None:
            self._maybe_fill_dataset_from_hf()

        if self.codeLink is None:
            self._maybe_fill_code_from_hf()


    @property
    def name(self) -> str:
        try:
            return self.hf_metadata.get("id", "").split("/")[1]
        except (AttributeError, IndexError):
            return "UNKNOWN_MODEL"

    @property
    def hf_metadata(self) -> Optional[Dict[str, Any]]:
        if self._hf_metadata is None:
            fetcher = HuggingFaceFetcher()
            self._hf_metadata = fetcher.fetch_metadata(self.modelLink)
        return self._hf_metadata

    @property
    def github_metadata(self) -> Optional[Dict[str, Any]]:
        if self._github_metadata is None:
            # Pass the validated GitHub token to the fetcher
            fetcher = GitHubFetcher(token=self._github_token)
            self._github_metadata = fetcher.fetch_metadata(self.codeLink)
        return self._github_metadata

    @property
    def dataset_metadata(self) -> Optional[Dict[str, Any]]:
        if self._dataset_metadata is None:
            fetcher = DatasetFetcher()
            self._dataset_metadata = fetcher.fetch_metadata(self.datasetLink)
        return self._dataset_metadata

    # ------------------------ NEW HELPERS ------------------------

    def _maybe_fill_dataset_from_hf(self) -> None:
        """
        If dataset URL missing, try to infer it from Hugging Face model card.
        Strategy:
          1) Read hf_metadata["card_data"]["datasets"] (string or list).
          2) Try to map each candidate name -> URL using the shared class dict.
          3) If a candidate looks like 'org/name', synthesize a HF dataset URL and cache it.
          4) First resolved URL wins; otherwise leave None.
        """
        meta = self.hf_metadata  # triggers fetch lazily
        if not meta:
            return

        card_data = meta.get("card_data") or meta.get("cardData") or {}
        candidates = card_data.get("datasets")
        if not candidates:
            return

        # Normalize to a list of strings
        if isinstance(candidates, str):
            names = [candidates]
        elif isinstance(candidates, list):
            # lists can contain strings or dicts; try to extract names
            names = []
            for item in candidates:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    # common shapes: {"id": "org/name"} or {"name": "..."}
                    val = item.get("id") or item.get("name")
                    if isinstance(val, str):
                        names.append(val)
        else:
            return

        for raw in names:
            key = raw.strip().lower()
            # 1) Look up in the shared cache
            cached = Model.DATASET_NAME_TO_URL.get(key)
            if cached:
                self.datasetLink = cached
                return

            # 2) If 'org/name' form, synthesize HF dataset URL and cache it
            if "/" in raw and len(raw.split("/")) == 2:
                org, ds = raw.split("/", 1)
                if org and ds:
                    url = f"https://huggingface.co/datasets/{org}/{ds}"
                    Model.DATASET_NAME_TO_URL[key] = url
                    self.datasetLink = url
                    return

        # If we reach here, we didn’t find anything trustworthy; leave None

    def _maybe_fill_code_from_hf(self) -> None:
        """
        If code URL missing, try a few safe heuristics from the HF card.
        We keep this conservative—only set when we’re confident.
        """
        meta = self.hf_metadata
        if not meta:
            return

        card_data = meta.get("card_data") or meta.get("cardData") or {}

        # Heuristic 1: explicit GitHub link fields some model cards include
        likely_fields = ["repository", "github", "source_repo", "code_url"]
        for f in likely_fields:
            val = card_data.get(f)
            if isinstance(val, str) and "github.com" in val:
                self.codeLink = val
                return

        # Heuristic 2: some cards put links under "links" or "paperswithcode"
        links = card_data.get("links")
        if isinstance(links, list):
            for x in links:
                if isinstance(x, str) and "github.com" in x:
                    self.codeLink = x
                    return
                if isinstance(x, dict):
                    u = x.get("url") or x.get("href")
                    if isinstance(u, str) and "github.com" in u:
                        self.codeLink = u
                        return

        # Nothing found -> leave None (graceful)

    # -------------------------------------------------------------

    def getScore(
        self, metric_name: str, default: float = 0.0
    ) -> Union[float, dict[str, float]]:
        value = self.evaluations.get(metric_name, default)
        if isinstance(value, dict):
            return {k: round(v, 2) for k, v in value.items()}
        return round(value, 2)

    def getLatency(self, metric_name: str) -> int:
        latency = self.evaluationsLatency.get(metric_name, 0.0)
        return int(latency * 1000)

    def evaluate_all(self, metrics: List[Metric]) -> None:
        for metric in metrics:
            self.evaluate(metric)
        self.computeNetScore()

    def evaluate(self, metric: Metric) -> None:
        start: float = time.time()
        score: Union[float, dict[str, float]] = metric.evaluate(self)
        end: float = time.time()

        metric_name: str = type(metric).__name__
        self.evaluations[metric_name] = score
        self.evaluationsLatency[metric_name] = end - start

    def getCategory(self) -> str:
        return "MODEL"

    def computeNetScore(self) -> float:
        def safe_score(key: str) -> float:
            val = self.evaluations.get(key)
            if key == "SizeMetric":
                # Only accept dict for SizeMetric, else 0.0
                if isinstance(val, dict) and val:
                    return sum(val.values()) / len(val)
                else:
                    return 0.0
            else:
                if isinstance(val, dict):
                    return sum(val.values()) / len(val) if val else 0.0
                return val if val is not None else 0.0

        license_score = safe_score("LicenseMetric")

        weighted_sum = (
            0.2 * safe_score("SizeMetric") +
            0.3 * safe_score("RampUpMetric") +
            0.1 * safe_score("BusFactorMetric") +
            0.1 * safe_score("AvailabilityMetric") +
            0.1 * safe_score("DatasetQualityMetric") +
            0.1 * safe_score("CodeQualityMetric") +
            0.1 * safe_score("PerformanceClaimsMetric")
        )

        net_score = license_score * weighted_sum

        self.evaluations["NetScore"] = net_score
        self.evaluationsLatency["NetScore"] = 0.0  # Derived metric, no latency
        self.evaluationsLatency["NetScore"] = sum(
            latency for key, latency in self.evaluationsLatency.items()
            if key != "NetScore"
        )
        
        return net_score
