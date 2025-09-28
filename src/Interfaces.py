"""
Interfaces.py
=============
Lightweight typing helpers (Protocols / TypedDicts / type aliases) that describe
the shapes passed between core components (Model, metrics, utilities).

Purpose
-------
- Centralize commonly used structural types so modules agree on field names.
- Improve editor autocomplete and static analysis without introducing heavy
  runtime dependencies.

Typical Types
-------------
- URL grouping:
    UrlsDict: dict[str, str | None] with keys "model", "code", "dataset"

- Metadata bundle (example fields; actual implementation may vary):
    CodeRepoMeta (TypedDict):
        - name: str
        - full_name: str
        - license: str | None
        - stars: int
        - forks: int
        - last_commit_iso: str | None
        - contributors: list[str] | None

    ModelCardMeta (TypedDict):
        - repo_id: str
        - tags: list[str]
        - card_text: str | None

    DatasetMeta (TypedDict):
        - url: str
        - reachable: bool
        - description: str | None

- Evaluation context passed to metrics:
    MetricContext (TypedDict, total=False):
        - urls: UrlsDict
        - code_repo: CodeRepoMeta | None
        - model_card: ModelCardMeta | None
        - dataset: DatasetMeta | None
        - github_token: str | None
        - extra: dict[str, object]

Usage
-----
Modules should import these types for annotations only; they should not impose
strict runtime checks. Keep names stable to avoid churn across the codebase.

Notes
-----
This file is intentionally narrow in scope: define only what multiple modules
consume. If a type becomes metric-specific, prefer declaring it in that metric’s
module to avoid coupling.
"""


from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol


class ModelData(Protocol):

    modelLink: str
    codeLink: Optional[str]
    datasetLink: Optional[str]

    @property
    @abstractmethod
    def hf_metadata(self) -> Optional[Dict[str, Any]]:
        """Cached HuggingFace metadata"""
        ...

    @property
    @abstractmethod
    def github_metadata(self) -> Optional[Dict[str, Any]]:
        """Cached GitHub metadata"""
        ...

    @property
    @abstractmethod
    def dataset_metadata(self) -> Optional[Dict[str, Any]]:
        """Cached Dataset metadata"""
        ...
