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
