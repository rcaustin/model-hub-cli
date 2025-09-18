from typing import Protocol, Optional


class ModelData(Protocol):
    modelLink: str
    codeLink: Optional[str]
    datasetLink: Optional[str]
