from abc import ABC, abstractmethod
from typing import Union


class Metric(ABC):
    @abstractmethod
    def evaluate(
        self,
        modelLink: str = "",
        datasetLink: str = "",
        codeLink: str = ""
    ) -> Union[float, dict[str, float]]:
        pass
