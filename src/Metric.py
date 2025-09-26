from abc import ABC, abstractmethod
from typing import Union

from src.Interfaces import ModelData


class Metric(ABC):
    @abstractmethod
    def evaluate(self, model: ModelData) -> Union[float, dict[str, float]]:
        pass
