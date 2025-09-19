from src.Interfaces import ModelData
from src.Metric import Metric


class SizeMetric(Metric):
    def evaluate(self, model: ModelData) -> dict[str, float]:
        # Implement size evaluation logic here
        return {}
