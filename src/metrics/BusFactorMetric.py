from src.Interfaces import ModelData
from src.Metric import Metric


class BusFactorMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        # Implement bus factor evaluation logic here
        return 0.0
