from src.Interfaces import ModelData
from src.Metric import Metric


class AvailabilityMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        # Implement availability evaluation logic here
        return 0.0
