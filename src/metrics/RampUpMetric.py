from src.Interfaces import ModelData
from src.Metric import Metric


class RampUpMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        # Implement ramp-up evaluation logic here
        return 0.0
