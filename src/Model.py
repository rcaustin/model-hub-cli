import time
from typing import Union

from src.Metric import Metric


class Model:

    def __init__(
        self,
        modelLink: str,
        codeLink: str | None = None,
        datasetLink: str | None = None
    ):
        self.name = None
        self.modelLink = modelLink
        self.codeLink = codeLink
        self.datasetLink = datasetLink

        """
        evaluations maps metric names to their scores.
        Scores can be a float or a dictionary of floats for complex metrics.
        evaluationsLatency maps metric names to the time taken to compute them.
        """
        self.evaluations: dict[str, Union[float, dict[str, float]]] = {}
        self.evaluationsLatency: dict[str, float] = {}

    def evaluate(self, metric: Metric) -> None:

        # Evaluate the given metric and record its score and evaluation time.
        start: float = time.time()
        score: Union[float, dict[str, float]] = metric.evaluate()
        end: float = time.time()
        elapsed: float = end - start

        # Record the evaluation results.
        metric_name: str = type(metric).__name__
        self.evaluations[metric_name] = score
        self.evaluationsLatency[metric_name] = elapsed

    def getEvals(self) -> dict[str, Union[float, dict[str, float]]]:
        return self.evaluations

    def getEvalsLatency(self) -> dict[str, float]:
        return self.evaluationsLatency
    
    def getCategory(self) -> str:
        categories = []
        if self.modelLink:
            categories.append("MODEL")
        if self.datasetLink:
            categories.append("DATASET")
        if self.codeLink:
            categories.append("CODE")
        return f"[{', '.join(categories)}]"

