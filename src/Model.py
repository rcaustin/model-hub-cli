from typing import Optional, Union
from Metric import Metric


class Model:

    """
    evaluations maps metric names to their scores.
    Scores can be a float or a dictionary of floats for complex metrics.
    """
    def __init__(
        self,
        name: str,
        modelLink: str,
        codeLink: str,
        datasetLink: str,
        evaluations: Optional[dict[str, Union[float, dict[str, float]]]]
    ):
        self.name = name
        self.modelLink = modelLink
        self.codeLink = codeLink
        self.datasetLink = datasetLink
        self.evaluations = evaluations if evaluations is not None else {}

    def evaluate(self, metric: Metric):
        score: Union[float, dict[str, float]] = metric.evaluate()
        self.evaluations[type(metric).__name__] = score

    def getEvals(self) -> dict[str, Union[float, dict[str, float]]]:
        return self.evaluations
