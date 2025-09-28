"""
Metric.py
=========

Defines the abstract base class `Metric`, which serves as the interface for all
model evaluation metrics used in the scoring system.

Responsibilities
----------------
- Enforce a standard interface for all metric implementations via the abstract
  `evaluate()` method.
- Allow individual metrics to encapsulate their own logic for evaluating a `Model`.

Usage
-----
To implement a new metric:
1. Subclass `Metric`.
2. Implement the `evaluate(model)` method.
3. Return either:
   - A float score (e.g., 0.85), or
   - A dictionary of float scores for multi-target metrics
         (e.g., {"pi": 0.6, "aws": 0.9}).

Example:
--------
class MyCustomMetric(Metric):
    def evaluate(self, model: ModelData) -> float:
        # Compute score...
        return 0.75
"""

from abc import ABC, abstractmethod
from typing import Union
from src.ModelData import ModelData


class Metric(ABC):
    @abstractmethod
    def evaluate(self, model: ModelData) -> Union[float, dict[str, float]]:
        pass
