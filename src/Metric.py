"""
Metric.py
=========
Defines the metric protocol/base class used by all concrete metrics under
``src/metrics/``. A metric consumes a prepared **context** and returns:

    (score: float, latency_ms: int)

where ``score`` is typically normalized to [0.0, 1.0] and ``latency_ms`` is the
milliseconds spent computing the metric.

Responsibilities
----------------
- Provide a consistent interface for all metrics.
- Expose a human-readable ``name`` and a numeric ``weight`` for NetScore
  aggregation.
- Define ``evaluate(context)`` which must be implemented by subclasses.

Expected Interface
------------------
- ``name: str`` — short identifier (e.g., "license", "ramp_up")
- ``weight: float`` — contribution toward the composite NetScore
- ``def evaluate(self, context) -> tuple[float, int]:``
    Compute and return ``(score, latency_ms)``.

Context
-------
The ``context`` is constructed by ``Model`` and commonly includes:
- grouped URLs: ``{"model": str|None, "code": str|None, "dataset": str|None}``
- fetched metadata for code/model/dataset
- any tokens/clients (e.g., GITHUB_TOKEN)
- lightweight caches or config flags

Implementation Notes
--------------------
- Implementations **must not** raise uncaught exceptions; return conservative
  scores (e.g., 0.0) and log warnings on unexpected conditions.
- Keep runtime bounded and deterministic for unit tests.
- Document the metric’s rubric and limitations in the module-level docstring
  of each concrete metric.

Testing Guidance
----------------
- Ensure 0.0 ≤ score ≤ 1.0 (unless justified otherwise).
- ``latency_ms`` should be a non-negative ``int``.
- Validate behavior for positive, partial, and negative evidence cases.
"""

from abc import ABC, abstractmethod
from typing import Union
from src.Interfaces import ModelData


class Metric(ABC):
    @abstractmethod
    def evaluate(self, model: ModelData) -> Union[float, dict[str, float]]:
        pass
