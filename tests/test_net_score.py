"""
Unit tests for computing NetScore in the Model class.
"""

import pytest
from src.Model import Model
from src.util.url_utils import URLSet


def make_mock_model(evals: dict) -> Model:
    """Helper to create a Model with predefined evaluation scores."""
    urls = URLSet(
        model="https://huggingface.co/models/foo",
        code="https://github.com/foo/bar",
        dataset="https://huggingface.co/datasets/bar"
    )
    model = Model(urls)
    model.evaluations = evals
    return model


@pytest.mark.parametrize("evals, expected", [
    # All metrics present
    ({
        "LicenseMetric": 1.0,
        "SizeMetric": {"average": 0.9},
        "RampUpMetric": 0.8,
        "BusFactorMetric": 0.7,
        "AvailabilityMetric": 0.6,
        "DatasetQualityMetric": 0.5,
        "CodeQualityMetric": 0.4,
        "PerformanceClaimsMetric": 0.3
    }, 1.0 * (0.2 * 0.9 + 0.3 * 0.8 + 0.1 * 0.7 + 0.1 * 0.6 + 0.1 * 0.5 + 0.1 * 0.4 +
              0.1 * 0.3)),

    # Missing metrics default to 0.0
    ({
        "LicenseMetric": 0.5,
        "SizeMetric": {"average": 0.8},
        "RampUpMetric": 0.6
    }, 0.5 * (0.2 * 0.8 + 0.3 * 0.6)),

    # No LicenseMetric -> NetScore is 0
    ({
        "SizeMetric": {"average": 1.0},
        "RampUpMetric": 1.0,
        "BusFactorMetric": 1.0,
        "AvailabilityMetric": 1.0,
        "DatasetQualityMetric": 1.0,
        "CodeQualityMetric": 1.0,
        "PerformanceClaimsMetric": 1.0
    }, 0.0),

    # SizeMetric missing "average" key -> treat as 0.0
    ({
        "LicenseMetric": 1.0,
        "SizeMetric": {},  # Missing "average"
        "RampUpMetric": 1.0,
        "BusFactorMetric": 1.0,
        "AvailabilityMetric": 1.0,
        "DatasetQualityMetric": 1.0,
        "CodeQualityMetric": 1.0,
        "PerformanceClaimsMetric": 1.0
    }, 1.0 * (0.3 * 1.0 + 0.1 * 1.0 + 0.1 * 1.0 + 0.1 * 1.0 + 0.1 * 1.0 + 0.1 * 1.0)),
])
def test_compute_netscore(evals, expected):
    model = make_mock_model(evals)
    actual = model.computeNetScore()
    assert actual == pytest.approx(expected, abs=1e-6)
