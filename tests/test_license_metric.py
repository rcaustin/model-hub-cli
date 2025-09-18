import pytest
from unittest.mock import patch
from dataclasses import dataclass
from typing import Optional

from src.metrics.LicenseMetric import LicenseMetric


@dataclass
class StubModelData:
    modelLink: str
    codeLink: Optional[str]
    datasetLink: Optional[str]


@pytest.fixture
def metric():
    return LicenseMetric()


@pytest.fixture
def base_model():
    """Base stub model with common fields (overridden in tests)."""
    return StubModelData(
        modelLink="https://huggingface.co/some/model",
        codeLink=None,
        datasetLink="https://huggingface.co/datasets/some/dataset"
    )


@patch("src.metrics.LicenseMetric.requests.get")
def test_compatible_license(mock_get, metric, base_model):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "license": {"spdx_id": "MIT"}
    }

    base_model.codeLink = "https://github.com/someuser/somerepo"
    score = metric.evaluate(base_model)
    assert score == 1.0


@patch("src.metrics.LicenseMetric.requests.get")
def test_incompatible_license(mock_get, metric, base_model):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "license": {"spdx_id": "GPL-3.0"}
    }

    base_model.codeLink = "https://github.com/someuser/somerepo"
    score = metric.evaluate(base_model)
    assert score == 0.0


@patch("src.metrics.LicenseMetric.requests.get")
def test_unknown_license(mock_get, metric, base_model):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "license": {"spdx_id": "Some-Weird-License"}
    }

    base_model.codeLink = "https://github.com/someuser/somerepo"
    score = metric.evaluate(base_model)
    assert score == 0.5


@patch("src.metrics.LicenseMetric.requests.get")
def test_no_license_field(mock_get, metric, base_model):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {}

    base_model.codeLink = "https://github.com/someuser/somerepo"
    score = metric.evaluate(base_model)
    assert score == 0.5


def test_non_github_url(metric, base_model):
    base_model.codeLink = "https://gitlab.com/some/repo"
    score = metric.evaluate(base_model)
    assert score == 0.5


def test_empty_code_link(metric, base_model):
    base_model.codeLink = ""
    score = metric.evaluate(base_model)
    assert score == 0.5
