import pytest
from unittest.mock import patch

from src.metrics.LicenseMetric import LicenseMetric


@pytest.fixture
def metric():
    return LicenseMetric()


@patch("src.metrics.LicenseMetric.requests.get")
def test_compatible_license(mock_get, metric):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "license": {"spdx_id": "MIT"}
    }

    score = metric.evaluate(codeLink="https://github.com/someuser/somerepo")
    assert score == 1.0


@patch("src.metrics.LicenseMetric.requests.get")
def test_incompatible_license(mock_get, metric):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "license": {"spdx_id": "GPL-3.0"}
    }

    score = metric.evaluate(codeLink="https://github.com/someuser/somerepo")
    assert score == 0.0


@patch("src.metrics.LicenseMetric.requests.get")
def test_unknown_license(mock_get, metric):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "license": {"spdx_id": "Some-Strange-License"}
    }

    score = metric.evaluate(codeLink="https://github.com/someuser/somerepo")
    assert score == 0.5


@patch("src.metrics.LicenseMetric.requests.get")
def test_no_license_field(mock_get, metric):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {}

    score = metric.evaluate(codeLink="https://github.com/someuser/somerepo")
    assert score == 0.5


def test_non_github_url(metric):
    score = metric.evaluate(codeLink="https://gitlab.com/some/repo")
    assert score == 0.5


def test_empty_code_link(metric):
    score = metric.evaluate(codeLink="")
    assert score == 0.5
