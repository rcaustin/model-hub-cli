from unittest.mock import Mock, patch

import pytest

from src.metrics.LicenseMetric import LicenseMetric
from tests.conftest import StubModelData


@pytest.fixture
def metric():
    return LicenseMetric()


@pytest.fixture
def hf_model():
    return StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink="https://github.com/huggingface/transformers",
        datasetLink="https://huggingface.co/datasets/squad"
    )


@pytest.fixture
def github_only_model():
    return StubModelData(
        modelLink="https://somewhere.else/model",
        codeLink="https://github.com/someuser/somerepo",
        datasetLink=None
    )


# HuggingFace License Tests
@patch("src.metrics.LicenseMetric.requests.get")
def test_license_from_huggingface(mock_get, metric, hf_model):
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"cardData": {"license": "MIT"}}

    score = metric.evaluate(hf_model)
    assert score == 1.0


@patch("src.metrics.LicenseMetric.requests.get")
def test_unknown_license_from_huggingface(mock_get, metric, hf_model):
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"cardData": {"license": "unknown"}}

    score = metric.evaluate(hf_model)
    assert score == 0.5


# Github Fallback Tests
@patch("src.metrics.LicenseMetric.requests.get")
def test_license_from_github_when_hf_fails(mock_get, metric, github_only_model):
    # Simulate Hugging Face failure (1st call)
    # Simulate GitHub success (2nd call)
    def side_effect(url, *args, **kwargs):
        if "huggingface.co" in url:
            return Mock(status_code=404)
        elif "github.com" in url:
            m = Mock(status_code=200)
            m.json.return_value = {"license": {"spdx_id": "GPL-3.0"}}
            return m
        return Mock(status_code=404)

    mock_get.side_effect = side_effect

    score = metric.evaluate(github_only_model)
    assert score == 0.0


# Fallthrough / Error Cases
@patch("src.metrics.LicenseMetric.requests.get")
def test_no_license_available(mock_get, metric, github_only_model):
    mock_get.return_value = Mock(status_code=404)  # Both HF and GitHub fail

    score = metric.evaluate(github_only_model)
    assert score == 0.5


def test_no_links(metric):
    model = StubModelData(modelLink="", codeLink="", datasetLink="")
    score = metric.evaluate(model)
    assert score == 0.5


# Malformed URL Tests
def test_malformed_model_url(metric):
    model = StubModelData(
        modelLink="https://huggingface.co/",  # Invalid: no repo ID
        codeLink="",  # No fallback
        datasetLink=""
    )
    score = metric.evaluate(model)
    assert score == 0.5


def test_non_hf_model_url(metric):
    model = StubModelData(
        modelLink="not-a-real-url",
        codeLink="",  # No fallback
        datasetLink=""
    )
    score = metric.evaluate(model)
    assert score == 0.5


def test_malformed_github_url(metric):
    model = StubModelData(
        modelLink="",  # No HF check
        codeLink="https://github.com/just-owner",  # Invalid GitHub URL
        datasetLink=""
    )
    score = metric.evaluate(model)
    assert score == 0.5


def test_non_github_code_url(metric):
    model = StubModelData(
        modelLink="",  # No HF check
        codeLink="https://gitlab.com/org/repo",  # Unsupported provider
        datasetLink=""
    )
    score = metric.evaluate(model)
    assert score == 0.5
