"""
Unit tests for the AvailabilityMetric class.
"""
from unittest.mock import Mock, patch
import pytest

from src.metrics.AvailabilityMetric import AvailabilityMetric
from tests.conftest import StubModelData


@pytest.fixture
def metric():
    return AvailabilityMetric()


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


@pytest.fixture
def dataset_only_model():
    return StubModelData(
        modelLink=None,
        codeLink=None,
        datasetLink="https://huggingface.co/datasets/squad"
    )


# Test successful availability checks
@patch("requests.get")
def test_all_resources_available(mock_get, metric, hf_model):
    """Test when all resources (model, code, dataset) are available."""
    mock_get.return_value = Mock(status_code=200)
    
    score = metric.evaluate(hf_model)
    assert score == 1.0
    assert mock_get.call_count == 3  # model, code, dataset


@patch("requests.get")
def test_partial_availability(mock_get, metric, hf_model):
    """Test when only some resources are available."""
    def side_effect(url, *args, **kwargs):
        if "huggingface.co/api/models" in url:
            return Mock(status_code=200)  # Model available
        elif "github.com" in url:
            return Mock(status_code=200)  # Code available
        elif "huggingface.co/api/datasets" in url:
            return Mock(status_code=404)  # Dataset not available
        return Mock(status_code=404)
    
    mock_get.side_effect = side_effect
    
    score = metric.evaluate(hf_model)
    assert score == 2/3  # 2 out of 3 resources available


@patch("requests.get")
def test_no_resources_available(mock_get, metric, hf_model):
    """Test when no resources are available."""
    mock_get.return_value = Mock(status_code=404)
    
    score = metric.evaluate(hf_model)
    assert score == 0.0


# Test individual resource types
@patch("requests.get")
def test_github_only_available(mock_get, metric, github_only_model):
    """Test when only GitHub repository is available."""
    def side_effect(url, *args, **kwargs):
        if "github.com" in url:
            return Mock(status_code=200)
        return Mock(status_code=404)
    
    mock_get.side_effect = side_effect
    
    score = metric.evaluate(github_only_model)
    assert score == 1.0  # 1 out of 1 resource available


@patch("requests.get")
def test_dataset_only_available(mock_get, metric, dataset_only_model):
    """Test when only dataset is available."""
    def side_effect(url, *args, **kwargs):
        if "huggingface.co/api/datasets" in url:
            return Mock(status_code=200)
        return Mock(status_code=404)
    
    mock_get.side_effect = side_effect
    
    score = metric.evaluate(dataset_only_model)
    assert score == 1.0  # 1 out of 1 resource available


# Test error handling
@patch("requests.get")
def test_network_timeout(mock_get, metric, hf_model):
    """Test handling of network timeouts."""
    mock_get.side_effect = Exception("Network timeout")
    
    score = metric.evaluate(hf_model)
    assert score == 0.0


@patch("requests.get")
def test_malformed_urls(mock_get, metric):
    """Test handling of malformed URLs."""
    model = StubModelData(
        modelLink="https://huggingface.co/",  # Malformed
        codeLink="https://github.com/just-owner",  # Malformed
        datasetLink="invalid-url"
    )
    
    score = metric.evaluate(model)
    assert score == 0.0


def test_no_urls_provided(metric):
    """Test when no URLs are provided."""
    model = StubModelData(modelLink="", codeLink="", datasetLink="")
    
    score = metric.evaluate(model)
    assert score == 0.0


# Test URL classification
@patch("requests.get")
def test_huggingface_model_url_classification(mock_get, metric):
    """Test that HuggingFace model URLs are correctly identified."""
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink=None,
        datasetLink=None
    )
    
    mock_get.return_value = Mock(status_code=200)
    score = metric.evaluate(model)
    assert score == 1.0
    
    # Verify the correct API endpoint was called
    assert any("huggingface.co/api/models" in call[0][0] for call in mock_get.call_args_list)


@patch("requests.get")
def test_huggingface_dataset_url_classification(mock_get, metric):
    """Test that HuggingFace dataset URLs are correctly identified."""
    model = StubModelData(
        modelLink=None,
        codeLink=None,
        datasetLink="https://huggingface.co/datasets/squad"
    )
    
    mock_get.return_value = Mock(status_code=200)
    score = metric.evaluate(model)
    assert score == 1.0
    
    # Verify the correct API endpoint was called
    assert any("huggingface.co/api/datasets" in call[0][0] for call in mock_get.call_args_list)


@patch("requests.get")
def test_github_url_classification(mock_get, metric):
    """Test that GitHub URLs are correctly identified."""
    model = StubModelData(
        modelLink=None,
        codeLink="https://github.com/huggingface/transformers",
        datasetLink=None
    )
    
    mock_get.return_value = Mock(status_code=200)
    score = metric.evaluate(model)
    assert score == 1.0
    
    # Verify the correct API endpoint was called
    assert any("api.github.com/repos" in call[0][0] for call in mock_get.call_args_list)
