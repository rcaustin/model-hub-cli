"""
Unit tests for the PerformanceClaimsMetric class.
"""

import pytest

from src.metrics.PerformanceClaimsMetric import PerformanceClaimsMetric
from tests.conftest import StubModelData


@pytest.fixture
def metric():
    return PerformanceClaimsMetric()


@pytest.fixture
def hf_model():
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink="https://github.com/huggingface/transformers",
        datasetLink="https://huggingface.co/datasets/squad",
    )
    # Mock the metadata properties
    model.hf_metadata = {
        "cardData": {
            "model_description": "This model achieves 95% accuracy on ImageNet",
            "model_summary": "High performance model with 0.92 F1 score",
        },
        "tags": ["accuracy", "imagenet"],
    }
    return model


@pytest.fixture
def github_only_model():
    model = StubModelData(
        modelLink="https://somewhere.else/model",
        codeLink="https://github.com/someuser/somerepo",
        datasetLink=None,
    )
    # Mock the metadata properties
    model.github_metadata = {"description": "Model with 94% accuracy on GLUE benchmark"}
    return model


# Test performance claims extraction from HuggingFace
def test_hf_claims_with_accuracy(metric, hf_model):
    """Test extraction of accuracy claims from HuggingFace metadata."""
    score = metric.evaluate(hf_model)
    assert score > 0.0  # Should find performance claims


def test_hf_claims_with_benchmark(metric):
    """Test extraction of benchmark-specific claims."""
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink=None,
        datasetLink=None,
    )
    model.hf_metadata = {
        "cardData": {
            "model_description": "Achieves 88.5 BLEU score on WMT dataset",
            "training_data": "Trained on Common Crawl and Wikipedia",
        }
    }

    score = metric.evaluate(model)
    assert score > 0.0


def test_hf_no_claims(metric):
    """Test when no performance claims are found."""
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink=None,
        datasetLink=None,
    )
    model.hf_metadata = {
        "cardData": {
            "model_description": "A general purpose language model",
            "model_summary": "Useful for various NLP tasks",
        },
        "tags": ["pytorch", "transformers"],
    }

    score = metric.evaluate(model)
    assert score == 0.0


# Test performance claims extraction from GitHub
def test_github_claims(metric, github_only_model):
    """Test extraction of claims from GitHub repository description."""
    score = metric.evaluate(github_only_model)
    assert score > 0.0


def test_github_no_claims(metric):
    """Test when GitHub repository has no performance claims."""
    model = StubModelData(
        modelLink="https://somewhere.else/model",
        codeLink="https://github.com/someuser/somerepo",
        datasetLink=None,
    )
    model.github_metadata = {
        "description": "A simple Python library for machine learning"
    }

    score = metric.evaluate(model)
    assert score == 0.0


# Test claim scoring
def test_claim_scoring_quantified_with_benchmark():
    """Test scoring of quantified claims with benchmark context."""
    metric = PerformanceClaimsMetric()

    claim = {
        "text": "95% accuracy on imagenet",
        "source": "test",
        "context": "performance metric",
    }

    score = metric._score_claim(claim)
    assert score == 1.0


def test_claim_scoring_quantified_without_benchmark():
    """Test scoring of quantified claims without benchmark context."""
    metric = PerformanceClaimsMetric()

    claim = {"text": "92% f1 score", "source": "test", "context": "performance metric"}

    score = metric._score_claim(claim)
    assert score == 0.8


def test_claim_scoring_vague_with_benchmark():
    """Test scoring of vague claims with benchmark context."""
    metric = PerformanceClaimsMetric()

    claim = {
        "text": "high accuracy on squad",
        "source": "test",
        "context": "performance metric",
    }

    score = metric._score_claim(claim)
    assert score == 0.6


def test_claim_scoring_benchmark_only():
    """Test scoring of benchmark mentions without specific metrics."""
    metric = PerformanceClaimsMetric()

    claim = {
        "text": "trained on imagenet",
        "source": "test",
        "context": "benchmark dataset",
    }

    score = metric._score_claim(claim)
    assert score == 0.2


# Test pattern matching
def test_performance_pattern_matching():
    """Test that performance patterns are correctly identified."""
    metric = PerformanceClaimsMetric()

    test_texts = [
        "95% accuracy",
        "F1: 0.92",
        "Precision: 88%",
        "BLEU score of 0.85",
        "ROUGE-L: 0.78",
        "Perplexity: 15.2",
        "Loss: 0.05",
    ]

    for text in test_texts:
        claims = metric._find_performance_claims(text)
        assert len(claims) > 0, f"Failed to find claims in: {text}"


def test_benchmark_pattern_matching():
    """Test that benchmark patterns are correctly identified."""
    metric = PerformanceClaimsMetric()

    test_texts = [
        "trained on ImageNet",
        "evaluated on GLUE",
        "SQuAD dataset",
        "WMT translation",
        "COCO object detection",
        "VQA visual question answering",
        "MS MARCO passage ranking",
    ]

    for text in test_texts:
        claims = metric._find_performance_claims(text)
        assert len(claims) > 0, f"Failed to find benchmark claims in: {text}"


# Test error handling
def test_network_error_handling(metric):
    """Test handling of network errors."""
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink=None,
        datasetLink=None,
    )
    model.hf_metadata = None  # Simulate network error

    score = metric.evaluate(model)
    assert score == 0.0


def test_invalid_json_response(metric):
    """Test handling of invalid JSON responses."""
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink=None,
        datasetLink=None,
    )
    model.hf_metadata = {"invalid": "data"}  # Invalid structure

    score = metric.evaluate(model)
    assert score == 0.0


def test_no_urls_provided(metric):
    """Test when no URLs are provided."""
    model = StubModelData(modelLink="", codeLink="", datasetLink="")

    score = metric.evaluate(model)
    assert score == 0.0


# Test edge cases
def test_empty_text_fields():
    """Test handling of empty text fields."""
    metric = PerformanceClaimsMetric()

    claims = metric._find_performance_claims("")
    assert len(claims) == 0

    claims = metric._find_performance_claims(None)
    assert len(claims) == 0


def test_malformed_metadata(metric):
    """Test handling of malformed metadata."""
    model = StubModelData(
        modelLink="https://huggingface.co/test/model", codeLink=None, datasetLink=None
    )
    model.hf_metadata = {"invalid": "data"}

    score = metric.evaluate(model)
    assert score == 0.0


# Test multiple claims scoring
def test_multiple_claims_averaging(metric):
    """Test that multiple claims are properly averaged."""
    model = StubModelData(
        modelLink="https://huggingface.co/facebook/bart-large",
        codeLink=None,
        datasetLink=None,
    )
    model.hf_metadata = {
        "cardData": {
            "model_description": "95% accuracy on ImageNet, 92% F1 on GLUE",
            "model_summary": "High performance model",
        }
    }

    score = metric.evaluate(model)
    assert 0.0 < score < 1.0  # Should be between 0 and 1
