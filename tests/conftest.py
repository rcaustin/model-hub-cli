"""
Pytest configuration and fixtures.
"""
import pytest


@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return [
        "https://huggingface.co/datasets/squad",
        "https://github.com/huggingface/transformers",
        "https://huggingface.co/microsoft/DialoGPT-medium"
    ]
