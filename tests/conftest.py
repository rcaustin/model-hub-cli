"""
Pytest configuration and reusable fixtures for model-hub-cli tests.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class StubModelData:
    """Stub implementation of the ModelData protocol for testing."""
    modelLink: str
    codeLink: Optional[str]
    datasetLink: Optional[str]


@pytest.fixture
def base_model() -> StubModelData:
    """
    Fixture that returns a fully populated StubModelData instance.
    Used as a baseline test model with valid HuggingFace and GitHub URLs.
    """
    return StubModelData(
        modelLink="https://huggingface.co/microsoft/DialoGPT-medium",
        codeLink="https://github.com/huggingface/transformers",
        datasetLink="https://huggingface.co/datasets/squad"
    )


@pytest.fixture
def sample_urls() -> list[str]:
    """
    Fixture that provides a sample list of model, dataset, and code URLs.
    Used to simulate bundled input.
    """
    return [
        "https://huggingface.co/datasets/squad",
        "https://github.com/huggingface/transformers",
        "https://huggingface.co/microsoft/DialoGPT-medium"
    ]
