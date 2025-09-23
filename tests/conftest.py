"""
Pytest configuration and reusable fixtures for model-hub-cli tests.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pytest


@dataclass
class StubModelData:
    """Stub implementation of the ModelData protocol for testing."""
    modelLink: str
    codeLink: Optional[str]
    datasetLink: Optional[str]
    
    # Add private attributes for metadata
    _hf_metadata: Optional[Dict[str, Any]] = None
    _github_metadata: Optional[Dict[str, Any]] = None
    
    # Add properties that the metrics expect
    @property
    def hf_metadata(self) -> Optional[Dict[str, Any]]:
        """Mock HuggingFace metadata for testing."""
        return self._hf_metadata
    
    @hf_metadata.setter
    def hf_metadata(self, value: Optional[Dict[str, Any]]):
        """Setter for HuggingFace metadata."""
        self._hf_metadata = value
    
    @property
    def github_metadata(self) -> Optional[Dict[str, Any]]:
        """Mock GitHub metadata for testing."""
        return self._github_metadata
    
    @github_metadata.setter
    def github_metadata(self, value: Optional[Dict[str, Any]]):
        """Setter for GitHub metadata."""
        self._github_metadata = value


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
