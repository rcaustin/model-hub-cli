"""
Pytest configuration and fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return [
        "https://huggingface.co/microsoft/DialoGPT-medium",
        "https://huggingface.co/datasets/squad",
        "https://github.com/huggingface/transformers"
    ]
