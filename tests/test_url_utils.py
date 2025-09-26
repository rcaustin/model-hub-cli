import pytest
from src.util import url_utils
from src.util.url_utils import URLSet


def test_classify_url_valid():
    assert url_utils.classify_url(
        "https://huggingface.co/datasets/squad"
    ) == "dataset"
    assert url_utils.classify_url(
        "https://huggingface.co/microsoft/DialoGPT-medium"
    ) == "model"
    assert url_utils.classify_url(
        "https://github.com/huggingface/transformers"
    ) == "code"


def test_classify_url_invalid():
    with pytest.raises(ValueError):
        url_utils.classify_url("")
    with pytest.raises(ValueError):
        url_utils.classify_url("not a url")
    with pytest.raises(ValueError):
        url_utils.classify_url("https://example.com/unknown")


def test_classify_urls_all_three(sample_urls):
    # sample_urls fixture has dataset, code, model in that order
    result = url_utils.classify_urls(sample_urls)
    assert isinstance(result, URLSet)
    assert result.model == "https://huggingface.co/microsoft/DialoGPT-medium"
    assert result.code == "https://github.com/huggingface/transformers"
    assert result.dataset == "https://huggingface.co/datasets/squad"


def test_classify_urls_various_orders():
    urls1 = [
        "https://huggingface.co/microsoft/DialoGPT-medium",
        "https://github.com/huggingface/transformers",
        "https://huggingface.co/datasets/squad",
    ]
    urls2 = [
        "https://github.com/huggingface/transformers",
        "https://huggingface.co/microsoft/DialoGPT-medium",
    ]
    urls3 = [
        "https://huggingface.co/microsoft/DialoGPT-medium",
    ]

    result1 = url_utils.classify_urls(urls1)
    assert result1.model == "https://huggingface.co/microsoft/DialoGPT-medium"
    assert result1.code == "https://github.com/huggingface/transformers"
    assert result1.dataset == "https://huggingface.co/datasets/squad"

    result2 = url_utils.classify_urls(urls2)
    assert result2.model == "https://huggingface.co/microsoft/DialoGPT-medium"
    assert result2.code == "https://github.com/huggingface/transformers"
    assert result2.dataset is None

    result3 = url_utils.classify_urls(urls3)
    assert result3.model == "https://huggingface.co/microsoft/DialoGPT-medium"
    assert result3.code is None
    assert result3.dataset is None


def test_classify_urls_duplicate_urls():
    urls = [
        "https://huggingface.co/microsoft/DialoGPT-medium",
        "https://huggingface.co/microsoft/DialoGPT-medium",  # duplicate model
    ]
    with pytest.raises(ValueError, match="Duplicate model URL found"):
        url_utils.classify_urls(urls)

    urls = [
        "https://github.com/huggingface/transformers",
        "https://github.com/huggingface/transformers",  # duplicate code
        "https://huggingface.co/microsoft/DialoGPT-medium",
    ]
    with pytest.raises(ValueError, match="Duplicate code URL found"):
        url_utils.classify_urls(urls)


def test_classify_urls_missing_model():
    urls = [
        "https://huggingface.co/datasets/squad",
        "https://github.com/huggingface/transformers",
    ]
    with pytest.raises(ValueError, match="At least one model URL is required"):
        url_utils.classify_urls(urls)


def test_classify_urls_too_many_urls():
    urls = [
        "https://huggingface.co/microsoft/DialoGPT-medium",
        "https://github.com/huggingface/transformers",
        "https://huggingface.co/datasets/squad",
        "https://example.com/extra"
    ]
    with pytest.raises(ValueError, match="No more than 3 URLs allowed"):
        url_utils.classify_urls(urls)



def test_parse_positional_and_missing():
    from src.util.url_utils import parse_urls_by_position
    u = parse_urls_by_position(["https://github.com/o/r", "", "https://huggingface.co/o/m"])
    assert u.code.endswith("/o/r")
    assert u.dataset is None
    assert u.model.endswith("/o/m")

def test_requires_model_third_slot():
    from src.util.url_utils import parse_urls_by_position
    try:
        parse_urls_by_position(["https://github.com/o/r", ""])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Model URL" in str(e)
