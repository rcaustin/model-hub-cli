import pytest

from src.util.URLBundler import URLBundle, bundle, classify_url


# -------------------------------------
# Tests for classify_url
# -------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://huggingface.co/some-model", "model"),
    ("https://huggingface.co/datasets/some-dataset", "dataset"),
    ("https://github.com/user/repo", "code"),
])
def test_classify_url_valid(url, expected):
    assert classify_url(url) == expected


@pytest.mark.parametrize("url", [
    "",                              # Empty string
    None,                            # None
    "   ",                           # Whitespace
    "ftp://somewhere.com/resource",  # Unsupported domain
    "https://huggingface.co",        # No path
    "https://unknown.com/item",      # Unknown domain
])
def test_classify_url_invalid(url):
    with pytest.raises(ValueError):
        classify_url(url)


# -------------------------------------
# Tests for URLBundle.clear
# -------------------------------------

def test_urlbundle_clear():
    b = URLBundle(model="model-url", code="code-url", dataset="data-url")
    b.clear()
    assert b.model is None
    assert b.code is None
    assert b.dataset is None


# -------------------------------------
# Tests for bundle
# -------------------------------------

def test_bundle_with_sample_urls(sample_urls):
    result = bundle(sample_urls)

    assert len(result) == 1
    bundle_result = result[0]

    assert bundle_result.model == \
        "https://huggingface.co/microsoft/DialoGPT-medium"
    assert bundle_result.dataset == \
        "https://huggingface.co/datasets/squad"
    assert bundle_result.code == \
        "https://github.com/huggingface/transformers"


# -----------------------------------
# Additional bundle tests (not using fixture)
# -----------------------------------

def test_bundle_multiple_models():
    urls = [
        "https://github.com/user/repo-1",
        "https://huggingface.co/model-1",
        "https://huggingface.co/datasets/data-2",
        "https://huggingface.co/model-2"
    ]
    result = bundle(urls)
    assert len(result) == 2
    assert result[0].model == "https://huggingface.co/model-1"
    assert result[0].code == "https://github.com/user/repo-1"
    assert result[0].dataset is None
    assert result[1].model == "https://huggingface.co/model-2"
    assert result[1].dataset == "https://huggingface.co/datasets/data-2"
    assert result[1].code is None


def test_bundle_code_and_dataset_before_model():
    urls = [
        "https://github.com/user/repo",
        "https://huggingface.co/datasets/some-dataset",
        "https://huggingface.co/some-model"
    ]
    result = bundle(urls)
    assert len(result) == 1
    assert result[0].model == "https://huggingface.co/some-model"
    assert result[0].code == "https://github.com/user/repo"
    assert result[0].dataset == "https://huggingface.co/datasets/some-dataset"


def test_bundle_code_without_model():
    urls = ["https://github.com/user/repo"]
    result = bundle(urls)
    assert result == []  # No model => no bundle


def test_classify_url_missing_netloc():
    # URL missing scheme, netloc will be empty
    with pytest.raises(ValueError, match="Malformed URL"):
        classify_url("huggingface.co/some-model")

    # Path-only string
    with pytest.raises(ValueError, match="Malformed URL"):
        classify_url("/some/path")
