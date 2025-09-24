from collections import namedtuple
from urllib.parse import urlparse

URLSet = namedtuple('URLSet', ['model', 'code', 'dataset'])


def classify_url(url: str) -> str:
    """
    Classify a URL as one of [model, dataset, code] according to its location.

    Args:
        url (str): the URL to be classified

    Returns:
        str: one of ["model", "dataset", "code"]

    Raises:
        ValueError: if the URL is unrecognized, invalid, or malformed.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Input mus be a non-empty URL string.")

    parsed = urlparse(url.strip())
    netloc = parsed.netloc
    path = parsed.path

    if not netloc:
        raise ValueError(f"Malformed URL: '{url}'")

    if netloc == "huggingface.co":
        if path.startswith("/datasets/"):
            return "dataset"
        elif path.startswith("/"):
            return "model"
        else:
            raise ValueError(f"Unknown Hugging Face URL: '{url}'")
    elif netloc == "github.com":
        return "code"
    else:
        raise ValueError(f"Unknown or unsupported URL domain: '{netloc}'")


def classify_urls(urls: list[str]) -> URLSet:
    """
    Classify up to 3 URLs and return a namedtuple grouping the
    model, code, and dataset URLs for convenient extraction.

    Args:
        urls (list[str]): list of URLs (max length 3)

    Returns:
        URLSet: namedtuple(model, code, dataset) with URLs

    Raises:
        ValueError: if invallid URLs, duplicates, or missing model URL
    """
    if not urls:
        raise ValueError("At least one URL is required.")
    if len(urls) > 3:
        raise ValueError("No more than 3 URLs allowed.")

    model = code = dataset = None

    for url in urls:
        kind = classify_url(url)

        if kind == "model":
            if model is not None:
                raise ValueError(f"Duplicate model URL found in group: {url}")
            model = url
        elif kind == "code":
            if code is not None:
                raise ValueError(f"Duplicate code URL found in group: {url}")
            code = url
        elif kind == "dataset":
            if dataset is not None:
                raise ValueError(f"Duplicate dataset URL found in group: {url}")
            dataset = url

    if model is None:
        raise ValueError("At least one model URL is required.")

    return URLSet(model=model, code=code, dataset=dataset)
