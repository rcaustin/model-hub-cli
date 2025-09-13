from dataclasses import dataclass
from typing import Optional, List
from urllib.parse import urlparse


@dataclass
class URLBundle:
    model: Optional[str] = None
    code: Optional[str] = None
    dataset: Optional[str] = None

    def clear(self):
        self.model = None
        self.code = None
        self.dataset = None


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


def bundle(urls: List[str]) -> List[URLBundle]:
    """
    Group URLs into URLBundles, each containing a model and its associated
    code/dataset if present.

    Args:
        urls (List[str]): list of URLs to be bundled

    Returns:
        List[URLBundle]: a list of complete URLBundles
    """
    bundles: List[URLBundle] = []
    current = URLBundle()

    for url in urls:
        category = classify_url(url)

        if category == "model":
            current.model = url
            bundles.append(URLBundle(
                model=current.model,
                code=current.code,
                dataset=current.dataset
            ))
            current.clear()
        else:
            setattr(current, category, url)

    return bundles
