# src/util/url_utils.py
from collections import namedtuple
from urllib.parse import urlparse
from typing import Optional, List

# Keep legacy field order used by the rest of the code:
# URLSet(model, code, dataset)
URLSet = namedtuple("URLSet", ["model", "code", "dataset"])

_MISSING_TOKENS = {"", "none", "null", "na", "n/a"}

_EXPECTED_BY_SLOT = {
    0: "code",     # 1st item
    1: "dataset",  # 2nd item
    2: "model",    # 3rd item
}


def _normalize_missing_token(token: Optional[str]) -> Optional[str]:
    """
    Trim and convert common 'missing' markers to None.
    """
    if token is None:
        return None
    t = token.strip()
    return None if t.lower() in _MISSING_TOKENS else (t if t else None)


def classify_url(url: str) -> str:
    """
    Domain-based probe for a single URL.
    Returns: "model" | "dataset" | "code"
    Raises: ValueError if URL is malformed or unsupported.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Input must be a non-empty URL string.")

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


def classify_urls(urls: List[str]) -> URLSet:
    """
    Strict position-validated classification of up to 3 URLs in a line.
    The input order is always: [code, dataset, model].

    Behavior:
      - Iterate exactly three times (slots 0..2).
      - For each non-empty URL, call classify_url(url) to detect its type.
      - Compare detected type with expected type for that slot:
          slot 0 -> 'code', slot 1 -> 'dataset', slot 2 -> 'model'
        If they don't match, raise a clear ValueError.
      - Missing values ('', 'none', 'null', 'na', 'n/a') are allowed for slots
        0 and 1, but slot 2 (model) is REQUIRED.

    Returns:
      URLSet(model, code, dataset)  # legacy field order used by callers
    """
    if not urls:
        raise ValueError("Expected 1–3 comma-separated items in order: code,dataset,model.")
    if len(urls) > 3:
        raise ValueError("At most 3 items allowed (code,dataset,model).")

    padded = (urls + ["", "", ""])[:3]
    code_url    = _normalize_missing_token(padded[0])
    dataset_url = _normalize_missing_token(padded[1])
    model_url   = _normalize_missing_token(padded[2])

    # Enforce required model (3rd slot)
    if model_url is None:
        raise ValueError("Model URL (3rd position) is required.")

    # Validate each non-empty URL by domain/type vs expected slot
    triplet = [code_url, dataset_url, model_url]
    for idx, url in enumerate(triplet):
        expected = _EXPECTED_BY_SLOT[idx]  # 'code' | 'dataset' | 'model'
        if url is None:
            # Missing is allowed for slots 0..1 only
            if idx < 2:
                continue
            # idx == 2 handled above (required)
        try:
            detected = classify_url(url) if url is not None else None
        except ValueError as e:
            # If it's a known-bad URL string (malformed/unsupported), surface it.
            # This keeps validation strict as requested.
            raise ValueError(
                f"Invalid {expected} URL in position {idx+1}: {url!r} ({e})"
            ) from e

        if detected is not None and detected != expected:
            pos_name = ["code (1st)", "dataset (2nd)", "model (3rd)"][idx]
            raise ValueError(
                f"URL in {pos_name} slot appears to be '{detected}': {url}"
            )

    # Return in legacy field order
    return URLSet(model=model_url, code=code_url, dataset=dataset_url)


# Optional: keep a pure positional parser (if other parts still import it)
def parse_urls_by_position(urls: List[str]) -> URLSet:
    """
    Pure positional mapping without domain validation. Kept for compatibility.
    Input order: [code, dataset, model].
    """
    if not urls:
        raise ValueError("Expected 1–3 comma-separated items in order: code,dataset,model.")
    if len(urls) > 3:
        raise ValueError("At most 3 items allowed (code,dataset,model).")

    padded = (urls + ["", "", ""])[:3]
    code_url    = _normalize_missing_token(padded[0])
    dataset_url = _normalize_missing_token(padded[1])
    model_url   = _normalize_missing_token(padded[2])

    if model_url is None:
        raise ValueError("Model URL (3rd position) is required.")

    return URLSet(model=model_url, code=code_url, dataset=dataset_url)
