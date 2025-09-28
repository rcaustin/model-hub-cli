# `src/util/` — URL & Metadata Utilities

Helper utilities for:
- **Classifying input URLs** into `{model, code, dataset}` buckets.
- **Fetching metadata** from external sources (e.g., GitHub, Hugging Face, dataset endpoints).
- **Resilience** against network hiccups and missing/partial data.

## Files

- **`url_utils.py`** — functions to normalize and classify raw URLs. Expected behaviors:
  - Identify **Hugging Face model** URLs (e.g., `huggingface.co/<org>/<repo>`).
  - Identify **code repository** URLs (e.g., `github.com/<owner>/<repo>`).
  - Treat anything else as a **dataset/other** URL unless proven otherwise.
  - Provide helpers to group up to three URLs per input line into a `{model, code, dataset}` tuple/dict.
  - Be robust to ordering and duplicates; prefer the first confident match per category.

- **`metadata_fetchers.py`** — wrappers around external APIs and HTML/JSON endpoints to obtain:
  - Repo metadata (topics, license, stars/forks, last commit, contributors)
  - Model card text / model tags (when available)
  - Dataset landing info (basic reachability, presence of description/files)
  - Light caching/retry behaviors to avoid rate limit issues
  - Optional use of `GITHUB_TOKEN` for higher rate limits

> **Note**: Implementations should degrade gracefully (return partial results and log warnings) instead of throwing hard errors on network failure.

## Environment Variables

- `GITHUB_TOKEN` *(optional but recommended)* — used by GitHub calls to increase rate limits and access private info if appropriate.

```bash
export GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Typical Flow

1. **Parse raw line** from the input file into a list of 1–3 URLs.
2. **Classify URLs** with `url_utils.py` into `{model, code, dataset}`.
3. **Fetch metadata** using `metadata_fetchers.py` for present categories.
4. **Return structured context** for `Model`/metrics to consume.

```
raw URLs → url_utils.classify() → {model, code, dataset}
        → metadata_fetchers.fetch_*() → context with enriched fields
```

## Error Handling & Resilience

- Use timeouts and retries for external calls.
- If a service is unreachable, **return partial metadata** and log a warning.
- Never crash metric computation because of a missing field—metrics should interpret missing data conservatively.

## Testing Notes

- Unit tests for these utilities live in `tests/` (e.g., `test_url_utils.py`, `test_metadata_fetchers.py`).
- Include fixtures to simulate network responses and rate-limit conditions.

