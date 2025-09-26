"""
main.py
========
CLI entry point for Model Hub CLI.

This module parses command-line arguments, reads an input file where each
line contains 1–3 URLs (order-agnostic) corresponding to a model, code
repository, and/or dataset, and delegates batch processing to the
``ModelCatalogue``. Results are emitted to STDOUT as one NDJSON object
per input line, including per-metric scores, a composite NetScore, and
latency fields.

Responsibilities
---------------
- Validate CLI arguments and input file readability.
- Read and iterate lines of raw URLs (whitespace-delimited).
- Delegate grouping, metadata fetching, metric execution, and aggregation
  to core modules (``ModelCatalogue``, ``Model``, ``metrics/*``,
  ``util/url_utils.py``, ``util/metadata_fetchers.py``).
- Stream NDJSON output to STDOUT (suitable for pipelines/CI).

Expected Input Format
---------------------
Each non-empty line contains 1–3 URLs separated by spaces. The order does
not matter; URLs are classified into {model, code, dataset} categories.

Example:
    https://huggingface.co/distilbert-base-uncased https://github.com/huggingface/transformers https://example.com/dataset
    https://huggingface.co/openai/whisper-base https://github.com/openai/whisper

Environment
-----------
- GITHUB_TOKEN (optional): Improves GitHub API rate limits and metadata completeness.

Exit Codes
----------
- 0: Success.
- Non-zero: CLI argument/IO errors or unrecoverable failures; details are logged.

Usage
-----
Run with the helper script:
    $ ./run /absolute/path/to/inputs.txt

Or directly with Python:
    $ python -m src.main /absolute/path/to/inputs.txt

Key Functions / Flow
--------------------
- parse_args(argv) -> argparse.Namespace
- main(argv=None) -> int
  - Validates input path
  - Streams lines to ModelCatalogue for evaluation
  - Writes NDJSON results to STDOUT
  - Returns an appropriate process exit code

Notes
-----
- This module should avoid performing network calls directly; those live in util/metadata_fetchers.py.
- Keep side effects minimal and testable. Use small helpers and pure functions where possible.
"""

import os
import sys
import requests

from loguru import logger

from src.Model import Model
from src.ModelCatalogue import ModelCatalogue


def validate_github_token() -> bool:
    """
    Validate the GITHUB_TOKEN environment variable.
    
    Returns:
        bool: True if token is valid, False otherwise
    """
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token:
        logger.error("GITHUB_TOKEN environment variable is not set")
        return False
    
    try:
        # Test token with a simple API call
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {github_token}"
        }
        
        # Use a simple API call to validate the token
        response = requests.get("https://api.github.com/user", headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info("GitHub token is valid")
            return True
        else:
            logger.error(f"GitHub token validation failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error validating GitHub token: {e}")
        return False


def run_catalogue(file_path: str) -> int:
    """
    Process an ASCII-encoded, newline-delimited file containing URLs and
    build a model catalogue. Finish by printing the generated catalogue report.

    Args:
        file_path (str): Absolute path to the file containing URLs.

    Returns:
        int: 0 if all URLs are processed successfully, 1 if any error occurs.
    """
    # Validate GitHub token before proceeding
    if not validate_github_token():
        logger.error("Invalid or missing GITHUB_TOKEN. Exiting.")
        return 1
    
    logger.info(f"Running model catalogue on file: {file_path}")
    catalogue = ModelCatalogue()
    success = True

    # Extract URLs line-by-line, create models, and add them to the catalogue
    try:
        with open(file_path, 'r', encoding='ascii') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    logger.warning(f"Skipping empty line {line_num}")
                    continue

                parts = [part.strip() for part in line.split(',')]
                if len(parts) != 3:
                    logger.error(
                        "Line {} must have exactly 3 comma-separated fields: {}",
                        line_num,
                        line
                    )
                    exit(1)

                # Filter out empty strings
                urls = [url for url in parts if url]

                try:
                    catalogue.addModel(Model(urls))
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    success = False

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error reading file: {e}")
        return 1

    # Evaluate the models according to our metrics
    catalogue.evaluateModels()

    # Print the results
    print(catalogue.generateReport())
    return 0 if success else 1


def configure_logging():
    logger.remove()
    log_level_env = os.getenv("LOG_LEVEL", "0").strip()
    log_file = os.getenv("LOG_FILE", "").strip()

    if log_level_env == "2":
        log_level = "DEBUG"
    elif log_level_env == "1":
        log_level = "INFO"
    else:
        return  # Silent -- No Logging

    if log_file:
        try:
            if not os.path.exists(log_file):
                print(f"Log file does not exist: '{log_file}'")
                sys.exit(1)
            logger.add(log_file, rotation="1 MB", level=log_level)
        except Exception as e:
            print(f"Failed to configure log file '{log_file}': {e}")
            exit(1)
    else:
        logger.add(sys.stderr, level=log_level)


if __name__ == "__main__":
    configure_logging()
    if len(sys.argv) < 2:
        print("Usage: run <absolute_path_to_input_file>")
        sys.exit(1)
    sys.exit(run_catalogue(sys.argv[1]))
