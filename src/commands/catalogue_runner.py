import os
import sys

from loguru import logger

from src.Model import Model
from src.ModelCatalogue import ModelCatalogue


def run_catalogue(file_path: str) -> int:
    """
    Process an ASCII-encoded, newline-delimited file containing URLs and
    build a model catalogue. Finish by printing the generated catalogue report.

    Args:
        file_path (str): Absolute path to the file containing URLs.

    Returns:
        int: 0 if all URLs are processed successfully, 1 if any error occurs.
    """
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
                    success = False
                    continue

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
        # Path is not a directory
        if os.path.isdir(log_file):
            print(f"LOG_FILE points to a directory, not a file: {log_file}")
            exit(1)

        # Parent directory exists
        log_dir = os.path.dirname(log_file) or "."
        if not os.path.exists(log_dir):
            print(f"Directory does not exist for LOG_FILE: {log_dir}")
            exit(1)

        # Parent directory is writable
        if not os.access(log_dir, os.W_OK):
            print(f"Cannot write to directory for LOG_FILE: {log_dir}")
            exit(1)

        # File is writable (if it already exists)
        if os.path.exists(log_file) and not os.access(log_file, os.W_OK):
            print(f"LOG_FILE exists but is not writable: {log_file}")
            exit(1)

        # Try opening the file for writing (sometimes won't raise until write a log)
        try:
            logger.add(log_file, rotation="1 MB", level=log_level)
        except Exception as e:
            print(f"Failed to configure log file '{log_file}': {e}")
            exit(1)
    else:
        logger.add(sys.stderr, leve=log_level)


if __name__ == "__main__":
    configure_logging()
    if len(sys.argv) < 2:
        print("Usage: run <absolute_path_to_input_file>")
        sys.exit(1)
    sys.exit(run_catalogue(sys.argv[1]))
