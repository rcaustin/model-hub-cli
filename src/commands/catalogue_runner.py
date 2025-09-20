import sys

from loguru import logger

from src.Model import Model
from src.ModelCatalogue import ModelCatalogue

# Configure loguru
logger.remove()  # Remove default logger
logger.add(sys.stderr, level="INFO")  # Console
logger.add("logs/run.log", rotation="1 MB", level="DEBUG")  # Log file


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
                    logger.warning(f"Skipping empty line{line_num}")
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


if __name__ == "__main__":
    run_catalogue(sys.argv[1])
