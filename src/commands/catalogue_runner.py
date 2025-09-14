import sys

from loguru import logger

from src.Model import Model
from src.ModelCatalogue import ModelCatalogue
from src.util.URLBundler import bundle

# Configure loguru
logger.remove()  # Remove default logger
logger.add(sys.stderr, level="INFO")  # Console
logger.add("logs/run.log", rotation="1 MB", level="DEBUG")  # Log file


def read_urls_from_file(file_path: str) -> list[str]:
    """
    Reads URLs from an ASCII-encoded, newline delimited file.

    Args:
        file_path (str): the absolute path to the URL file

    Returns:
        list[str]: a list of URL strings read from the file

    Raises:
        FileNotFoundError: if the file does not exist
        IOError: if there is an error reading the file
        UnicodeDecodeError: if the file is not ASCII encoded
    """
    urls = []
    with open(file_path, "r", encoding="ascii") as file:
        for line in file:
            url = line.strip()
            if url:  # skip empty lines
                urls.append(url)
    logger.debug(f"Read {len(urls)} URLs from file: {file_path}")
    return urls


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

    try:
        urls = read_urls_from_file(file_path)
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        return 1

    try:
        url_bundles = bundle(urls)
    except ValueError as e:
        logger.error(f"Error while bundling URLs: {e}")
        return 1

    catalogue = ModelCatalogue()

    for url_bundle in url_bundles:
        catalogue.addModel(Model(url_bundle))

    print(catalogue.generateReport())
    return 0
