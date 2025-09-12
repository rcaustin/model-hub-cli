#!/usr/bin/env python3
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def print_usage() -> None:
    print("""Usage: python main.py [OPTIONS] ARG

Arguments:
  ARG  [required]  One of:
                   "install"              Run install logic
                   "test"                 Run test logic
                   <absolute file path>   Process the given file""")


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
            if url: # skip empty lines
                urls.append(url)
    return urls


def main(arg: str) -> int:
    """
    Application Entry-Point

    Args:
        arg (str): one of "install", "test", or the absolute path of an ASCII
                   encoded, newline delimited file containing URLs
        
    Returns:
        int: 0 on successful termination, 1 on errored termination
    """
    if arg == "install":
        print("Installing dependencies to virtual environment...")
        cmd = ["cmd.exe", "/c", "setup.bat"] if platform.system() == "Windows" else ["bash", "setup.sh"]
        result = subprocess.run(cmd)
        return result.returncode
    
    if arg == "test":
        print("Running Pytest suite...")
        # Call the test runner here
        return 0

    # ./run.py $URL_FILE
    if Path(arg).is_absolute():
        print(f"Processing file: {arg}...")
        try:
            urls = read_urls_from_file(arg)
        except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
            print(f"Error reading file '{arg}': {e}")
            return 1

        for url in urls:
            try:
                category = classify_url(url)
                print(f"URL: {url} -> Category: {category}")
            except ValueError as e:
                print(f"URL '{url}' -> Error: {e}")
                return 1
        return 0

    # ./run.py $NOT_ABSOLUTE_PATH
    else:
        print_usage()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    main(sys.argv[1])
