#!/usr/bin/env python3
import platform
import subprocess
import sys
from pathlib import Path

from src.Model import Model
from src.ModelCatalogue import ModelCatalogue
from src.util.URLBundler import bundle


def print_usage() -> None:
    print("""Usage: python main.py [OPTIONS] ARG

Arguments:
  ARG  [required]  One of:
                   "install"              Run install logic
                   "test"                 Run test logic
                   <absolute file path>   Process the given file""")


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
    return urls


def run_install() -> int:
    """
    Run installation script to setup dependencies in virtual environment

    Uses platform-specific commands:
        - Windows: runs 'setup.bat' via cmd.exe
        - Unix-like Systems: runs 'setup.sh' via bash

    Returns:
        int: the return code from the subprocess command
    """
    print("Installing dependencies to virtual environment...")
    cmd = (
        ["cmd.exe", "/c", "setup.bat"]
        if platform.system() == "Windows"
        else ["bash", "setup.sh"]
    )
    result = subprocess.run(cmd)
    return result.returncode


def run_test() -> int:
    """
    Run the test suite.

    Returns:
        int: exit code incicating success (0) or failure (non-zero)
    """
    print("Running Pytest suite...")
    # Call pytest or your test runner here
    return 0


def run_catalogue(file_path: str) -> int:
    """
    Process an ASCII-encoded, newline-delimited file containing URLs and
    build a model catalogue. Finish by printing the generated catalogue report.

    Args:
        file_path (str): Absolute path to the file containing URLs.

    Returns:
        int: 0 if all URLs are processed successfully, 1 if any error occurs.
    """
    print(f"Processing file: {file_path}...")
    try:
        urls = read_urls_from_file(file_path)
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        print(f"Error reading file '{file_path}': {e}")
        return 1

    try:
        url_bundles = bundle(urls)
    except ValueError as e:
        print(f"Error while bundling URLs: {e}")
        return 1

    catalogue = ModelCatalogue()

    for b in url_bundles:
        catalogue.addModel(Model(b.model, b.code, b.dataset))

    print(catalogue.generateReport())
    return 0


def main(arg: str) -> int:
    """
    Entry point for the application.

    Determines the action to take based on the provided argument:
      - "install": run installation
      - "test": run test suite
      - absolute file path: process the file containing URLs
      - otherwise, prints usage information and returns error

    Args:
        arg (str): Command-line argument specifying the operation mode or file.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    if arg == "install":
        return run_install()
    if arg == "test":
        return run_test()
    if Path(arg).is_absolute():
        return run_catalogue(arg)
    print_usage()
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    main(sys.argv[1])
