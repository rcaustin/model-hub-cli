#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
from pathlib import Path


def print_usage() -> None:
    print("""Usage: python main.py [OPTIONS] ARG

Arguments:
  ARG  [required]  One of:
                   "install"              Run install logic
                   "test"                 Run test logic
                   <absolute file path>   Process the given file""")


def run_install() -> int:
    """
    Run installation script to setup dependencies in virtual environment

    Uses platform-specific commands:
        - Windows: runs 'setup.bat' via cmd.exe
        - Unix-like Systems: runs 'setup.sh' via bash

    Returns:
        int: the return code from the subprocess command
    """
    cmd = (
        ["cmd.exe", "/c", "setup.bat"]
        if platform.system() == "Windows"
        else ["bash", "setup.sh"]
    )
    result = subprocess.run(cmd)
    return result.returncode


def run_test() -> int:
    """
    Run the test suite using pytest invoked from the python interpreter in
    the virtual environment.

    Returns:
        int: exit code incicating success (0) or failure (non-zero)
    """
    try:
        venv_python = os.path.join(".venv", "bin", "python3")
        if platform.system() == "Windows":
            venv_python = os.path.join(".venv.", "Scripts", "python.exe")
        cmd = [venv_python, "-m", "pytest", "-v", "tests/"]
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest is not installed.")
        print("Install dependencies first by running: run.py install")
        return 1


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
        try:
            from src.commands.catalogue_runner import run_catalogue
        except ModuleNotFoundError:
            print("Error: Required dependencies are not installed.")
            print("Install dependencies first by running: run.py install")
            return 1
        return run_catalogue(arg)
    print_usage()
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    main(sys.argv[1])
