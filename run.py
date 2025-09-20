#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

# Constants
SETUP_SCRIPT_WINDOWS = ["cmd.exe", "/c", "setup.bat"]
SETUP_SCRIPT_UNIX = ["bash", "setup.sh"]
VENV_PYTHON_WIN = os.path.join(".venv", "Scripts", "python.exe")
VENV_PYTHON_UNIX = os.path.join(".venv", "bin", "python3")


def print_usage() -> None:
    print("""Usage: python main.py [OPTIONS] ARG

Arguments:
  ARG  [required]  One of:
                   "install"              Run install logic
                   "test"                 Run test logic
                   <absolute file path>   Process the given file""")


def get_venv_python() -> str:
    """Return the path to the Python interpreter in the virtual environment."""
    return VENV_PYTHON_WIN if platform.system() == "Windows" else VENV_PYTHON_UNIX


def run_install() -> int:
    """Run the platform-specific install script."""
    cmd = SETUP_SCRIPT_WINDOWS if platform.system() == "Windows" else SETUP_SCRIPT_UNIX
    return subprocess.run(cmd).returncode


def run_test() -> int:
    """Run tests using pytest from the virtual environment."""
    try:
        # Run the unit tests
        test_cmd = [
            get_venv_python(), "-m", "coverage", "run", "-m", "pytest", "-v", "tests/"
        ]
        subprocess.run(test_cmd, check=False).returncode

        # Generate the coverage report
        report_cmd = [get_venv_python(), "-m", "coverage", "report"]
        subprocess.run(report_cmd, check=False)
        return 0
    except FileNotFoundError:
        print("Error: pytest is not installed.")
        print("Install dependencies first by running: python main.py install")
        return 1


def run_program(file_path: str) -> int:
    """Run the main application logic with the given file path."""
    cmd = [get_venv_python(), "-m", "src.commands.catalogue_runner", file_path]
    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError:
        print("Error: Required dependencies are not installed.")
        print("Install dependencies first by running: python main.py install")
        return 1


def main(arg: str) -> int:
    """Main entry point for the application."""
    actions: dict[str, Callable[[], int]] = {
        "install": run_install,
        "test": run_test,
    }

    if arg in actions:
        return actions[arg]()
    elif Path(arg).is_absolute():
        return run_program(arg)
    else:
        print_usage()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    sys.exit(main(sys.argv[1]))
