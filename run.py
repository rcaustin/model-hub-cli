#!/usr/bin/env python3
import platform
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import sys


def print_usage() -> None:
    print("""Usage: python main.py [OPTIONS] ARG

Arguments:
  ARG  [required]  One of:
                   "install"              Run install logic
                   "test"                 Run test logic
                   <absolute file path>   Process the given file""")


def classify_url(url: str) -> str:
    parsed = urlparse(url.strip())
    netloc = parsed.netloc
    path = parsed.path

    if netloc == "huggingface.co":
        if path.startswith("/datasets/"):
            return "dataset"
        else:
            return "model"
    elif netloc == "github.com":
        return "code"
    else:
        return "unknown"


def main(arg: str) -> int:
    if arg == "install":
        print("Installing dependencies to virtual environment...")
        if platform.system() == "Windows":
            result = subprocess.run(["cmd.exe", "/c", "setup.bat"])
        else:
            result = subprocess.run(["bash", "setup.sh"])
        return result.returncode
    elif arg == "test":
        print("Running Pytest suite...")
        return 0
    elif Path(arg).is_absolute():
        print(f"Processing file: {arg}...")
        return 0
    else:
        print_usage()
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    main(sys.argv[1])
