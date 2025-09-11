#!/usr/bin/env python3

import platform
import subprocess
from pathlib import Path

import typer

app = typer.Typer()


def print_usage() -> None:
    typer.echo("""Usage: python main.py [OPTIONS] ARG

Arguments:
  ARG  [required]  One of:
                   "install"              Run install logic
                   "test"                 Run test logic
                   <absolute file path>   Process the given file""")


def main(arg: str) -> int:
    if arg == "install":
        typer.echo("Installing dependencies to virtual environment...")
        if platform.system() == "Windows":
            result = subprocess.run(["cmd.exe", "/c", "setup.bat"])
        else:
            result = subprocess.run(["bash", "setup.sh"])
        raise typer.Exit(code=result)
    elif arg == "test":
        typer.echo("Running Pytest suite...")
    elif Path(arg).is_absolute():
        typer.echo(f"Processing file: {arg}...")
        raise typer.Exit()
    else:
        print_usage()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app.command()(main)
    app()
