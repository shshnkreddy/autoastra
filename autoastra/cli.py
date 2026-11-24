"""Preview research tasks, run agents in Docker, and report their results."""

import argparse
from importlib import import_module
from pathlib import Path
import sys


# Load only the implementation used by the selected command.
COMMANDS = {
    "preview": ("autoastra.preview.preview", "Render the task files and instructions"),
    "run": ("autoastra.autoresearch.__main__", "Start a supervised autoresearch run"),
    "sandbox": ("autoastra.sandbox.sandbox", "Run a command in a task sandbox"),
    "result": ("autoastra.wrappers.basic", "Summarize a run and plot its fitness history"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="autoastra", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        subcommands.add_parser(name, help=description)
    args = parser.parse_args(argv[:1])
    handler = import_module(COMMANDS[args.command][0]).main
    if args.command in ("preview", "run", "sandbox"):
        # An installed console script starts in the venv's bin directory.
        # Let authors supply a module from their current project as well.
        sys.path.append(str(Path.cwd()))
    return handler(argv[1:])
