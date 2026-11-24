"""Create and print task previews without starting containers."""

import argparse
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import re

from ..task import Task
from ..task.layout import create_layout


def create_preview(task: Task, destination: str | Path, *, run_config: Mapping | None = None) -> Path:
    """Render a new task directory without starting a sandbox or evaluator."""
    return create_layout(task, destination, run_config=run_config)


def print_preview(preview: str | Path, output: str | Path | None = None) -> Path:
    """Write the directory tree and every UTF-8 file into one Markdown document.

    By default, write a sibling file named after the preview directory. The
    output must be new and outside the preview so it cannot include itself.
    Symlinks and non-regular files are rejected. File contents are fenced to
    preserve their source, including Markdown files with nested code blocks.
    """
    preview = Path(preview).absolute()
    if preview.is_symlink() or not preview.is_dir():
        raise ValueError("preview must be a real directory")
    output = Path(output).absolute() if output is not None else preview.with_name(preview.name + ".md")
    if output.resolve().is_relative_to(preview.resolve()):
        raise ValueError("The printed output must be outside the preview directory")

    tree = [preview.name + "/"]
    files = []

    def visit(directory, prefix=""):
        children = sorted(directory.iterdir(), key=lambda path: path.name)
        for index, path in enumerate(children):
            if path.is_symlink():
                raise ValueError(f"Preview contains a symbolic link at {path.relative_to(preview)}")
            last = index == len(children) - 1
            connector = "└── " if last else "├── "
            if path.is_dir():
                tree.append(prefix + connector + path.name + "/")
                visit(path, prefix + ("    " if last else "│   "))
            elif path.is_file():
                tree.append(prefix + connector + path.name)
                files.append((path.relative_to(preview), path.read_text(encoding="utf-8")))
            else:
                raise ValueError(f"Preview contains an unsupported file at {path.relative_to(preview)}")

    visit(preview)
    sections = ["# Research task", "```text\n" + "\n".join(tree) + "\n```"]
    languages = {".md": "markdown", ".json": "json", ".py": "python", ".toml": "toml"}
    for relative, content in files:
        # A longer outer fence preserves any code fences in the original file.
        longest = max((len(match) for match in re.findall(r"`+", content)), default=0)
        fence = "`" * max(3, longest + 1)
        language = languages.get(relative.suffix, "text")
        if content.startswith("#!/usr/bin/env python3"):
            language = "python"
        ending = "" if content.endswith("\n") else "\n"
        sections.append(f"## {relative.as_posix()}\n\n{fence}{language}\n{content}{ending}{fence}")

    document = "\n\n".join(sections) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Import reference such as tasks.affine:task")
    parser.add_argument("--output", required=True, type=Path, help="New preview directory")
    parser.add_argument("--provider", choices=("codex", "openrouter", "anthropic", "opencode"), help="Agent provider for this preview")
    parser.add_argument("--run-config", type=Path, help="Optional JSON run configuration")
    parser.add_argument("--print-to", type=Path, help="Also print the new preview into a single Markdown file")
    args = parser.parse_args(argv)
    try:
        module_name, separator, object_name = args.task.partition(":")
        if not separator or not module_name or not object_name:
            raise ValueError("Use a task import reference such as tasks.affine:task")
        task = getattr(importlib.import_module(module_name), object_name)
        config = json.loads(args.run_config.read_text()) if args.run_config else None
        if args.provider is not None:
            config = {**(config or {}), "provider": args.provider}
        preview = create_preview(task, args.output, run_config=config)
        print(preview)
        if args.print_to is not None:
            print(print_preview(preview, args.print_to))
    except (ValueError, TypeError, OSError, ImportError, AttributeError) as error:
        parser.exit(1, f"Preview creation failed. {error}\n")
