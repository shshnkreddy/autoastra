"""Code policies for Gymnax Continuous Mountain Car."""

import json
from pathlib import Path

from autoastra.task import Task
from .evaluate import evaluate

_directory = Path(__file__).parent

task = Task(
    task_spec=json.loads((_directory / "task_spec.json").read_text()),
    fitness_fn=evaluate,
    image="autoastra-mountain-car:cpu",
    starter=(_directory / "starter.py").read_text(),
    context_files={name: (_directory / name).read_text() for name in
                   ("policy_api.md", "environment.md")},
    run_config=json.loads((_directory / "run_config.json").read_text()),
)
