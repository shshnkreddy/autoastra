"""A deterministic affine regression demo."""

import json
from pathlib import Path

from autoastra.task import Task
from .evaluate import evaluate

task = Task(
    task_spec=json.loads(Path(__file__).with_name("task_spec.json").read_text()),
    fitness_fn=evaluate,
    image="python:3.12-slim",
    run_config=json.loads(Path(__file__).with_name("run_config.json").read_text()),
)
