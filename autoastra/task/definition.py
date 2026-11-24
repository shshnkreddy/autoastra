"""The problem author's specification, fitness evaluator, and software image."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class FitnessFunction(Protocol):
    def __call__(self, solution_path: Path, *, split: str, seed: int) -> float: ...


@dataclass
class Task:
    task_spec: dict
    fitness_fn: FitnessFunction
    image: str
    context_files: dict[str, str] = field(default_factory=dict)
    starter: str | None = None
    run_config: dict = field(default_factory=dict)

    def resolve_run_config(self, overrides=None):
        """Combine task defaults with explicit run settings, preserving null limits."""
        if not isinstance(self.run_config, Mapping):
            raise TypeError('Task run_config must be a mapping')
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError('run_config must be a mapping')
        return {**self.run_config, **(overrides or {})}

    def __post_init__(self):
        if (not isinstance(self.image, str) or not self.image
                or self.image.startswith("-")
                or any(character.isspace() or ord(character) < 32 for character in self.image)):
            raise ValueError("image must be a Docker image name, ID, or digest reference")
