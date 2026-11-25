"""Official development fitness; imported without requiring JAX on the host."""

from pathlib import Path


def evaluate(solution_path: Path, *, split: str, seed: int) -> float:
    if split != "dev":
        raise ValueError("Official fitness uses split='dev'; use the mountain_car wrapper for final testing")
    from .rollouts import evaluate_policy
    return evaluate_policy(solution_path, split="dev", seed=seed)["mean_return"]
