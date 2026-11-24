"""Evaluate a submitted solution inside the temporary evaluation container."""

import math
from pathlib import Path
import runpy


def evaluate(solution_path: Path, *, split: str, seed: int) -> float:
    """Return mean absolute error. This deterministic task does not use seed."""
    queries = {"dev": [2, 4, 8], "final": [-1, 7, 12]}
    if split not in queries:
        raise ValueError("split must be 'dev' or 'final'")
    inputs = queries[split]
    module = runpy.run_path(str(solution_path))
    predict = module.get('predict')
    if not callable(predict):
        raise ValueError('final_solution.py must define predict(xs)')
    predictions = predict(list(inputs))
    if not isinstance(predictions, list) or len(predictions) != len(inputs):
        raise ValueError("predict(xs) must return one prediction per input")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in predictions):
        raise ValueError("Every prediction must be a finite number")
    return sum(abs(prediction - (2 * x + 1)) for prediction, x in zip(predictions, inputs)) / len(inputs)
