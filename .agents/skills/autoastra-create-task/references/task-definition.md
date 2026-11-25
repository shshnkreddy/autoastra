# Task definition and evaluator

Use this guide when creating or adapting the task's Python package and its agent-facing instructions. The source of truth is [Task](../../../../autoastra/task/definition.py), the [layout renderer](../../../../autoastra/task/layout.py), and the [affine example](../../../../tasks/affine/__init__.py).

## Files and fields

A useful starting layout is:

```text
my_tasks/example/
├── __init__.py          # Exports task
├── task_spec.json       # Problem description as structured data
├── evaluate.py         # Importable fitness function
└── run_config.json     # Task's default budgets
```

Make `my_tasks` an importable package as well. Additional starter, public context, and Docker files are optional. Users can keep the dictionaries inline instead of using JSON files.

```python
import json
from pathlib import Path

from autoastra.task import Task
from .evaluate import evaluate

directory = Path(__file__).parent
task = Task(
    task_spec=json.loads((directory / "task_spec.json").read_text()),
    fitness_fn=evaluate,
    image="my-task:latest",
    run_config=json.loads((directory / "run_config.json").read_text()),
)
```

Run this object as `my_tasks.example:task`. Replace the image with one built or pulled on the execution machine. Keep task-module imports lightweight so previews do not require the environment's training libraries on the host.

| Field | Meaning |
| --- | --- |
| `task_spec` | JSON-compatible dictionary rendered into `context/task_spec.md`. |
| `fitness_fn` | Importable evaluator function, not a lambda, closure, or function defined in `__main__`. |
| `image` | Docker image name, ID, or digest containing Python and environment dependencies. |
| `starter` | Optional Python source string for `workspace/final_solution.py`. |
| `context_files` | Optional mapping of relative paths to text contents, placed under read-only `context/`. Do not overwrite generated files. |
| `run_config` | Optional default budgets and selection settings. Explicit run overrides take precedence. |

## Write a legible specification

The renderer requires nonempty strings for `name`, `description`, `solution_interface`, `metric`, and `direction`. Direction must be `minimize` or `maximize`. Use `title`, `metric_description`, `constraints`, examples, and other JSON-compatible fields when useful; extra fields become readable sections.

Explain what the solution should achieve, permitted information and resources, exact function signatures, input/output types and shapes, invalid outputs, and how fitness is calculated. Include a small example where it removes ambiguity. `final_solution.py` must be self-contained under the evaluator's available dependencies; `scratch.py` and other development files are not submitted with it.

If development fitness is only a proxy, use `development_proxy` to explain both the real objective and what the agent can measure. Keep `direction` aligned with the returned development fitness. Do not claim that higher self-play necessarily implies better cross-play or that held-out scoring happens automatically.

## Evaluator contract

```python
def evaluate(solution_path, *, split, seed) -> float:
    ...
```

The evaluator receives the frozen `final_solution.py` path. Load it, validate its interface, run the task, and return one finite scalar score. Raise an informative exception for invalid submissions; do not silently convert failures into successful scores. Use the supplied seed for stochastic evaluation, and explain whether the task is deterministic.

AutoAstra copies the evaluator's Python package into the evaluation container. Put private scoring helpers and small private fixtures in that package, or explicitly arrange evaluator-only dependencies. Imports from unrelated host directories and arbitrary host data paths will not work automatically. Inspect the [packaging implementation](../../../../autoastra/sandbox/evaluation.py) when adapting a large codebase.

Official `fitness-eval` calls use `split="dev"`. A `final` branch in the evaluator does not cause the runner to call it: implement held-out evaluation in a wrapper when required. See [wrappers](wrappers-and-extensions.md).

The submitted solution and fitness function share the temporary evaluator container and Python process. The development agent cannot inspect that running evaluator, but submitted Python is trusted not to interfere with scoring. A task requiring stronger isolation needs an implementation change, not a promise in its prompt.

## Budgets, selection, and provenance

For a two-hour task with no token or evaluation cap:

```json
{
  "time_seconds": 7200,
  "max_tokens": null,
  "max_evaluations": null,
  "evaluation_timeout_seconds": 600
}
```

Time is an upper bound; a session may finish sooner. `null` removes the token or evaluation cap; zero does not mean unlimited. Size evaluation timeouts for container startup plus the actual workload. Provider, model, CPUs, memory, and GPUs are separate launch options rather than task budget fields.

Selection defaults to the best officially scored development revision. With `--selection model`, the research agent must nominate a scored revision using `experiment-ledger select ATTEMPT_ID --reason "..."`. Record that choice separately from the best development score.

Keep the generated ledger workflow: register the exact candidate, evaluate it officially, and record the conclusion. Informal observations belong in exploratory entries. Official scores refer to immutable snapshots; do not manufacture official scores by editing the ledger. Finished outputs retain the selected solution, `result.json`, readable `ledger.md`, `ledger.jsonl`, and source snapshots.
