"""Render the task files shared by previews and running sandboxes."""

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
import textwrap

from .definition import Task


SPEC_FIELDS = {
    "name": "Task",
    "description": "Objective",
    "solution_interface": "Solution interface",
    "metric": "Fitness metric",
    "direction": "Optimization direction",
}

COMMANDS = {
    "fitness-eval": "Request official development fitness for final_solution.py.",
    "experiment-ledger": "Register revisions, record experiments, inspect history, and restore source snapshots.",
    "lifetime": "Inspect elapsed development time and configured run budgets.",
}

AGENTS = """# Autoresearch agent

You are an autoresearch agent. {goal}

## Understand the task

The task information is provided in `context/task_spec.md`. It describes the
goal, required solution interface, available information, and constraints.

Work from `workspace/`. From that directory, read `../context/task_spec.md`
and `../context/fitness_api.md`. Implement your solution in `final_solution.py`.
Use `scratch.py` for informal tests. Keep the final solution self-contained.
The files in `context/`, the commands in `bin/`, and this file are read-only.

## Keep an experiment ledger

The ledger provides provenance for the research process. Someone reviewing
the run should be able to reconstruct what you tried, why you tried it,
which source was evaluated, what happened, and how the evidence informed
your next change. Record unsuccessful experiments as well as improvements.

Use `experiment-ledger` to maintain this history. Each registered revision
gets an attempt ID and an immutable source snapshot. The runner records
official evaluation results and errors against that exact revision. Your
entries explain the hypothesis, interpretation, limitations, and next step.

Follow this sequence for each experiment.

1. Review previous results with `experiment-ledger history`.
2. Make a focused change and register the source before official evaluation.
   State the hypothesis, planned change, and success criterion. Include
   `--parent` with the earlier attempt ID when revising an existing candidate.
3. Evaluate the registered source with `fitness-eval final_solution.py`.
4. Record informal test observations as exploratory. Include what you tested
   and the result so that someone else can understand the observation.
5. Conclude the attempt. Explain what the evidence supports, its limitations,
   and what you plan to try next. Choose `keep`, `reject`, or `follow_up`.

For example, register and score an initial candidate with

```sh
experiment-ledger register final_solution.py \\
  --title "Initial solution" \\
  --hypothesis "The initial method will satisfy the solution interface" \\
  --planned-change "Implement the first complete solution" \\
  --success-criterion "Complete an official development evaluation"
fitness-eval final_solution.py
```

Use the attempt ID returned by registration in subsequent entries. These
examples assume the returned ID is `A00`.

```sh
experiment-ledger observe --attempt-id A00 --kind exploratory \\
  --summary "Describe the informal test and its observed result"
experiment-ledger conclude --attempt-id A00 --decision follow_up \\
  --summary "Explain the result, limitations, and next experiment"
experiment-ledger inspect A00
experiment-ledger restore A00
```

The ledger and archived revisions are maintained by the runner and are readable
at `../ledger/ledger.jsonl` and `../ledger/snapshots/`. The ledger implementation
is readable at `../context/ledger.py`. These files are read-only. Use the ledger
commands to append records or restore a working copy. Register a new revision
after changing previously evaluated source.

## Use the fitness API

From `workspace/`, run `fitness-eval final_solution.py`. The commands in
`../bin/` will be on PATH during a research run. Calling `fitness-eval` without
a path uses `final_solution.py`.

The evaluator scores the registered solution on development data. Interpret
the score as described in `../context/task_spec.md`.
A failed evaluation produces an error record without a valid fitness score.
Consult `../context/fitness_api.md` for the interface and result semantics.

Informal checks in `scratch.py` help you investigate ideas. Label their
observations exploratory in the ledger. Official fitness comes from
`fitness-eval`. Calling that command from a scratch test still counts as
an official evaluation.

Use `lifetime` to check elapsed time and remaining evaluation allowances.
When present, `../context/run_config.json` records the effective run settings.
A `null` value for `max_tokens` or `max_evaluations` means that budget is unlimited.
Token usage is available when the model runner reports it; otherwise it is
shown as unknown. Time starts when the sandbox service starts. New evaluations
are rejected when a configured time, evaluation, or tracked token budget is
exhausted. You can still inspect the ledger and record conclusions.

{selection_instruction}

Commands print JSON. Evaluation failures print an error record and return a
nonzero exit code. Ledger entries and immutable source snapshots are stored
outside your workspace and remain available to the host after shutdown.
"""

FITNESS_API = """# Fitness API

## Evaluate a solution

{score_scope}From `workspace/`, submit your registered solution for development scoring.

```sh
fitness-eval final_solution.py
```

Omitting the path selects `final_solution.py`. Register the exact source first
with `experiment-ledger register`. Evaluation uses that immutable snapshot.
If the same source has several registrations, the latest registration is
used. Pass `--attempt-id A00` to select a specific matching registration.
Changing the source requires a new registration.

A successful evaluation returns a finite numeric fitness value associated
with the evaluated revision. See `task_spec.md` for the objective and the
meaning of the development score. An invalid solution or failed execution produces
an error record without a valid score.

The runner records the source snapshot and evaluation outcome in the
experiment ledger. Each started official evaluation counts towards the
configured allowance, including evaluations that fail. Informal local tests
can be recorded separately as exploratory observations.

## Evaluator interface

The task author supplies this importable Python function in a task module.

```python
def evaluate(solution_path, *, split, seed) -> float:
    ...
```

`solution_path` points to the frozen source inside a temporary evaluation
container. The fitness function can load that Python file, call its functions,
and calculate fitness directly. The evaluator and submitted solution share
the same container and Python process. Submitted code is trusted not to
interfere with scoring.

There are at most two containers for a run: the persistent development
container and one temporary evaluation container. The evaluator uses the
task's image and configured resource limits. It cannot access the development
workspace, ledger, or Docker socket. Only the host service writes official
results to the ledger.

The command always uses the development split. The runner supplies `seed`
(zero by default). `evaluation_timeout_seconds` limits the entire evaluation,
including container startup and submitted code; it defaults to 30 seconds.
Remaining run time can shorten this limit. Only one official evaluation runs
at a time. Final scoring is not exposed through the agent command.
"""

def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _agents_markdown(spec, selection='best-fitness'):
    instruction = ('The runner restores the best officially scored revision. The earliest score wins a tie.'
                   if selection == 'best-fitness' else
                   'Choose the scored revision you expect to best satisfy the task objective. Run '
                   'experiment-ledger select A00 --reason "your reasoning" with its attempt ID. '
                   'You may update your choice. Without a selection the run has no final solution.')
    direction, preference = {
        "minimize": ("minimise", "Lower"),
        "maximize": ("maximise", "Higher"),
    }[spec["direction"]]
    metric = spec.get('metric_label', spec['metric'].replace('_', ' '))
    goal = spec.get('development_proxy') or (
        f'Your goal is to {direction} the value returned by the predefined fitness function. '
        f'This task uses {metric}. {preference} scores are better.'
    )
    if spec.get('development_proxy') and selection == 'best-fitness':
        rank = 'lowest' if spec['direction'] == 'minimize' else 'highest'
        instruction = (f'The runner restores the revision with the {rank} {metric}. '
                       'The earliest score wins a tie.')
    return AGENTS.format(selection_instruction=instruction, goal=goal)


def _detail_markdown(value):
    """Render JSON-compatible task details as readable Markdown."""
    if isinstance(value, dict):
        return "\n\n".join(
            f"**{str(key).replace('_', ' ').capitalize()}**\n\n{_detail_markdown(item)}"
            for key, item in value.items()
        )
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and item for item in value):
            columns = list(value[0])
            if all(list(item) == columns and
                   all(not isinstance(cell, (dict, list)) for cell in item.values())
                   for item in value):
                rows = [[str(key).replace("_", " ").capitalize() for key in columns]]
                rows.append(["---"] * len(columns))
                rows.extend([_detail_markdown(item[key]) for key in columns] for item in value)
                return "\n".join("| " + " | ".join(
                    cell.replace("|", "\\|").replace("\n", "<br>") for cell in row
                ) + " |" for row in rows)
        return "\n".join("- " + textwrap.indent(_detail_markdown(item), "  ").lstrip()
                         for item in value)
    if value is None:
        return "Not specified"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _task_markdown(spec):
    sections = ["# " + spec.get("title", spec["name"])]
    for key, label in SPEC_FIELDS.items():
        if key in ('metric', 'direction') and spec.get('development_proxy'):
            if key == 'metric':
                sections.append(f"## {spec.get('metric_label', 'Development score')}\n\n{spec['development_proxy']}")
            continue
        if key != "name":
            value = spec[key]
            if key == "metric":
                value = spec.get("metric_description", spec.get("metric_label", value.replace("_", " ")))
            elif key == "direction":
                value = {"minimize": "Minimise the fitness value. Lower scores are better.",
                         "maximize": "Maximise the fitness value. Higher scores are better."}[value]
            sections.append(f"## {label}\n\n{value}")
    for key, value in spec.items():
        if key not in {*SPEC_FIELDS, "title", "metric_description", "metric_label", "development_proxy", "zsc"}:
            label = str(key).replace("_", " ").capitalize()
            sections.append(f"## {label}\n\n{_detail_markdown(value)}")
    return "\n\n".join(sections) + "\n"


def _command(name, description):
    return (Path(__file__).resolve().parents[1] / 'sandbox' / 'commands.py').read_text(encoding='utf-8')


def create_layout(task: Task, destination: str | Path, *, run_config: Mapping | None = None) -> Path:
    """Create a new task workspace directory and return its absolute path.

    The destination must not exist. Explicit run settings override task defaults;
    the resolved JSON configuration is recorded in context. No task code runs
    during rendering.
    """
    if not isinstance(task, Task) or not isinstance(task.task_spec, dict):
        raise TypeError("Provide a Task with a dictionary task_spec")
    if not callable(task.fitness_fn):
        raise TypeError("fitness_fn must be callable")
    for field in SPEC_FIELDS:
        if not isinstance(task.task_spec.get(field), str) or not task.task_spec[field].strip():
            raise ValueError(f"task_spec requires a nonempty {field}")
    for field in ("title", "metric_description", "metric_label", "development_proxy"):
        if field in task.task_spec and (not isinstance(task.task_spec[field], str) or not task.task_spec[field].strip()):
            raise ValueError(f"{field} must be a nonempty string when supplied")
    if task.task_spec["direction"] not in {"minimize", "maximize"}:
        raise ValueError("direction must be minimize or maximize")
    if run_config is not None and not isinstance(run_config, Mapping):
        raise TypeError("run_config must be a JSON-compatible mapping")

    # Snapshot caller-owned data before rendering or writing any output.
    spec = json.loads(_json(task.task_spec))
    resolved = task.resolve_run_config(run_config)
    config = json.loads(_json(resolved)) if resolved or run_config is not None else None
    provider = (config or {}).get("provider", "codex")
    if provider not in ("codex", "openrouter", "anthropic", "opencode"):
        raise ValueError("Unknown model provider")
    instructions = "task/CLAUDE.md" if provider == "anthropic" else "task/AGENTS.md"
    score_scope = spec.get("development_proxy", "")
    files = {
        instructions: _agents_markdown(spec, (config or {}).get('selection', 'best-fitness')),
        "task/context/task_spec.md": _task_markdown(spec),
        "task/context/fitness_api.md": FITNESS_API.format(score_scope=score_scope + "\n\n" if score_scope else ""),
        "task/context/ledger.py": (Path(__file__).resolve().parents[1] / 'sandbox' / 'runtime.py').read_text(encoding='utf-8'),
        "task/ledger/ledger.jsonl": '',
        "task/workspace/final_solution.py": task.starter or '"""Implement the solution interface described in ../context/task_spec.md."""\n',
        "task/workspace/scratch.py": '"""Optional informal tests. Record observations as exploratory."""\n',
    }
    for relative, content in task.context_files.items():
        path = Path(relative)
        key = f'task/context/{path}'
        if path.is_absolute() or '..' in path.parts or not path.name or key in files or path.name == 'run_config.json':
            raise ValueError(f'Invalid context asset: {relative}')
        files[key] = content
    if config is not None:
        files["task/context/run_config.json"] = _json(config)
    for name, description in COMMANDS.items():
        files[f"task/bin/{name}"] = _command(name, description)

    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    directories = set()
    try:
        for name, content in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            directories.add(path.parent)
            path.write_text(content, encoding="utf-8")
            if name.startswith("task/bin/"):
                path.chmod(0o555)
            elif name.startswith(("task/context/", "task/ledger/")) or name == instructions:
                path.chmod(0o444)
        for name in ("task/context", "task/bin", "task/ledger/snapshots", "task/ledger"):
            directory = destination / name
            directory.mkdir(parents=True, exist_ok=True)
            directories.add(directory)
            directory.chmod(0o555)
    except BaseException:
        for directory in directories:
            directory.chmod(0o700)
        shutil.rmtree(destination)
        raise
    return destination
