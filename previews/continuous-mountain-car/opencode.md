# Research task

```text
opencode/
└── task/
    ├── AGENTS.md
    ├── bin/
    │   ├── experiment-ledger
    │   ├── fitness-eval
    │   └── lifetime
    ├── context/
    │   ├── environment.md
    │   ├── fitness_api.md
    │   ├── ledger.py
    │   ├── policy_api.md
    │   ├── run_config.json
    │   └── task_spec.md
    ├── ledger/
    │   ├── ledger.jsonl
    │   └── snapshots/
    └── workspace/
        ├── final_solution.py
        └── scratch.py
```

## task/AGENTS.md

````markdown
# Autoresearch agent

You are an autoresearch agent. Your goal is to maximise the value returned by the predefined fitness function. This task uses mean episode return. Higher scores are better.

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
experiment-ledger register final_solution.py \
  --title "Initial solution" \
  --hypothesis "The initial method will satisfy the solution interface" \
  --planned-change "Implement the first complete solution" \
  --success-criterion "Complete an official development evaluation"
fitness-eval final_solution.py
```

Use the attempt ID returned by registration in subsequent entries. These
examples assume the returned ID is `A00`.

```sh
experiment-ledger observe --attempt-id A00 --kind exploratory \
  --summary "Describe the informal test and its observed result"
experiment-ledger conclude --attempt-id A00 --decision follow_up \
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

The runner restores the best officially scored revision. The earliest score wins a tie.

Commands print JSON. Evaluation failures print an error record and return a
nonzero exit code. Ledger entries and immutable source snapshots are stored
outside your workspace and remain available to the host after shutdown.
````

## task/bin/experiment-ledger

```python
#!/usr/bin/env python3
"""Client installed under each of the three task/bin command names."""

import argparse
import base64
import json
from pathlib import Path
import socket
import sys

LIMIT = 4 * 1024 * 1024
WORKSPACE = Path('/task/workspace')


def source_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('Source must be inside /task/workspace')
    return path


def main():
    name = Path(sys.argv[0]).name
    parser = argparse.ArgumentParser(prog=name)
    if name == 'fitness-eval':
        parser.description = 'Evaluate an exact registered source revision on the development split.'
        parser.add_argument('source', nargs='?', default='final_solution.py')
        parser.add_argument('--attempt-id')
    elif name == 'experiment-ledger':
        parser.description = 'Record experiments and inspect immutable source and evaluation history.'
        commands = parser.add_subparsers(dest='action', required=True)
        register = commands.add_parser('register')
        register.add_argument('source', nargs='?', default='final_solution.py')
        for field in ('title', 'hypothesis', 'planned-change', 'success-criterion'):
            register.add_argument('--' + field, required=True)
        register.add_argument('--parent')
        select = commands.add_parser('select')
        select.add_argument('attempt_id')
        select.add_argument('--reason', required=True)
        commands.add_parser('history')
        inspect = commands.add_parser('inspect')
        inspect.add_argument('attempt_id')
        restore = commands.add_parser('restore')
        restore.add_argument('attempt_id')
        restore.add_argument('--output', default='final_solution.py')
        observe = commands.add_parser('observe')
        observe.add_argument('--attempt-id', required=True)
        observe.add_argument('--kind', choices=['exploratory'], default='exploratory')
        observe.add_argument('--summary', required=True)
        conclude = commands.add_parser('conclude')
        conclude.add_argument('--attempt-id', required=True)
        conclude.add_argument('--decision', choices=['keep', 'reject', 'follow_up'], required=True)
        conclude.add_argument('--summary', required=True)
    elif name == 'lifetime':
        parser.description = 'Show elapsed time, evaluation usage, and available budget accounting.'
    else:
        parser.error('Unknown command')
    args = vars(parser.parse_args())
    try:
        payload = {'command': name, **args}
        if 'source' in args:
            path = source_path(args['source'])
            with path.open('rb') as stream:
                source = stream.read(1024 * 1024 + 1)
            if len(source) > 1024 * 1024:
                raise ValueError('Source must be at most 1 MiB')
            source.decode('utf-8')
            payload['source'] = base64.b64encode(source).decode()
        output = source_path(args['output']) if args.get('action') == 'restore' else None
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3600)
            connection.connect('/run/autoastra/agent.sock')
            connection.sendall(json.dumps(payload, allow_nan=False).encode() + b'\n')
            with connection.makefile('rb') as stream:
                line = stream.readline(LIMIT + 1)
        if len(line) > LIMIT or not line.endswith(b'\n'):
            raise RuntimeError('Invalid service response')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['error'])
        result = response['result']
        if output is not None:
            output.write_bytes(base64.b64decode(result.pop('source'), validate=True))
            result['restored_to'] = str(output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get('status') == 'error' else 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'{name}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
```

## task/bin/fitness-eval

```python
#!/usr/bin/env python3
"""Client installed under each of the three task/bin command names."""

import argparse
import base64
import json
from pathlib import Path
import socket
import sys

LIMIT = 4 * 1024 * 1024
WORKSPACE = Path('/task/workspace')


def source_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('Source must be inside /task/workspace')
    return path


def main():
    name = Path(sys.argv[0]).name
    parser = argparse.ArgumentParser(prog=name)
    if name == 'fitness-eval':
        parser.description = 'Evaluate an exact registered source revision on the development split.'
        parser.add_argument('source', nargs='?', default='final_solution.py')
        parser.add_argument('--attempt-id')
    elif name == 'experiment-ledger':
        parser.description = 'Record experiments and inspect immutable source and evaluation history.'
        commands = parser.add_subparsers(dest='action', required=True)
        register = commands.add_parser('register')
        register.add_argument('source', nargs='?', default='final_solution.py')
        for field in ('title', 'hypothesis', 'planned-change', 'success-criterion'):
            register.add_argument('--' + field, required=True)
        register.add_argument('--parent')
        select = commands.add_parser('select')
        select.add_argument('attempt_id')
        select.add_argument('--reason', required=True)
        commands.add_parser('history')
        inspect = commands.add_parser('inspect')
        inspect.add_argument('attempt_id')
        restore = commands.add_parser('restore')
        restore.add_argument('attempt_id')
        restore.add_argument('--output', default='final_solution.py')
        observe = commands.add_parser('observe')
        observe.add_argument('--attempt-id', required=True)
        observe.add_argument('--kind', choices=['exploratory'], default='exploratory')
        observe.add_argument('--summary', required=True)
        conclude = commands.add_parser('conclude')
        conclude.add_argument('--attempt-id', required=True)
        conclude.add_argument('--decision', choices=['keep', 'reject', 'follow_up'], required=True)
        conclude.add_argument('--summary', required=True)
    elif name == 'lifetime':
        parser.description = 'Show elapsed time, evaluation usage, and available budget accounting.'
    else:
        parser.error('Unknown command')
    args = vars(parser.parse_args())
    try:
        payload = {'command': name, **args}
        if 'source' in args:
            path = source_path(args['source'])
            with path.open('rb') as stream:
                source = stream.read(1024 * 1024 + 1)
            if len(source) > 1024 * 1024:
                raise ValueError('Source must be at most 1 MiB')
            source.decode('utf-8')
            payload['source'] = base64.b64encode(source).decode()
        output = source_path(args['output']) if args.get('action') == 'restore' else None
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3600)
            connection.connect('/run/autoastra/agent.sock')
            connection.sendall(json.dumps(payload, allow_nan=False).encode() + b'\n')
            with connection.makefile('rb') as stream:
                line = stream.readline(LIMIT + 1)
        if len(line) > LIMIT or not line.endswith(b'\n'):
            raise RuntimeError('Invalid service response')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['error'])
        result = response['result']
        if output is not None:
            output.write_bytes(base64.b64decode(result.pop('source'), validate=True))
            result['restored_to'] = str(output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get('status') == 'error' else 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'{name}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
```

## task/bin/lifetime

```python
#!/usr/bin/env python3
"""Client installed under each of the three task/bin command names."""

import argparse
import base64
import json
from pathlib import Path
import socket
import sys

LIMIT = 4 * 1024 * 1024
WORKSPACE = Path('/task/workspace')


def source_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('Source must be inside /task/workspace')
    return path


def main():
    name = Path(sys.argv[0]).name
    parser = argparse.ArgumentParser(prog=name)
    if name == 'fitness-eval':
        parser.description = 'Evaluate an exact registered source revision on the development split.'
        parser.add_argument('source', nargs='?', default='final_solution.py')
        parser.add_argument('--attempt-id')
    elif name == 'experiment-ledger':
        parser.description = 'Record experiments and inspect immutable source and evaluation history.'
        commands = parser.add_subparsers(dest='action', required=True)
        register = commands.add_parser('register')
        register.add_argument('source', nargs='?', default='final_solution.py')
        for field in ('title', 'hypothesis', 'planned-change', 'success-criterion'):
            register.add_argument('--' + field, required=True)
        register.add_argument('--parent')
        select = commands.add_parser('select')
        select.add_argument('attempt_id')
        select.add_argument('--reason', required=True)
        commands.add_parser('history')
        inspect = commands.add_parser('inspect')
        inspect.add_argument('attempt_id')
        restore = commands.add_parser('restore')
        restore.add_argument('attempt_id')
        restore.add_argument('--output', default='final_solution.py')
        observe = commands.add_parser('observe')
        observe.add_argument('--attempt-id', required=True)
        observe.add_argument('--kind', choices=['exploratory'], default='exploratory')
        observe.add_argument('--summary', required=True)
        conclude = commands.add_parser('conclude')
        conclude.add_argument('--attempt-id', required=True)
        conclude.add_argument('--decision', choices=['keep', 'reject', 'follow_up'], required=True)
        conclude.add_argument('--summary', required=True)
    elif name == 'lifetime':
        parser.description = 'Show elapsed time, evaluation usage, and available budget accounting.'
    else:
        parser.error('Unknown command')
    args = vars(parser.parse_args())
    try:
        payload = {'command': name, **args}
        if 'source' in args:
            path = source_path(args['source'])
            with path.open('rb') as stream:
                source = stream.read(1024 * 1024 + 1)
            if len(source) > 1024 * 1024:
                raise ValueError('Source must be at most 1 MiB')
            source.decode('utf-8')
            payload['source'] = base64.b64encode(source).decode()
        output = source_path(args['output']) if args.get('action') == 'restore' else None
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3600)
            connection.connect('/run/autoastra/agent.sock')
            connection.sendall(json.dumps(payload, allow_nan=False).encode() + b'\n')
            with connection.makefile('rb') as stream:
                line = stream.readline(LIMIT + 1)
        if len(line) > LIMIT or not line.endswith(b'\n'):
            raise RuntimeError('Invalid service response')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['error'])
        result = response['result']
        if output is not None:
            output.write_bytes(base64.b64decode(result.pop('source'), validate=True))
            result['restored_to'] = str(output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get('status') == 'error' else 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'{name}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
```

## task/context/environment.md

```markdown
# Environment

This task uses Gymnax `ContinuousMountainCar`, pinned to commit
`77d4863333e7ec203d0b4a43f998b2b691fdb997`.
[Read the environment source](https://github.com/RobertTLange/gymnax/blob/77d4863333e7ec203d0b4a43f998b2b691fdb997/gymnax/environments/classic_control/continuous_mountain_car.py).
The same public source is installed in the Docker image.

The observation is `[position, velocity]`. Position is bounded by `[-1.2, 0.6]`
and velocity by `[-0.07, 0.07]`. Each episode starts with position uniform in
`[-0.6, -0.4]` and velocity zero.

At each step, the environment adds
`0.0015 * action - 0.0025 * cos(3 * position)` to the velocity, clips it to its
bounds, and adds it to the position, which is also clipped. Use the pinned
implementation for exact boundary behaviour; other Mountain Car libraries
need not implement identical dynamics.

Reaching position `0.45` with nonnegative velocity ends the episode successfully.
The maximum episode length is 999 steps. Reward at each step is
`100 * reached_goal - 0.1 * action**2`. The successful step counts, and no later
rewards count. Fitness is the mean undiscounted episode return, not success
rate or speed alone. Success rate, episode length, and summed squared actions
are additional diagnostics in the final test report.

The image includes Python, JAX (CPU), NumPy, Gymnax, Flax, and Matplotlib.
CPU execution is sufficient for this task. No GPU is required.
```

## task/context/fitness_api.md

````markdown
# Fitness API

## Evaluate a solution

From `workspace/`, submit your registered solution for development scoring.

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
````

## task/context/ledger.py

```python
"""Host-owned experiment records, evaluation dispatch, and lifetime accounting."""

import base64
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading
import time

from .evaluation import EvaluationJob
from .protocol import Server


def validate_run_config(value):
    if value is not None and not isinstance(value, Mapping):
        raise TypeError('run_config must be a mapping')
    config = dict(value or {})
    for name in ('max_evaluations', 'max_tokens'):
        if config.get(name) is not None and (type(config[name]) is not int or config[name] < 0):
            raise ValueError(f'{name} must be a nonnegative integer or null for no limit')
    if 'seed' in config and (type(config['seed']) is not int or config['seed'] < 0):
        raise ValueError('seed must be a nonnegative integer')
    for name in ('time_seconds', 'evaluation_timeout_seconds'):
        if name in config and (type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] <= 0):
            raise ValueError(f'{name} must be a positive finite number')
    if config.get('selection', 'best-fitness') not in ('best-fitness', 'model'):
        raise ValueError('Unknown final selection mode')
    return config


class Runtime:
    def __init__(self, task, directory, image_id, image_path, config, run_config):
        self.task = task
        self.directory = directory / '.runtime'
        self.directory.mkdir(mode=0o700)
        self.snapshots = self.directory / 'snapshots'
        self.snapshots.mkdir()
        self.ledger = self.directory / 'ledger.jsonl'
        self.ledger.touch(mode=0o600)
        self.image_id = image_id
        self.image_path = image_path
        self.config = config
        self.run_config = validate_run_config(run_config)
        self.started = time.monotonic()
        self.finished = None
        self.closed = False
        self.lock = threading.RLock()
        self.evaluation_lock = threading.Lock()
        self.evaluation_done = threading.Event()
        self.evaluation_done.set()
        self.job = None
        self.events = []
        self.selected = None
        self.attempts = {}
        self.evaluations = []
        self.tokens_used = None
        try:
            source = inspect.getsourcefile(task.fitness_fn)
        except TypeError:
            source = None
        self.append('run_started', {
            'task_spec': task.task_spec, 'image': task.image, 'image_id': image_id,
            'sandbox_config': asdict(config), 'run_config': self.run_config,
            'fitness_module': getattr(task.fitness_fn, '__module__', None),
            'fitness_function': getattr(task.fitness_fn, '__qualname__', None),
            'fitness_module_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
        })
        self.server = Server(self.directory / 'ipc/agent.sock', self.dispatch)

    def append(self, kind, payload):
        event = {
            'sequence': len(self.events), 'type': kind,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': round(time.monotonic() - self.started, 6),
            'previous_hash': self.events[-1]['event_hash'] if self.events else None,
            **payload,
        }
        canonical = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False)
        event['event_hash'] = hashlib.sha256(canonical.encode()).hexdigest()
        with self.ledger.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    def lifetime(self):
        elapsed = (self.finished or time.monotonic()) - self.started
        time_limit = self.run_config.get('time_seconds')
        max_evaluations = self.run_config.get('max_evaluations')
        max_tokens = self.run_config.get('max_tokens')
        return {
            'elapsed_seconds': round(elapsed, 3), 'time_seconds': time_limit,
            'remaining_seconds': max(0, round(time_limit - elapsed, 3)) if time_limit is not None else None,
            'evaluations_used': len(self.evaluations), 'max_evaluations': max_evaluations,
            'evaluations_remaining': max(0, max_evaluations - len(self.evaluations)) if max_evaluations is not None else None,
            'tokens_used': self.tokens_used, 'max_tokens': max_tokens,
            'tokens_remaining': max(0, max_tokens - self.tokens_used) if max_tokens is not None and self.tokens_used is not None else None,
            'token_tracking': 'reported by runner' if self.tokens_used is not None else 'not connected',
            'closed': self.closed,
        }

    def record_tokens(self, count):
        """Trusted runner hook. This operation is deliberately absent from RPC."""
        if type(count) is not int or count < 0:
            raise ValueError('Token count must be a nonnegative integer')
        with self.lock:
            if self.closed:
                raise RuntimeError('Runtime is closed')
            self.tokens_used = (self.tokens_used or 0) + count
            self.append('token_usage', {'tokens_added': count, 'tokens_used': self.tokens_used})

    @staticmethod
    def source(payload):
        value = payload.get('source')
        if not isinstance(value, str):
            raise ValueError('Provide source bytes')
        source = base64.b64decode(value, validate=True)
        if len(source) > 1024 * 1024:
            raise ValueError('Source must be at most 1 MiB')
        source.decode('utf-8')
        return source, hashlib.sha256(source).hexdigest()

    @staticmethod
    def text(payload, key):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 16000:
            raise ValueError(f'{key} must be nonempty text of at most 16000 characters')
        return value

    def attempt(self, attempt_id):
        if not isinstance(attempt_id, str) or attempt_id not in self.attempts:
            raise ValueError('Unknown attempt ID')
        return self.attempts[attempt_id]

    def best(self):
        successes = [result for result in self.evaluations if result['status'] == 'ok']
        if not successes:
            return None
        direction = 1 if self.task.task_spec['direction'] == 'minimize' else -1
        return min(successes, key=lambda result: direction * result['fitness'])

    def dispatch(self, payload):
        command = payload.get('command')
        if command == 'fitness-eval':
            return self.evaluate(payload)
        with self.lock:
            if self.closed:
                raise RuntimeError('Runtime is closed')
            if command == 'lifetime':
                return self.lifetime()
            if command != 'experiment-ledger':
                raise ValueError('Unknown command')
            action = payload.get('action')
            if action == 'register':
                source, digest = self.source(payload)
                fields = {key: self.text(payload, key) for key in ('title', 'hypothesis', 'planned_change', 'success_criterion')}
                parent = payload.get('parent')
                if parent is not None:
                    self.attempt(parent)
                attempt_id = f'A{len(self.attempts):02d}'
                snapshot = self.snapshots / (attempt_id + '.py')
                with snapshot.open('xb') as stream:
                    stream.write(source)
                snapshot.chmod(0o444)
                attempt = {'attempt_id': attempt_id, 'source_sha256': digest, 'parent': parent, **fields}
                self.append('registered', attempt)
                self.attempts[attempt_id] = attempt
                return attempt
            if action == 'history':
                return {'attempts': list(self.attempts.values()), 'evaluations': self.evaluations,
                        'best': self.best()}
            attempt = self.attempt(payload.get('attempt_id'))
            if action == 'select':
                if self.run_config.get('selection') != 'model':
                    raise ValueError('Model selection is disabled for this run')
                scored = next((e for e in reversed(self.evaluations) if e['attempt_id'] == attempt['attempt_id']
                               and e['status'] == 'ok'), None)
                if scored is None:
                    raise ValueError('Select an attempt with a successful evaluation')
                reason = self.text(payload, 'reason')
                self.append('selection', {**scored, 'reason': reason})
                self.selected = dict(scored, reason=reason)
                return self.selected
            if action == 'inspect':
                return {**attempt, 'events': [event for event in self.events if event.get('attempt_id') == attempt['attempt_id']]}
            if action == 'restore':
                source = (self.snapshots / (attempt['attempt_id'] + '.py')).read_bytes()
                self.append('restore_requested', {'attempt_id': attempt['attempt_id']})
                return {'attempt_id': attempt['attempt_id'], 'source_sha256': attempt['source_sha256'],
                        'source': base64.b64encode(source).decode()}
            summary = self.text(payload, 'summary')
            if action == 'observe':
                if payload.get('kind') != 'exploratory':
                    raise ValueError('Agent observations must be exploratory')
                return self.append('observation', {'attempt_id': attempt['attempt_id'], 'kind': 'exploratory', 'summary': summary})
            if action == 'conclude':
                if payload.get('decision') not in ('keep', 'reject', 'follow_up'):
                    raise ValueError('Choose keep, reject, or follow_up')
                return self.append('conclusion', {'attempt_id': attempt['attempt_id'], 'decision': payload['decision'], 'summary': summary})
            raise ValueError('Unknown ledger action')

    def evaluate(self, payload):
        if not self.evaluation_lock.acquire(blocking=False):
            raise RuntimeError('An evaluation is already running')
        result = None
        self.evaluation_done.clear()
        try:
            with self.lock:
                if self.closed:
                    raise RuntimeError('Runtime is closed')
                _, digest = self.source(payload)
                requested = payload.get('attempt_id')
                if requested is not None:
                    attempt = self.attempt(requested)
                else:
                    matches = [attempt for attempt in self.attempts.values() if attempt['source_sha256'] == digest]
                    if not matches:
                        raise ValueError('Register this exact source with experiment-ledger before evaluating')
                    attempt = matches[-1]
                if attempt['source_sha256'] != digest:
                    raise ValueError('Source changed since registration; register a new revision')
                budget = self.lifetime()
                for key in ('remaining_seconds', 'evaluations_remaining', 'tokens_remaining'):
                    if budget[key] is not None and budget[key] <= 0:
                        raise RuntimeError('Run budget exhausted: ' + key)
                timeout = self.run_config.get('evaluation_timeout_seconds', 30)
                if budget['remaining_seconds'] is not None:
                    timeout = min(timeout, budget['remaining_seconds'])
                result = {'evaluation_id': f'E{len(self.evaluations):02d}', 'attempt_id': attempt['attempt_id'],
                          'source_sha256': digest, 'split': 'dev', 'seed': self.run_config.get('seed', 0),
                          'image_id': self.image_id, 'status': 'running', 'fitness': None}
                self.append('evaluation_started', result)
                self.evaluations.append(result)
                self.job = EvaluationJob(self.task, self.snapshots / (attempt['attempt_id'] + '.py'),
                                         self.directory / 'evaluations' / result['evaluation_id'],
                                         self.image_id, self.image_path, self.config, result['seed'], timeout)
            score = self.job.run()
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('Fitness must be a finite number')
            with self.lock:
                result.update(status='ok', fitness=float(score))
        except Exception as error:
            if result is None:
                raise
            with self.lock:
                result.update(status='error', error=type(error).__name__ + ': ' + str(error)[:2000])
        finally:
            with self.lock:
                if result is not None:
                    self.append('evaluation_finished', result)
                self.job = None
            self.evaluation_done.set()
            self.evaluation_lock.release()
        return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            job = self.job
        if job is not None:
            job.cancel()
        self.evaluation_done.wait(timeout=60)
        self.server.close()
        with self.lock:
            self.finished = time.monotonic()
            self.append('run_closed', {'lifetime': self.lifetime(), 'best': self.best()})
```

## task/context/policy_api.md

````markdown
# Policy API

Implement these two functions in `final_solution.py`:

```python
import jax.numpy as jnp

def init_state():
    return jnp.asarray(0, dtype=jnp.int32)

def act(obs, state, rng):
    position, velocity = obs[0], obs[1]
    action = jnp.zeros((1,), dtype=jnp.float32)
    return action, state
```

`obs` is a float32 JAX array of shape `(2,)`, containing position and velocity.
`rng` is a fresh JAX PRNG key for this step. Use it if your policy is stochastic.
`state` is your own memory, initialised separately for each episode. It can be
an array, a tuple or dictionary of arrays, or an empty structure such as `()`.
All leaves must be finite numeric or boolean JAX arrays. Their shapes and dtypes
must stay fixed. Return the updated memory as the second item of the result.

`action` must be a floating-point JAX array of shape `(1,)`, with a finite value
in `[-1, 1]`. Negative values push left; positive values push right.

Both functions must be pure and compatible with JAX transformations. Use
`jax.numpy`, `jax.lax.cond`, or `jnp.where` for operations that depend on the
observation. A Python `if` on a traced observation will fail. Avoid side effects,
Python random generators, and file access. The evaluator batches episodes with
`jax.vmap` and compiles the rollout with `jax.jit` and `jax.lax.scan`.

For informal tests in `scratch.py`, construct the same public environment:

```python
import jax
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar
from final_solution import init_state, act

env = ContinuousMountainCar()
params = env.default_params
key = jax.random.PRNGKey(42)  # Choose your own exploratory seed.
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, params)
memory = init_state()
episode_return = 0.0
for _ in range(params.max_steps_in_episode):
    key, policy_key, step_key = jax.random.split(key, 3)
    action, memory = act(obs, memory, policy_key)
    obs, env_state, reward, terminated, truncated, info = env.step(
        step_key, env_state, action, params)
    episode_return += float(reward)
    if bool(terminated | truncated):
        break
print(episode_return)
```

This pinned Gymnax version returns six values from `step` and automatically
resets the environment at an episode boundary. Always stop counting rewards
at the first termination or truncation. Informal tests are exploratory;
register the revision and call `fitness-eval` for its official score.
````

## task/context/run_config.json

```json
{
  "time_seconds": 7200,
  "max_tokens": null,
  "max_evaluations": null,
  "evaluation_timeout_seconds": 600,
  "provider": "opencode"
}
```

## task/context/task_spec.md

```markdown
# Write a Continuous Mountain Car policy

## Objective

Write a policy that drives an underpowered car up the right-hand hill. Maximise the mean episode return.

## Solution interface

Submit one self-contained Python file named final_solution.py. Define init_state(), which returns the policy's initial memory, and act(obs, state, rng), which returns (action, next_state). The observation contains position and velocity. Return a floating-point JAX array of shape (1,) with a finite action between -1 and 1. Read policy_api.md for the complete interface and a working example.

## Fitness metric

An episode earns 100 when the car reaches position 0.45 with nonnegative velocity. Every step, including the successful step, costs 0.1 times the square of the action. Fitness is the average sum of these native rewards across 256 development episodes. Higher is better. There is no separate reward for finishing quickly. A policy that never acts scores zero but does not solve the task.

## Optimization direction

Maximise the fitness value. Higher scores are better.

## Constraints

- Keep all policy code and any learned parameters in final_solution.py. It must work without scratch.py, other workspace files, checkpoints, or network access.
- Use only the current observation, your policy memory, the supplied random key, and fixed constants when selecting actions. Do not inspect evaluator state, evaluation seeds, other files, or other processes.
- Write pure JAX-compatible functions that can run under jax.jit, jax.vmap, and jax.lax.scan. Memory must have the same structure, shapes, and dtypes on every step. Its numeric values must remain finite.
- Every action must have shape (1,), a floating-point dtype, and finite values in [-1, 1]. Invalid actions or incompatible memory make the evaluation fail; they do not receive a numeric fitness.
- You may design a controller analytically, search its parameters, or train a policy in scratch.py. Put the resulting controller and parameters into final_solution.py.
- Do not modify the environment or the fitness definition. The public environment implementation is available in the installed gymnax package.

## Evaluation

Each episode starts with position sampled uniformly from [-0.6, -0.4] and velocity zero. An episode ends on success or after 999 steps. Memory resets for each episode, and rewards after the first episode boundary are never counted. Official fitness-eval calls use the same development episodes and random keys for a given run seed, so revisions can be compared fairly. Separate run seeds use separate development batches. After research, the mountain_car wrapper can score the selected policies on a shared, separate batch of 1024 test episodes. Test results are for final reporting and must not be used to tune the policies.

## Available information

Read environment.md for the pinned environment, dynamics, and reward. Read policy_api.md for policy examples and informal testing. Use fitness-eval for official development scores and experiment-ledger to record each revision and the reasoning behind it.

## Expected result

Leave a complete working final_solution.py. The default runner selects the revision with the highest official development fitness. Follow the final selection instructions in the agent instructions if the run uses a different selection mode. Explain your experiments and remaining limitations in the ledger.
```

## task/ledger/ledger.jsonl

```text

```

## task/workspace/final_solution.py

```python
"""A valid starting policy. Replace this with your own controller."""

import jax.numpy as jnp


def init_state():
    return jnp.asarray(0, dtype=jnp.int32)


def act(obs, state, rng):
    return jnp.zeros((1,), dtype=jnp.float32), state
```

## task/workspace/scratch.py

```python
"""Optional informal tests. Record observations as exploratory."""
```
