# AutoAstra

AutoAstra runs a coding agent in Docker to improve a Python solution against a fitness function you define. The agent edits and evaluates its solution in a single session. Each run saves the selected solution and an experiment ledger showing what the agent tried and how it scored.

## Quickstart

Install AutoAstra in a Python 3.11+ environment. To start a live run, use a Linux host with Docker and your chosen agent CLI.

```sh
python -m pip install git+https://github.com/shshnkreddy/autoastra.git
```

The bundled affine demo asks the agent to infer a linear rule from examples. You can preview the task before configuring model access.

```sh
autoastra preview tasks.affine:task --output previews/affine --print-to previews/affine.md
```

Choose one way to connect your agent:

| Provider | Setup on the execution host |
| --- | --- |
| Codex (default) | Set `AUTOASTRA_CODEX_HOME` to a dedicated profile and sign in with `CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex -c 'cli_auth_credentials_store="file"' login`. |
| OpenRouter via Codex | Export `OPENROUTER_API_KEY` and pass `--provider openrouter --model PROVIDER/MODEL`. |
| Claude Code | Export `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` and pass `--provider anthropic`. |
| OpenCode | Export your backend’s API key and pass `--provider opencode --model provider/model`. |

Run the demo, then inspect the result:

```sh
autoastra run tasks.affine:task --output runs/affine
autoastra result runs/affine
```

Open `runs/affine/task/workspace/final_solution.py` for the selected solution and `runs/affine/task/ledger/ledger.md` for the experiment history. Use `autoastra run --help` for budgets, parallel seeds, and GPU options.

## Create a task

Define a `Task` in your own Python module with three inputs:

- **Task specification** — what the agent should build, the solution interface, and which metric to minimize or maximize.
- **Fitness function** — a Python function that loads the submitted solution and returns a numeric score.
- **Docker image** — the environment containing Python and your task’s dependencies. Use a locally built image or one from a registry.

The [affine task](tasks/affine/) is a complete example. If you define an object named `task` in `my_task.py`, run it from that directory with:

```sh
autoastra run my_task:task --output runs/my-task
```

## Citation
