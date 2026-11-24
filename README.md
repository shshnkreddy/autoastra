<h1 align="center">AutoAstra 🚀</h1>

<p align="center"><strong>A coding agent writes and tests Python solutions against a fitness function you define.</strong></p>

<p align="center">
  <a href="#set-up-with-an-agent">Agent setup</a> ·
  <a href="#run-the-demo">Quickstart</a> ·
  <a href="#bring-your-own-task">Custom tasks</a> ·
  <a href="tasks/affine/">Example task</a>
</p>

AutoAstra runs one Codex, Claude Code, or OpenCode session in Docker. The agent edits its solution, evaluates it, and uses the score to decide what to try next. Each run saves the selected `final_solution.py` and an experiment ledger linking each attempt to its source and score.

## Set up with an agent

Paste this into your coding agent from your project directory:

```text
Help me set up AutoAstra using https://github.com/shshnkreddy/autoastra.
Install it in my Python environment and show me the command to run the affine demo on my Docker host.
```

## Run the demo

The affine demo asks the agent to infer a linear rule from examples. With Python 3.11 or newer, install AutoAstra from PyPI and preview the task before signing in.

```sh
pip install autoastra
autoastra preview tasks.affine:task --output previews/affine --print-to previews/affine.md
```

For a live run, use a Linux host with Docker and your chosen agent CLI. With a configured Codex subscription:

```sh
autoastra run tasks.affine:task --output runs/affine
autoastra result runs/affine
```

<details>
<summary><strong>Configure model access</strong></summary>

| Provider | Setup on the execution host |
| --- | --- |
| Codex (default) | Set `AUTOASTRA_CODEX_HOME` to a dedicated profile directory. Sign in with `CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex -c 'cli_auth_credentials_store="file"' login`. |
| OpenRouter via Codex | Export `OPENROUTER_API_KEY` and pass `--provider openrouter --model PROVIDER/MODEL`. |
| Claude Code | Export `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` and pass `--provider anthropic`. |
| OpenCode | Export your backend’s API key and pass `--provider opencode --model provider/model`. |

</details>

Read the experiment history in `runs/affine/task/ledger/ledger.md`. For budgets, parallel seeds, and GPUs, see `autoastra run --help`.

## Bring your own task

Define a `Task` in a Python module. Give it a **task specification** describing what to build and which metric to optimize, a **fitness function** that scores the submitted solution, and a **Docker image** with the task’s dependencies. See the [affine task definition](tasks/affine/__init__.py) and its [evaluator](tasks/affine/evaluate.py).

If your module is `my_task.py` and the object is named `task`, run:

```sh
autoastra run my_task:task --output runs/my-task
```

## Citation
