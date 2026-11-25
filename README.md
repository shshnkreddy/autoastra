<h1 align="center">AutoAstra 🚀</h1>

<p align="center"><strong>A coding agent writes and tests Python solutions against a fitness function you define.</strong></p>

<p align="center">
  <a href="#autoresearch-loop">Autoresearch loop</a> ·
  <a href="#set-up-with-an-agent">Agent setup</a> ·
  <a href="#run-the-demo">Quickstart</a> ·
  <a href="#bring-your-own-task">Custom tasks</a> ·
  <a href="tasks/affine/">Example task</a>
</p>

## Autoresearch loop

One Codex, Claude Code, or OpenCode session runs the experiment in Docker. The agent uses each score to decide what to try next.

```text
task_spec.md ──▶ agent ──edits──▶ final_solution.py
                   ▲                      │
                   │                      │ submit for eval
                   │                      ▼
                   └── score ────── fitness-eval
```

<details>
<summary><strong>Sandbox directory structure</strong></summary>

```text
/task/
├── AGENTS.md              # CLAUDE.md for Claude Code
├── bin/                   # Read-only commands
│   ├── fitness-eval
│   ├── experiment-ledger
│   └── lifetime
├── context/               # Read-only task files
│   ├── task_spec.md
│   ├── fitness_api.md
│   ├── run_config.json
│   └── ledger.py
├── ledger/                # Read-only experiment history
│   ├── ledger.jsonl
│   └── snapshots/
└── workspace/             # Editable
    ├── final_solution.py
    └── scratch.py
```

</details>

## Set up with an agent

Paste this into your coding agent from your project directory:

```text
I want to use AutoAstra to optimise my task. Its documentation is at
https://github.com/shshnkreddy/autoastra. Help me set it up for this project.

1. Read the README. Help me install AutoAstra and my chosen coding agent,
   configure an API key or subscription login/token, and set up a custom
   task if needed.
2. If I need a custom task, first read and follow the task-creation skill:
   https://github.com/shshnkreddy/autoastra/blob/main/.agents/skills/autoastra-create-task/SKILL.md
   Read its linked guides as needed for Docker, multiple seeds, and wrappers.
   a. Ask whether I want to create it from an existing codebase.
   b. Always generate and show me a preview of the initial sandbox.
   c. Ensure the required Docker image is available locally on the machine
      that will run the task; build or pull it if necessary.
```

## Setup

With Python 3.11 or newer, install AutoAstra:

```sh
pip install autoastra
```

This installs AutoAstra and its Python dependencies. Also install [Docker](https://docs.docker.com/engine/install/) and your chosen agent CLI on a Linux machine.

## Configure coding agent

**Install your chosen agent(s):**

| Agent | Install command |
| --- | --- |
| [Codex](https://learn.chatgpt.com/docs/codex/cli) | `curl -fsSL https://chatgpt.com/codex/install.sh \| sh` |
| [Claude Code](https://code.claude.com/docs/en/setup) | `curl -fsSL https://claude.ai/install.sh \| bash` |
| [OpenCode v1](https://opencode.ai/install) | `curl -fsSL https://opencode.ai/install \| bash` |

Make the installed commands available in your launch shell:

```sh
export PATH="$HOME/.local/bin:$HOME/.opencode/bin:$PATH"
```

**Codex subscription (default):** sign in with the ChatGPT account you want to use:

```sh
export AUTOASTRA_CODEX_HOME="$HOME/.codex-autoastra"
CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex -c 'cli_auth_credentials_store="file"' login
```

Use a different profile directory and sign in again for each subscription account.

**Other access options:** create an API key in your provider's console, or use a Claude subscription token. Set the variable in the same shell that launches AutoAstra.

| Access | Setup | Add to `autoastra run` |
| --- | --- | --- |
| OpenRouter via Codex | `export OPENROUTER_API_KEY="your-key"` | `--provider openrouter --model PROVIDER/MODEL` |
| Claude API | `export ANTHROPIC_API_KEY="your-key"` | `--provider anthropic` |
| [Claude subscription](https://code.claude.com/docs/en/authentication#generate-a-long-lived-token) | Run `claude setup-token`, sign in, then `export CLAUDE_CODE_OAUTH_TOKEN="the-returned-token"`. Leave `ANTHROPIC_API_KEY` unset to use the subscription. | `--provider anthropic` |
| OpenCode | Export the key for your model: `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENCODE_API_KEY`. For example, `export OPENROUTER_API_KEY="your-key"`. | `--provider opencode --model provider/model` |

OpenCode uses API keys here; AutoAstra does not import its saved interactive logins.

## Run the demo

The affine demo asks the agent to infer a linear rule from examples. Preview the task to see what the agent will receive:

```sh
autoastra preview tasks.affine:task --output previews/affine --print-to previews/affine.md
```

Once configured, run the demo (Codex is the default):

```sh
autoastra run tasks.affine:task --output runs/affine
autoastra result runs/affine
```

Read the experiment history in `runs/affine/task/ledger/ledger.md`. For budgets, parallel seeds, and GPUs, see `autoastra run --help`.

General internet access is disabled by default. The sandbox can reach only the
selected model provider's HTTPS domains through a host-side gateway; direct
connections, other websites, and package downloads are blocked. Use
`--allow-internet` to explicitly allow general access, or `--model-host` to set
exact provider domains. Evaluators always remain offline.

## Continuous Mountain Car

The agent writes a JAX policy in `final_solution.py` for Gymnax Continuous
Mountain Car. See the [task guide](tasks/continuous_mountain_car/README.md) for
building its CPU image, running one or multiple seeds, and scoring the final
policies. Defaults are two hours per seed with no token or evaluation-count limit.

Read the complete initial sandbox previews:
[Codex](previews/continuous-mountain-car/codex.md),
[Claude Code](previews/continuous-mountain-car/claude.md), or
[OpenCode](previews/continuous-mountain-car/opencode.md).

## Bring your own task

Define a `Task` in a Python module. Give it a **task specification** describing what to build and which metric to optimize, a **fitness function** that scores the submitted solution, and a **Docker image** with the task’s dependencies. See the [affine task definition](tasks/affine/__init__.py) and its [evaluator](tasks/affine/evaluate.py).

The image must contain Python and your environment's dependencies. You can build it on the execution host or use an image from Docker Hub or another registry. AutoAstra reuses the local image if present and pulls it if missing; it does not build images automatically.

<details>
<summary><strong>Example: the affine demo task object</strong></summary>

From [`tasks/affine/__init__.py`](tasks/affine/__init__.py):

```python
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
```

[`task_spec.json`](tasks/affine/task_spec.json) describes the problem, `evaluate` scores the submitted solution, and [`run_config.json`](tasks/affine/run_config.json) sets the run budgets. This demo only needs Python, so it uses `python:3.12-slim`.

</details>

If your module is `my_task.py` and the object is named `task`, preview it first to check the instructions and files the coding agent will see:

```sh
autoastra preview my_task:task --output previews/my-task --print-to previews/my-task.md
```

Then start the run:

```sh
autoastra run my_task:task --output runs/my-task
```

## Citation
