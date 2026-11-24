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
Help me set up AutoAstra using https://github.com/shshnkreddy/autoastra.
Install it in my Python environment and show me the command to run the affine demo on my Docker host.
```

## Run the demo

The affine demo asks the agent to infer a linear rule from examples. With Python 3.11 or newer, install AutoAstra from PyPI and preview the task before signing in.

```sh
pip install autoastra
autoastra preview tasks.affine:task --output previews/affine --print-to previews/affine.md
```

`pip install autoastra` installs AutoAstra and its Python dependencies. For live runs, separately install [Docker](https://docs.docker.com/engine/install/) and one of the agent CLIs below on your Linux execution host.

<details>
<summary><strong>Configure model access</strong></summary>

**Install your chosen agent** (you only need one):

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

</details>

Once configured, run the demo (Codex is the default):

```sh
autoastra run tasks.affine:task --output runs/affine
autoastra result runs/affine
```

Read the experiment history in `runs/affine/task/ledger/ledger.md`. For budgets, parallel seeds, and GPUs, see `autoastra run --help`.

## Bring your own task

Define a `Task` in a Python module. Give it a **task specification** describing what to build and which metric to optimize, a **fitness function** that scores the submitted solution, and a **Docker image** with the task’s dependencies. See the [affine task definition](tasks/affine/__init__.py) and its [evaluator](tasks/affine/evaluate.py).

If your module is `my_task.py` and the object is named `task`, preview it first to check the instructions and files the coding agent will see:

```sh
autoastra preview my_task:task --output previews/my-task --print-to previews/my-task.md
```

Then start the run:

```sh
autoastra run my_task:task --output runs/my-task
```

## Citation
