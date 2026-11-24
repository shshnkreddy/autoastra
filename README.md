# AutoAstra

AutoAstra runs one Codex, Claude Code, or OpenCode session to improve a solution against a predefined
fitness function. The agent edits, evaluates, and records experiments in Docker.
The runner watches the budgets and saves the complete sandbox in its finished
state, with the selected revision in `workspace/final_solution.py`.

## Install AutoAstra

On the Linux execution host, use Python 3.11 or newer and a non-root account
that can run Docker. Create a virtual environment in your own project folder.
Install directly from the public GitHub repository.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install git+https://github.com/shshnkreddy/autoastra.git
autoastra --help
```

For a wheel, use `python -m pip install /path/to/autoastra-0.1.0-py3-none-any.whl`.
Both installs include the affine demo and the result report. To work on AutoAstra itself, run `python -m pip install -e .`
from the checkout. To build a wheel for someone else, run
`python -m pip wheel --no-deps . -w dist` there.

Try the affine demo from your project folder. The preview needs only Python.

```sh
autoastra preview tasks.affine:task --output previews/affine --print-to previews/affine.md
```

For a live run, install the agent CLI and configure one provider below, then
run `autoastra run tasks.affine:task --output runs/affine` for a Codex subscription.
To supply your own problem, see [Create a task](#create-a-task).

## Set up model access

Set up access on the execution host. You do not need credentials to inspect
a task or create a sandbox without an agent.

### Claude Code with Anthropic

On the Linux execution host, install the native Claude Code CLI or include
`claude` in the task image. Set `ANTHROPIC_API_KEY` in the launch shell for
API billing. For subscription access, run `claude setup-token` and export
the returned value as `CLAUDE_CODE_OAUTH_TOKEN`. AutoAstra uses the API key
when both are set. [Claude Code setup](https://code.claude.com/docs/en/setup) ·
[Authentication](https://code.claude.com/docs/en/authentication)

### OpenCode

Install the native Linux OpenCode CLI on the execution host or include
`opencode` in the task image. Use `--opencode-binary /path/to/opencode` to
select a native executable explicitly. Choose the backend with the model
name, for example `anthropic/claude-sonnet-4-6`. AutoAstra
passes the corresponding host API key into the development process when set:
`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`OPENCODE_API_KEY` (Zen and Go). [OpenCode CLI](https://opencode.ai/docs/cli/)

### OpenRouter

Create a key on the [OpenRouter API keys page](https://openrouter.ai/settings/keys),
then set it in the terminal where you will run AutoAstra.

```sh
export OPENROUTER_API_KEY="your-key"
```

Keep the key out of task files and Git. See the
[OpenRouter authentication guide](https://openrouter.ai/docs/api/reference/authentication)
for account and key setup.

### Codex with a ChatGPT subscription

Install the Codex CLI as described below, then sign in with the ChatGPT account
that has Codex access. This uses your account's Codex allowance and does not
require an OpenAI API key.

```sh
export AUTOASTRA_CODEX_HOME="$HOME/.local/share/autoastra/codex"
mkdir -p "$AUTOASTRA_CODEX_HOME"
CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex -c 'cli_auth_credentials_store="file"' login
CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex login status
```

On a server without a browser, add `--device-auth` to the login command and
follow the displayed link and code. Device login may need to be enabled in
your account or workspace settings. See the
[Codex sign-in guide](https://developers.openai.com/codex/auth/).

This dedicated directory is mounted into the development container so Codex
can use and refresh its login. It is not mounted into the evaluator. Keep it
out of Git and do not share it between concurrent runs. Use `--codex-home`
to select another dedicated directory. If you already have a signed-in Codex
profile, check it and pass that directory to the runner:

```sh
export AUTOASTRA_CODEX_HOME="/path/to/existing/codex-profile"
CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex login status
```

A cached login can still be expired or revoked. If a run reports a revoked
refresh token, renew the same profile with device login before retrying:

```sh
CODEX_HOME="$AUTOASTRA_CODEX_HOME" codex -c 'cli_auth_credentials_store="file"' login --device-auth
```

Complete the browser authorization with the intended account. Keep the failed
run directory for diagnostics and choose a new `--output` directory for the retry.

## Set up the execution host

Install Docker Engine and give your Linux user permission to run Docker. The
affine demo uses `python:3.12-slim`, which Docker pulls on first use.
[Docker installation](https://docs.docker.com/engine/install/) ·
[Linux user setup](https://docs.docker.com/engine/install/linux-postinstall/)

For the Codex and OpenRouter providers, install the
[Codex CLI](https://developers.openai.com/codex/cli/) on the machine where
you will run it.

```sh
curl -fsSL https://chatgpt.com/codex/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
codex --version
```

The runner mounts the native Linux `codex` binary found on PATH, together with
its packaged tool executables and resources. Keep the standalone installation
intact; copying only the main executable can break tool execution. Pass
`--codex-binary /path/to/codex` to select it explicitly. An npm launcher is
not a standalone binary. Alternatively, with no host `codex` on PATH, include
Codex in the task image. The binary must match the image's CPU architecture.
The app-server integration has been checked with Codex CLI 0.153.4, and a
complete Sol 6.1 demo run has been verified with Codex CLI 0.172.0.

For NVIDIA GPUs, also install compatible drivers and the
[NVIDIA Container Toolkit](https://docs.docker.com/engine/containers/gpu/).
Use an image with the task's CUDA dependencies and select GPUs with
`--gpus 1` or `--gpu-device 0`.

## Run an autoresearch loop

Run these examples from your project folder on the execution host.

The included task is **affine** (`tasks.affine:task`). It asks the agent to
infer a linear rule from examples and implement `predict(xs)` in
`final_solution.py`.

### Start the loop

With your Codex subscription:

```sh
autoastra run tasks.affine:task \
  --output runs/affine \
  --time-seconds 600 \
  --max-evaluations 20
```

To select GPT-6.1 Sol with an existing subscription profile:

```sh
autoastra run tasks.affine:task \
  --model gpt-6.1-sol \
  --codex-home "$AUTOASTRA_CODEX_HOME" \
  --output runs/affine-sol-6.1 \
  --time-seconds 600 \
  --max-evaluations 20
```

Or with your OpenRouter key:

```sh
autoastra run tasks.affine:task \
  --provider openrouter \
  --model PROVIDER/MODEL \
  --output runs/affine-openrouter \
  --time-seconds 600 \
  --max-evaluations 20
```

Replace `PROVIDER/MODEL` with a model that supports coding tools through
OpenRouter's Responses API. See [OpenRouter's Codex setup](https://openrouter.ai/blog/tutorials/codex-cli-openrouter/).
Use `--model` with subscription access too if you want to select a model.

With Claude Code, select `anthropic` and a Claude model or alias. The runner
uses the native `claude` on PATH, accepts an explicit `--claude-binary`, or
uses the CLI from the task image if the host has none.

```sh
autoastra run tasks.affine:task \
  --provider anthropic --model sonnet --output runs/affine-claude \
  --evaluation-timeout-seconds 120 --memory-mb 4096
```

To use OpenCode with your Anthropic API key:

```sh
autoastra run tasks.affine:task \
  --provider opencode --model anthropic/claude-sonnet-4-6 \
  --output runs/affine-opencode --evaluation-timeout-seconds 120 --memory-mb 4096
```

Use `--provider opencode` with `autoastra preview` to inspect the
same layout before launching. A run saves `opencode.events.jsonl` and
`opencode.stderr.log` beside the usual ledger, solution, and results.

Claude Code receives `task/CLAUDE.md`. Codex, OpenRouter through Codex, and OpenCode receive
`task/AGENTS.md`. To inspect the Claude version before a run:

```sh
autoastra preview tasks.affine:task --provider anthropic \
  --output previews/affine-claude --print-to previews/affine-claude.md
```

The affine examples allow up to ten minutes and twenty official evaluations.
The agent writes the solution, calls the evaluator, and records experiments
itself; you do not need to run the experiment commands manually. Keep the
terminal open until the run finishes. Use a new output directory each time.

Each run starts one conversation through Codex app-server, Claude Code print
mode, or OpenCode run. The agent handles every edit, evaluation, and ledger entry in that session.
Claude runs write `claude.events.jsonl` and `claude.stderr.log`. The session ends
when the agent finishes or reaches a budget. After the last evaluation, it has up to
ten seconds to record its conclusion, subject to the time and token limits.
The runner then removes the containers, restores the selected revision,
and saves the complete finished sandbox directory at `--output`.

### Follow the run and read the result

From another terminal on the same host, follow the experiment ledger:

```sh
tail -f runs/affine/.runtime/ledger.jsonl
```

When the run finishes, the runner prints the path to the finished sandbox.
For `--output runs/affine`, the saved directory looks like this:

```text
runs/affine/
├── task/
│   ├── AGENTS.md
│   ├── bin/
│   │   ├── fitness-eval
│   │   ├── experiment-ledger
│   │   └── lifetime
│   ├── context/
│   │   ├── task_spec.md
│   │   ├── fitness_api.md
│   │   ├── ledger.py
│   │   └── run_config.json
│   ├── ledger/
│   │   ├── ledger.md
│   │   ├── ledger.jsonl
│   │   └── snapshots/
│   └── workspace/
│       ├── final_solution.py
│       ├── scratch.py
│       └── ...                 # Other files created during the run
├── codex.events.jsonl
├── codex.stderr.log
├── result.json
└── .runtime/                   # Retained runtime records
```

The workspace keeps its files, with `final_solution.py` restored to the selected
officially evaluated revision. Context and tools are preserved. The finished
ledger and all registered source snapshots are ordinary files in `task/ledger/`,
so you can inspect or copy the task directory after the containers are gone.
The commands in `bin/` require a running sandbox service for operational use.

At shutdown, the runner also generates `task/ledger/ledger.md`: a readable
run summary followed by each attempt's hypothesis, change, official scores,
exploratory observations, and conclusions. Failed evaluations and attempts
without conclusions remain visible. `ledger.jsonl` remains the source record.
This report is generated from the records without another model call.

Read the saved summary, solution, and experiment history with:

```sh
cat runs/affine/result.json
cat runs/affine/task/workspace/final_solution.py
cat runs/affine/task/ledger/ledger.jsonl
```

For older completed runs, generate the readable ledger with:

```sh
python3 -m autoastra.sandbox.ledger_report runs/affine
```

To print the final solution and its recorded development fitness, and render
a plot of official attempts versus fitness:

```sh
autoastra result runs/affine
```

In Python, use `print_result("runs/affine")` from
`autoastra.wrappers.basic`. This reads the saved files without running another
evaluation. If no evaluation succeeded, it reports the fitness as unavailable.
The plot is saved to `runs/affine/fitness.png`. It includes only official
evaluations, with attempt IDs along the x-axis; repeated evaluations remain
separate points. Failed evaluations are marked without assigning them a score.
Use `plot_fitness("runs/affine")` from the same module to render only the plot.

Use `runs/affine-openrouter` instead for the OpenRouter example. `result.json`
records the stop reason, usage, best development score, and sandbox and solution
paths. Session events and diagnostics are retained alongside the task directory.

If no evaluation succeeds, the result has no selected solution; any remaining
workspace file is unvalidated. A failed session is reported as failed even if
an earlier successful solution was recovered. Held-out final scoring is not
part of this runner yet. Use a new output directory for every run.

### Adjust the run

The affine demo loads its ten-minute and twenty-evaluation limits from
`tasks/affine/run_config.json`. To run two independent seeds in parallel,
use `autoastra run tasks.affine:task --seeds 0 1 --jobs 2 --output runs/affine-seeds`.
Each seed gets its own output directory and budget.

To override task defaults for a run, pass `--run-config path/to/config.json`. Explicit command-line
budget flags override that file, which overrides the task defaults. Settings
not mentioned in an override are preserved, including unlimited budgets.
Previews and standalone sandboxes also load task defaults. Effective settings
are saved in `task/context/run_config.json` and the experiment ledger.

Add `--max-tokens 100000`, for example, to stop on the agent's reported cumulative token usage, including
cached input. An in-flight request can exceed this limit before its usage
arrives; this is not a hard billing cap. Time includes session startup, and
failed evaluations count towards the evaluation allowance.

The development container has network access for model requests. The evaluator
stays offline. Default research resources are one CPU and 2 GiB RAM; use
`--cpus`, `--memory-mb`, `--gpus`, or `--gpu-device` to change them.

### Optional: inspect a task or run a standalone sandbox

Create the task files and a single Markdown document showing their contents.

```sh
autoastra preview tasks.affine:task \
  --output runs/affine-preview --print-to runs/affine-preview.md
```

Create a sandbox for the same task and run a small check inside it.

```sh
autoastra sandbox tasks.affine:task \
  --output runs/affine-sandbox -- python -c "print('Sandbox ready')"
```

The container is removed after the command finishes. Its files remain in
`runs/affine-sandbox/`, with the solution file at
`task/workspace/final_solution.py`. The task instructions and tools are
read-only; the workspace is editable. Use a new output directory each time.

Inside a running sandbox, write `final_solution.py`, then use

```sh
experiment-ledger register final_solution.py \
  --title "First idea" --hypothesis "A linear rule fits the examples" \
  --planned-change "Implement that rule" --success-criterion "Zero development error"
fitness-eval final_solution.py
experiment-ledger history
lifetime
```

All commands return JSON. The ledger also supports `observe`, `conclude`,
`inspect`, and `restore`; use `--help` for their arguments. Records and source
snapshots persist in `.runtime/` on the host and survive sandbox shutdown.
Inside the development container, they are readable at `/task/ledger/ledger.jsonl`
and `/task/ledger/snapshots/`. The ledger source is at `/task/context/ledger.py`.
All three are read-only; use the commands to append records. Each ledger event
includes the previous event's hash.

For a standalone sandbox, pass `--run-config config.json` to set `time_seconds`, `max_evaluations`,
`evaluation_timeout_seconds`, and `seed`. Failed evaluations count towards
the allowance. Token usage stays unknown until a model runner reports it
through `sandbox.record_token_usage(count)`; `max_tokens` can then limit
further evaluations. Time and budget limits stop new evaluations; the ledger
remains available for inspection and conclusions.

## Create a task

Save the following as `my_task.py` in your project folder. The spec tells the
agent what to build. The evaluator loads the submitted file and returns one
finite score. The Docker image supplies Python and any task dependencies.

```python
import runpy

from autoastra import Task


def evaluate(solution_path, *, split, seed):
    predict = runpy.run_path(str(solution_path))["predict"]
    inputs = [-2, 4, 8]
    expected = [2 * x + 1 for x in inputs]
    actual = predict(inputs)
    return sum(abs(a - b) for a, b in zip(actual, expected, strict=True)) / len(inputs)


task = Task(
    task_spec={
        "name": "linear_rule",
        "description": "Infer the affine rule from (0, 1), (1, 3), and (3, 7).",
        "solution_interface": "Implement predict(xs) in final_solution.py. Return one number per input.",
        "metric": "mean_absolute_error",
        "direction": "minimize",
    },
    fitness_fn=evaluate,
    image="python:3.12-slim",
    run_config={"time_seconds": 600, "max_evaluations": 20},
)
```

Use `module:object` to select the task. Here `my_task` is the filename without
`.py`, and `task` is the Python object defined above. You can also use an
installed Python package such as `my_lab.problems:task`. The name `tasks` is
used by the bundled examples. Put helper modules and data for your evaluator
in the same Python package, and read data paths relative to `__file__`.

```sh
autoastra preview my_task:task --output previews/linear
autoastra run my_task:task --output runs/linear
autoastra result runs/linear
```

To launch the same run from Python with your configured Codex subscription:

```python
from autoastra.autoresearch import run
from my_task import task

result = run(task, "runs/linear-python")
print(result["solution"])
```

The optional `run_config` sets default budgets. Command-line flags override
them. Keep credentials in the execution host's environment or agent login
profile. For additional task dependencies, supply a Docker image containing
them and both `python` and `python3`. Import evaluator-only libraries inside
the fitness function so loading the task on the host only needs AutoAstra.

You can build an image on the execution host with
`docker build -t my-task:v1 .` and use `image="my-task:v1"`. Publishing it
is optional. To share an image, upload it to Docker Hub or another registry
and use its full name in the task.

AutoAstra uses the host's existing image when available and pulls it when
missing. For private images, first run `docker login` for that registry on
the execution host. Pulls happen before the isolated sandbox starts.
Use a digest reference such as `myusername/my-task@sha256:...` to select an
exact published version. See [Docker's image guide](https://docs.docker.com/reference/cli/docker/image/pull/).

CPU, memory, and GPU settings remain run options. The image comes from the task.

Each run uses one persistent development container and one temporary evaluation
container while scoring. The latter receives the frozen `final_solution.py`,
runs `fitness_fn`, returns a finite fitness value, and is removed. Both use
the task's image. The evaluation container has no access to the development
workspace or ledger. For now, the evaluator and solution share a Python
process, and we trust submissions not to interfere with scoring.
