---
name: autoastra-create-task
description: Create or adapt AutoAstra tasks from a problem or existing codebase, including fitness evaluators, Docker environments, sandbox previews, multiple-seed experiments, and custom scoring wrappers.
---

# Create an AutoAstra task

Turn the user's problem into an importable `Task`, a working evaluation environment, and a preview of what the research agent will see. Extend the experiment workflow when the problem needs population scoring or other analysis after research.

## Understand the task

- Ask whether to start from an existing codebase unless the user has already specified one. Inspect its solution interface, evaluator, data, and dependencies before introducing new conventions.
- Establish the objective, fitness direction, constraints, expected solution, and available development feedback. Distinguish the feedback the agent can use to explore, train, or select a solution from the metric that measures its true fitness. For example, an agent might use training loss to update a model and validation loss to choose a checkpoint, while held-out test loss measures the selected checkpoint’s final performance.
- Establish the execution machine, resource requirements, budgets, and desired number of independent runs. Respect the user's choice of machine; do not assume execution should happen on the machine where files are edited.
- Use the repository's [README](../../../README.md) for AutoAstra installation, agent installation, and authentication. Inspect the installed version and CLI help before relying on an option; these references describe the current repository interfaces.

## Define and prepare

1. Read [Task definition](references/task-definition.md). Create the specification, importable evaluator, task object, run configuration, and any starter or public context files. Keep task-specific code in the user's project.
2. Write specifications and all agent-facing prompts in clear, human-legible language. Explicitly state the objective, how success is measured, the constraints, and what the solution must deliver: its interface, inputs, outputs, and final artifact. Explain unfamiliar terms and avoid unstated assumptions. Check that generated instructions agree with the actual evaluator and selection rule.
3. Read [Docker environment](references/docker-environment.md). Build or pull the required image on the execution machine, and verify the dependencies and data paths needed for evaluation. Do not describe a preview alone as a working sandbox.
4. If multiple runs are needed, read [Multiple seeds](references/multiple-seeds.md). Make seed count, concurrency, per-run budgets, and prompt variants explicit.
5. If one development fitness score is insufficient, read [Wrappers and extensions](references/wrappers-and-extensions.md). Decide whether the requirement belongs in the task, an experiment wrapper, or AutoAstra itself.

## Preview and validate

Always generate and show the user the initial sandbox preview. Use the same provider and run settings intended for execution, including the selection rule. Present the generated Markdown, not just a command the user could run. After material changes to the task or instructions, regenerate it in a new output directory.

```sh
autoastra preview my_task:task --output previews/my-task --print-to previews/my-task.md
```

Check that the displayed objective, constraints, fitness API, solution interface, and starter agree. Confirm that public context is readable, the workspace is editable, and private answers have not been included in agent-visible files or images.

On the execution machine, validate a known candidate through the evaluator in the intended image. Check one meaningful invalid submission and the expected score or ordering for a known case. Run a small end-to-end sandbox check when needed to verify packaging, snapshots, or dependencies. Keep validation separate from the user's research output and explain anything not yet tested. Do not launch an unrequested research population as a validation step.

## Deliver

Provide the task files, image reference, rendered preview, validation outcome, and commands for the requested single or multiple-seed experiment. Include custom scoring and report commands when applicable. Preserve the complete finished sandbox, ledger, and source snapshots as experiment outputs. If the user also requested execution, launch within that scope and verify that sessions actually start.
