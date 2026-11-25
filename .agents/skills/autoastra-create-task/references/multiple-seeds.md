# Multiple seeds and experiment scheduling

Use this guide when the user needs independent solutions, prompt variants, or several environments. Use AutoAstra's existing seed runner before implementing a scheduler.

## One task, independent sessions

```sh
autoastra run my_tasks.example:task --seeds 0 1 2 3 4 5 --jobs 6 --output runs/example
```

Each seed gets a separate agent session, sandbox, full run budget, and `seed-N/` output directory. The population directory records the seed-to-task mapping. Use a new output directory for a new population; it is not an automatic resume command.

`--jobs` controls concurrent sessions within that invocation and defaults to one. State both total run count and concurrency. Six seeds over six environments means 36 sessions. If environments must run concurrently, the experiment wrapper must launch those invocations concurrently too, with an explicit overall resource limit; six jobs inside a sequential environment loop does not do that.

The seed controls task evaluation randomness. It does not guarantee deterministic model sampling or identical solutions across reruns. Keep independent sessions separate; do not share their writable workspaces or writable agent state. The built-in runner isolates parallel Codex state.

## Prompt variants

For six equivalent task descriptions, export `p0` through `p5` from the same task module:

```sh
autoastra run my_tasks.example:p0 --seeds 0 1 2 3 4 5 --paired-paraphrases --jobs 6 --output runs/example-variants
```

This option requires exactly six seeds, paired in the order supplied. Without it, every seed receives the same task object. Keep environment, constraints, fitness definition, and solution interface equivalent across paraphrases; change only the intended wording. Show previews of the variants when authoring them so the user can inspect those differences.

For arbitrary populations, the Python API accepts `run_seeds(tasks_by_seed, destination, jobs=..., config=..., sandbox_config=..., run_config=...)`. It yields `(seed, result)` as runs finish. Inspect the [current implementation](../../../../autoastra/autoresearch/batch.py) before writing a wrapper.

## Budgets and completion

Task defaults are overridden by `--run-config` JSON, then explicit budget flags. Each seed receives its own effective budget. Use `null` in JSON to remove a token or evaluation limit inherited from the task. Choose selection explicitly when comparing different experiment designs.

Verify actual startup for every requested seed; a launcher PID alone is not evidence of a working model session. Setup failures can occur before `result.json` is written, so preserve launcher output and record missing results as failures or incomplete work. Never treat absent results as zero fitness.

Before population scoring, confirm that each required run has an officially scored selected solution. Record excluded or failed seeds explicitly if partial-population scoring is intentionally requested. Keep development seeds separate from held-out scoring seeds and record episode counts, model/provider, prompt variants, image identity, and selection rule with the results.
