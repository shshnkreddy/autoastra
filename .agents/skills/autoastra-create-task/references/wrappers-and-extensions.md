# Wrappers, held-out scoring, and extensions

Use this guide when the user's objective requires more than the scalar development fitness returned by a task.

## Choose where the change belongs

| Requirement | Place it in |
| --- | --- |
| A new environment, scalar metric, solution interface implemented in one Python file, or public context | Task package and Docker image |
| Multiple environments, population evaluation, several report metrics, plots, or experiment aggregation | A project-owned wrapper, for example `wrappers/ixp.py` |
| Multiple submitted artifacts, a different evaluator transport, stronger isolation, or unsupported runner behavior | An explicit AutoAstra extension with focused integration validation |

Keep the fitness function's scalar contract. If a problem needs multiple objectives, agree on a development scalar or implement an explicit interface extension; do not return a dictionary where AutoAstra expects a number. Put extra held-out metrics in the wrapper's report.

A wrapper can call `autoastra run` or the Python `run`/`run_seeds` APIs, collect finished outputs, and evaluate the selected solutions. It should preserve the standard sandbox and ledger instead of replacing them with an unrelated experiment history. The [basic result wrapper](../../../../autoastra/wrappers/basic.py) is a small reporting example. Cross-play is an optional workflow implemented by a task project, not a built-in guarantee for every AutoAstra task.

## Example: IXP after self-play development

During research, each agent sees only its own allowed development feedback. After independent runs finish:

1. Read each run's `result.json` and selected `task/workspace/final_solution.py`. Validate that its source hash matches the selected revision. Do not silently substitute the best development revision if the experiment selected a different one.
2. Check compatibility: environment configuration, solution interface, runtime image, and task variant. Keep model, development seed, prompt variant, and selection metadata attached to each policy.
3. Freeze the population. Evaluate every ordered pair on a documented held-out batch, with the row policy in seat 0 and the column policy in seat 1. For six policies, this is 36 cells, including six self-play cells. Use the same evaluation conditions across cells.
4. Save the matrix, labels, episode count, held-out seed, source hashes, image identity, and evaluator/configuration identity. Cache completed cells using this full identity; reject incompatible inputs instead of reusing stale scores. Record failures without turning them into zero returns.
5. Produce a labelled CSV, JSON summary, and heatmap. Keep these reports separate from development fitness and the original ledgers.

For a matrix `M` with `n` policies:

```text
SP  = sum(M[i, i]) / n
IXP = sum(M[i, j] for i != j) / (n * (n - 1))
```

IXP is undefined for a single policy; report `null`. Do not assume symmetry: swapping seats may change the result. These formulas apply to within-population cross-play. Comparisons between two populations need an explicitly defined pairing and aggregation rule.

Keep held-out results out of the research loop when measuring zero-shot coordination. If the experiment intentionally feeds cross-play back into development, describe that changed protocol rather than presenting it as independent held-out evaluation.

## Validate and deliver a wrapper

Check matrix orientation and aggregation on a tiny known case, verify selected-source loading, and check that resuming reuses only matching completed cells. Ensure failures and incomplete populations remain visible. Run expensive scoring only within the user's requested scope.

If the workflow needs changes to AutoAstra itself, explain the existing limitation, implement the smallest reusable extension, update the relevant API documentation and preview, and validate its actual integration behavior. Do not hide task-specific code in generated sandbox commands or depend on undocumented internals without making that dependency explicit.
