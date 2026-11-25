# Continuous Mountain Car

The coding agent writes a JAX policy in `final_solution.py` to drive an
underpowered car up a hill. Fitness is mean native episode return: reach the
goal while using little control effort. The task uses 256 development episodes
and a separate final report with 1024 test episodes.

Run these commands from your AutoAstra checkout on the Linux machine that will
run Docker. Set up your chosen agent and credentials using the main README.

## Build the environment

```sh
docker build -t autoastra-mountain-car:cpu tasks/continuous_mountain_car
```

This CPU image pins Gymnax and its main dependencies. The complete installed
dependency list is saved at `/opt/mountain-car/installed-requirements.txt` in
the image. No GPU is needed. Its Python startup hook limits JAX's thread pool
using the CPU count that AutoAstra supplies, including for informal tests.

## Inspect what the agent sees

```sh
autoastra preview tasks.continuous_mountain_car:task \
  --provider codex --output previews/mountain-car-codex \
  --print-to previews/mountain-car-codex.md
```

Use `--provider anthropic` or `--provider opencode` with a different output
path to inspect those agents' instructions. Preview generation needs neither
Docker nor JAX on the host.

## Run research

```sh
autoastra run tasks.continuous_mountain_car:task \
  --provider codex --model gpt-6.1-sol --seed 0 \
  --cpus 2 --memory-mb 4096 --output runs/mountain-car
```

The defaults are two hours, no token limit, and no evaluation-count limit.
Each evaluation has a ten-minute timeout. To run six independent research
sessions, replace `--seed 0` with `--seeds 0 1 2 3 4 5 --jobs 6` and use a new
output directory. Each session gets its own sandbox, development batch, ledger,
and selected policy. Resource limits apply separately to each container.

## Report final performance

After research and policy selection are finished, run:

```sh
python -m autoastra.wrappers.mountain_car runs/mountain-car \
  --output reports/mountain-car-test
autoastra result runs/mountain-car
```

For multiple seeds, pass `runs/mountain-car/seed-*` to the test wrapper. All
selected policies receive the same separate test batch. Use test results for
final reporting, not policy tuning. The JSON and CSV report includes return,
success rate, episode length, control effort, source hashes, evaluator hash,
and immutable image IDs. It also saves the selected source and test evaluator.
The recorded images must still be available on that Docker host.

`autoastra result` prints the selected source and development fitness and plots
the official attempts. It does not perform final testing.

## Validate the task

```sh
docker run --rm --network none --cpus 2 --memory 4g --pids-limit 256 \
  -e PYTHONPATH=/code -v "$PWD:/code:ro" -w /code \
  autoastra-mountain-car:cpu \
  python -m unittest discover -s tests -p 'test_continuous_mountain_car.py' -v
```

The tests check zero-action and momentum controllers, reproducible seeds,
separate evaluation splits, episode boundaries, invalid policy outputs, and
final-source provenance.

To also check official evaluation, the ledger, lifetime, and the final-test
wrapper through real containers, run this on the Docker host:

```sh
AUTOASTRA_TEST_MOUNTAIN_CAR=1 python -m unittest discover -s tests \
  -p 'test_continuous_mountain_car_sandbox.py' -v
```
