"""Score finished Mountain Car policies on a shared test batch, outside research."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile
import uuid

from autoastra.sandbox import SandboxConfig
from autoastra.sandbox.sandbox import _container_options, _docker


def _selected(directory):
    directory = Path(directory).resolve()
    result = json.loads((directory / "result.json").read_text())
    if result.get("task") != "continuous_mountain_car":
        raise ValueError(f"Not a Mountain Car run: {directory}")
    selected = result.get("selected")
    if not selected or selected.get("status") != "ok":
        raise ValueError(f"No successfully evaluated selection: {directory}")
    source = (directory / "task/workspace/final_solution.py").read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != selected.get("source_sha256"):
        raise ValueError(f"Final solution differs from the selected revision: {directory}")
    image_id = result.get("image_id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError(f"Missing immutable image ID: {directory}")
    return directory, result, source, digest, image_id


def score_runs(directories, output, *, seed=2000003, episodes=1024, timeout=600):
    """Evaluate each selection in its recorded image; retain source and score provenance."""
    # Importing the Task package does not import JAX on the execution host.
    from tasks import continuous_mountain_car
    task_dir = Path(continuous_mountain_car.__file__).parent
    settings = json.loads((task_dir / "environment.json").read_text())
    if type(seed) is not int or not 0 <= seed < 2**32 or type(episodes) is not int or episodes <= 0:
        raise ValueError("Use a uint32 seed and a positive episode count")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    inputs = [_selected(directory) for directory in directories]
    if not inputs or len({item[0] for item in inputs}) != len(inputs):
        raise ValueError("Provide one or more distinct run directories")
    for directory, result, *_ in inputs:
        evaluation_id = result["selected"].get("evaluation_id", "")
        if not re.fullmatch(r"E[0-9]+", evaluation_id):
            raise ValueError(f"Missing selected evaluation ID: {directory}")
        # Use the private evaluator snapshot, never expose configuration merely
        # to make it available to post-run reporting.
        recorded_path = (directory / ".runtime/evaluations" / evaluation_id /
                         "code/tasks/continuous_mountain_car/environment.json")
        if not recorded_path.is_file():
            # Compatibility with older exports that retained only public context.
            recorded_path = directory / "task/context/environment.json"
        recorded = json.loads(recorded_path.read_text())
        if recorded != settings:
            raise ValueError(f"Environment or scoring protocol differs from this wrapper: {directory}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = SandboxConfig(cpus=2, memory_mb=4096, pids_limit=256, tmp_mb=256)
    records = []
    evaluator = (task_dir / "rollouts.py").read_bytes()
    environment = (task_dir / "environment.json").read_bytes()
    evaluator_hash = hashlib.sha256(evaluator).hexdigest()
    (output / "rollouts.py").write_bytes(evaluator)
    (output / "environment.json").write_bytes(environment)
    for index, (directory, result, source, digest, image_id) in enumerate(inputs):
        name = "autoastra-mountain-test-" + uuid.uuid4().hex
        solution = output / f"{index:02d}-final_solution.py"
        solution.write_bytes(source)
        # Freeze the evaluator too, so the report can be reproduced after code changes.
        with tempfile.TemporaryDirectory(prefix="autoastra-mountain-test-") as temporary:
            runtime = Path(temporary)
            (runtime / "rollouts.py").write_bytes(evaluator)
            (runtime / "environment.json").write_bytes(environment)
            try:
                response = _docker(
                    "run", "--name", name, "--label", "autoastra.managed=true",
                    *_container_options(config, "/usr/local/bin:/usr/bin:/bin"),
                    "--workdir", "/tmp",
                    "--mount", f"type=bind,src={runtime},dst=/evaluator,readonly",
                    "--mount", f"type=bind,src={solution},dst=/submission/final_solution.py,readonly",
                    "--entrypoint", "python", image_id, "-I", "/evaluator/rollouts.py",
                    "/submission/final_solution.py", "--split", "test", "--seed", str(seed),
                    "--episodes", str(episodes), timeout=timeout)
                scores = json.loads(response.stdout.splitlines()[-1])
            finally:
                _docker("rm", "--force", name, check=False)
        record = {
            "run": str(directory), "run_seed": result["seed"],
            "attempt_id": result["selected"]["attempt_id"],
            "development_fitness": result["selected"]["fitness"],
            "source_sha256": digest, "image_id": image_id,
            "evaluator_sha256": evaluator_hash, "environment": settings,
            **scores,
        }
        records.append(record)
        (output / f"{index:02d}-test.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    fields = ["run", "run_seed", "attempt_id", "development_fitness", "mean_return",
              "return_standard_error", "success_rate", "mean_episode_length", "mean_control_effort",
              "seed", "episodes", "source_sha256", "image_id", "evaluator_sha256"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--seed", type=int, default=2000003, help="Shared final-test seed")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    records = score_runs(args.runs, args.output, seed=args.seed, episodes=args.episodes, timeout=args.timeout)
    for record in records:
        print(f"{record['run']}: test return {record['mean_return']:.4f}, "
              f"success {record['success_rate']:.1%}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
