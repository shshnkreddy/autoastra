"""Schedule a population of independent autoresearch sessions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import threading

from .config import RunnerConfig
from .runner import run


@contextmanager
def _seed_config(config, seed, parallel):
    if config.agent != 'codex':
        yield config
        return

    home = config.profile_path
    if '{seed}' in str(home):
        yield replace(config, codex_home=Path(str(home).replace('{seed}', str(seed))))
        return
    if not parallel:
        yield config
        return

    # SQLite state, transcripts, and writable working files must be seed-local.
    with tempfile.TemporaryDirectory(prefix=f'autoastra-seed-{seed}-') as tmp:
        private = Path(tmp)
        for name in ('config.toml', 'auth.json'):
            if name == 'auth.json' and config.provider != 'codex':
                continue
            source = home / name
            if source.is_file():
                shutil.copyfile(source, private / name)
                (private / name).chmod(0o600)
            elif name == 'auth.json':
                raise ValueError(f'No Codex login in {home}')
        yield replace(config, codex_home=private)


def run_seeds(tasks_by_seed, destination, *, jobs=1, config=None, sandbox_config=None, run_config=None):
    """Yield (seed, result) as each session finishes. Map each seed to its task or paraphrase."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError('jobs must be a positive integer')
    if not tasks_by_seed or any(type(seed) is not int or seed < 0 for seed in tasks_by_seed):
        raise ValueError('Supply at least one nonnegative integer seed')
    config = config or RunnerConfig()
    workers = min(jobs, len(tasks_by_seed))
    directory = Path(destination)
    population = [{'seed': seed, 'task': task.task_spec['name'], 'directory': f'seed-{seed}'}
                  for seed, task in tasks_by_seed.items()]
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'population.json').write_text(json.dumps(population, indent=2) + '\n')
    cancel = threading.Event()

    def run_one(seed, task):
        target = directory / f'seed-{seed}'
        try:
            with _seed_config(config, seed, workers > 1) as seed_config:
                return run(task, target, config=seed_config, sandbox_config=sandbox_config,
                           run_config={**(run_config or {}), 'seed': seed}, cancel_event=cancel)
        except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
            # A setup failure can precede creation of the sandbox or its result.json.
            return {'sandbox': str(target), 'solution': None, 'status': 'failed',
                    'stop_reason': 'failed', 'error': f'Seed {seed}: {exc}'}

    if workers == 1:
        for seed, task in tasks_by_seed.items():
            result = run_one(seed, task)
            yield seed, result
            if result['stop_reason'] == 'interrupted':
                return
        return

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='autoastra-seed')
    try:
        futures = {pool.submit(run_one, seed, task): seed for seed, task in tasks_by_seed.items()}
        for future in as_completed(futures):
            yield futures[future], future.result()
    finally:
        cancel.set()
        pool.shutdown(wait=True, cancel_futures=True)
