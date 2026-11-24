"""Run the fitness function and submitted solution in one temporary container."""

import inspect
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

class EvaluationJob:
    def __init__(self, task, snapshot, directory, image_id, image_path, config, seed, timeout):
        self.task = task
        self.snapshot = snapshot
        self.directory = directory
        self.image_id = image_id
        self.image_path = image_path
        self.config = config
        self.seed = seed
        self.deadline = time.monotonic() + timeout
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.containers = set()
        self.directory.mkdir(parents=True)

    def cancel(self):
        from .sandbox import _docker
        self.cancelled.set()
        with self.lock:
            containers = list(self.containers)
        for name in containers:
            _docker('rm', '--force', name, check=False)

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if self.cancelled.is_set():
            raise RuntimeError('Evaluation was cancelled')
        if remaining <= 0:
            raise TimeoutError('Evaluation time limit exceeded')
        return remaining

    def run_container(self):
        from .sandbox import _container_options, _docker
        name = 'autoastra-eval-' + uuid.uuid4().hex
        runtime = Path(__file__).resolve().parent.parent
        command = [
            'create', '--name', name, '--label', 'autoastra.managed=true',
            *_container_options(self.config, self.image_path),
            '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1', '--log-opt', 'compress=false',
            '--workdir', '/tmp',
            '--mount', f'type=bind,src={runtime},dst=/runtime/autoastra,readonly,bind-recursive=disabled',
            '--mount', f'type=bind,src={self.directory},dst=/input,readonly,bind-recursive=disabled',
            '--mount', f'type=bind,src={self.directory}/code,dst=/evaluator/code,readonly,bind-recursive=disabled',
            '--mount', f'type=bind,src={self.snapshot},dst=/submission/final_solution.py,readonly,bind-recursive=disabled',
        ]
        command += ['--entrypoint', self.config.python_executable, self.image_id,
                    '-I', '-u', '/runtime/autoastra/sandbox/evaluation_worker.py']
        try:
            with self.lock:
                self.remaining()
                # Track the unique name before create, including a lost reply.
                self.containers.add(name)
                _docker(*command, timeout=min(30, self.remaining()))
            _docker('start', name, timeout=min(30, self.remaining()))
            result = _docker('wait', name, timeout=self.remaining())
            logs = _docker('logs', name)
            if result.stdout.strip() != '0':
                raise RuntimeError(f'Evaluator container exited with code {result.stdout.strip()}: {logs.stderr[-800:]}')
            lines = logs.stdout.splitlines()
            if not lines:
                raise RuntimeError('Evaluator returned no result')
            response = json.loads(lines[-1])
            if not response.get('ok'):
                raise RuntimeError(response.get('error', 'Execution failed'))
            self.remaining()
            return response['result']
        finally:
            _docker('rm', '--force', name, check=False)
            with self.lock:
                self.containers.discard(name)

    def prepare_evaluator(self):
        function = self.task.fitness_fn
        module = inspect.getmodule(function)
        if (not inspect.isfunction(function) or module is None or module.__name__ == '__main__'
                or '<' in function.__qualname__):
            raise ValueError('fitness_fn must be an importable function in a task module')
        source = Path(inspect.getfile(module)).resolve()
        parts = module.__name__.split('.')
        code = self.directory / 'code'
        code.mkdir()
        if len(parts) == 1 and source.name != '__init__.py':
            shutil.copyfile(source, code / source.name)
        else:
            root = source.parent if source.name == '__init__.py' else source
            for _ in parts:
                root = root.parent
            package = root / parts[0]
            shutil.copytree(package, code / parts[0], ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.venv'))
        (self.directory / 'request.json').write_text(json.dumps({
            'module': module.__name__, 'qualname': function.__qualname__, 'seed': self.seed,
        }))

    def run(self):
        try:
            self.prepare_evaluator()
            return self.run_container()
        finally:
            self.cancel()
