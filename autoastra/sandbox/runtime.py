"""Host-owned experiment records, evaluation dispatch, and lifetime accounting."""

import base64
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading
import time

from .evaluation import EvaluationJob
from .protocol import Server


def validate_run_config(value):
    if value is not None and not isinstance(value, Mapping):
        raise TypeError('run_config must be a mapping')
    config = dict(value or {})
    for name in ('max_evaluations', 'max_tokens'):
        if config.get(name) is not None and (type(config[name]) is not int or config[name] < 0):
            raise ValueError(f'{name} must be a nonnegative integer or null for no limit')
    if 'seed' in config and (type(config['seed']) is not int or config['seed'] < 0):
        raise ValueError('seed must be a nonnegative integer')
    for name in ('time_seconds', 'evaluation_timeout_seconds'):
        if name in config and (type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] <= 0):
            raise ValueError(f'{name} must be a positive finite number')
    if config.get('selection', 'best-fitness') not in ('best-fitness', 'model'):
        raise ValueError('Unknown final selection mode')
    return config


class Runtime:
    def __init__(self, task, directory, image_id, image_path, config, run_config):
        self.task = task
        self.directory = directory / '.runtime'
        self.directory.mkdir(mode=0o700)
        self.snapshots = self.directory / 'snapshots'
        self.snapshots.mkdir()
        self.ledger = self.directory / 'ledger.jsonl'
        self.ledger.touch(mode=0o600)
        self.image_id = image_id
        self.image_path = image_path
        self.config = config
        self.run_config = validate_run_config(run_config)
        self.started = time.monotonic()
        self.finished = None
        self.closed = False
        self.lock = threading.RLock()
        self.evaluation_lock = threading.Lock()
        self.evaluation_done = threading.Event()
        self.evaluation_done.set()
        self.job = None
        self.events = []
        self.selected = None
        self.attempts = {}
        self.evaluations = []
        self.tokens_used = None
        try:
            source = inspect.getsourcefile(task.fitness_fn)
        except TypeError:
            source = None
        self.append('run_started', {
            'task_spec': task.task_spec, 'image': task.image, 'image_id': image_id,
            'sandbox_config': asdict(config), 'run_config': self.run_config,
            'fitness_module': getattr(task.fitness_fn, '__module__', None),
            'fitness_function': getattr(task.fitness_fn, '__qualname__', None),
            'fitness_module_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
        })
        self.server = Server(self.directory / 'ipc/agent.sock', self.dispatch)

    def append(self, kind, payload):
        event = {
            'sequence': len(self.events), 'type': kind,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': round(time.monotonic() - self.started, 6),
            'previous_hash': self.events[-1]['event_hash'] if self.events else None,
            **payload,
        }
        canonical = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False)
        event['event_hash'] = hashlib.sha256(canonical.encode()).hexdigest()
        with self.ledger.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        self.events.append(event)
        return event

    def lifetime(self):
        elapsed = (self.finished or time.monotonic()) - self.started
        time_limit = self.run_config.get('time_seconds')
        max_evaluations = self.run_config.get('max_evaluations')
        max_tokens = self.run_config.get('max_tokens')
        return {
            'elapsed_seconds': round(elapsed, 3), 'time_seconds': time_limit,
            'remaining_seconds': max(0, round(time_limit - elapsed, 3)) if time_limit is not None else None,
            'evaluations_used': len(self.evaluations), 'max_evaluations': max_evaluations,
            'evaluations_remaining': max(0, max_evaluations - len(self.evaluations)) if max_evaluations is not None else None,
            'tokens_used': self.tokens_used, 'max_tokens': max_tokens,
            'tokens_remaining': max(0, max_tokens - self.tokens_used) if max_tokens is not None and self.tokens_used is not None else None,
            'token_tracking': 'reported by runner' if self.tokens_used is not None else 'not connected',
            'closed': self.closed,
        }

    def record_tokens(self, count):
        """Trusted runner hook. This operation is deliberately absent from RPC."""
        if type(count) is not int or count < 0:
            raise ValueError('Token count must be a nonnegative integer')
        with self.lock:
            if self.closed:
                raise RuntimeError('Runtime is closed')
            self.tokens_used = (self.tokens_used or 0) + count
            self.append('token_usage', {'tokens_added': count, 'tokens_used': self.tokens_used})

    @staticmethod
    def source(payload):
        value = payload.get('source')
        if not isinstance(value, str):
            raise ValueError('Provide source bytes')
        source = base64.b64decode(value, validate=True)
        if len(source) > 1024 * 1024:
            raise ValueError('Source must be at most 1 MiB')
        source.decode('utf-8')
        return source, hashlib.sha256(source).hexdigest()

    @staticmethod
    def text(payload, key):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 16000:
            raise ValueError(f'{key} must be nonempty text of at most 16000 characters')
        return value

    def attempt(self, attempt_id):
        if not isinstance(attempt_id, str) or attempt_id not in self.attempts:
            raise ValueError('Unknown attempt ID')
        return self.attempts[attempt_id]

    def best(self):
        successes = [result for result in self.evaluations if result['status'] == 'ok']
        if not successes:
            return None
        direction = 1 if self.task.task_spec['direction'] == 'minimize' else -1
        return min(successes, key=lambda result: direction * result['fitness'])

    def dispatch(self, payload):
        command = payload.get('command')
        if command == 'fitness-eval':
            return self.evaluate(payload)
        with self.lock:
            if self.closed:
                raise RuntimeError('Runtime is closed')
            if command == 'lifetime':
                return self.lifetime()
            if command != 'experiment-ledger':
                raise ValueError('Unknown command')
            action = payload.get('action')
            if action == 'register':
                source, digest = self.source(payload)
                fields = {key: self.text(payload, key) for key in ('title', 'hypothesis', 'planned_change', 'success_criterion')}
                parent = payload.get('parent')
                if parent is not None:
                    self.attempt(parent)
                attempt_id = f'A{len(self.attempts):02d}'
                snapshot = self.snapshots / (attempt_id + '.py')
                with snapshot.open('xb') as stream:
                    stream.write(source)
                snapshot.chmod(0o444)
                attempt = {'attempt_id': attempt_id, 'source_sha256': digest, 'parent': parent, **fields}
                self.append('registered', attempt)
                self.attempts[attempt_id] = attempt
                return attempt
            if action == 'history':
                return {'attempts': list(self.attempts.values()), 'evaluations': self.evaluations,
                        'best': self.best()}
            attempt = self.attempt(payload.get('attempt_id'))
            if action == 'select':
                if self.run_config.get('selection') != 'model':
                    raise ValueError('Model selection is disabled for this run')
                scored = next((e for e in reversed(self.evaluations) if e['attempt_id'] == attempt['attempt_id']
                               and e['status'] == 'ok'), None)
                if scored is None:
                    raise ValueError('Select an attempt with a successful evaluation')
                reason = self.text(payload, 'reason')
                self.append('selection', {**scored, 'reason': reason})
                self.selected = dict(scored, reason=reason)
                return self.selected
            if action == 'inspect':
                return {**attempt, 'events': [event for event in self.events if event.get('attempt_id') == attempt['attempt_id']]}
            if action == 'restore':
                source = (self.snapshots / (attempt['attempt_id'] + '.py')).read_bytes()
                self.append('restore_requested', {'attempt_id': attempt['attempt_id']})
                return {'attempt_id': attempt['attempt_id'], 'source_sha256': attempt['source_sha256'],
                        'source': base64.b64encode(source).decode()}
            summary = self.text(payload, 'summary')
            if action == 'observe':
                if payload.get('kind') != 'exploratory':
                    raise ValueError('Agent observations must be exploratory')
                return self.append('observation', {'attempt_id': attempt['attempt_id'], 'kind': 'exploratory', 'summary': summary})
            if action == 'conclude':
                if payload.get('decision') not in ('keep', 'reject', 'follow_up'):
                    raise ValueError('Choose keep, reject, or follow_up')
                return self.append('conclusion', {'attempt_id': attempt['attempt_id'], 'decision': payload['decision'], 'summary': summary})
            raise ValueError('Unknown ledger action')

    def evaluate(self, payload):
        if not self.evaluation_lock.acquire(blocking=False):
            raise RuntimeError('An evaluation is already running')
        result = None
        self.evaluation_done.clear()
        try:
            with self.lock:
                if self.closed:
                    raise RuntimeError('Runtime is closed')
                _, digest = self.source(payload)
                requested = payload.get('attempt_id')
                if requested is not None:
                    attempt = self.attempt(requested)
                else:
                    matches = [attempt for attempt in self.attempts.values() if attempt['source_sha256'] == digest]
                    if not matches:
                        raise ValueError('Register this exact source with experiment-ledger before evaluating')
                    attempt = matches[-1]
                if attempt['source_sha256'] != digest:
                    raise ValueError('Source changed since registration; register a new revision')
                budget = self.lifetime()
                for key in ('remaining_seconds', 'evaluations_remaining', 'tokens_remaining'):
                    if budget[key] is not None and budget[key] <= 0:
                        raise RuntimeError('Run budget exhausted: ' + key)
                timeout = self.run_config.get('evaluation_timeout_seconds', 30)
                if budget['remaining_seconds'] is not None:
                    timeout = min(timeout, budget['remaining_seconds'])
                result = {'evaluation_id': f'E{len(self.evaluations):02d}', 'attempt_id': attempt['attempt_id'],
                          'source_sha256': digest, 'split': 'dev', 'seed': self.run_config.get('seed', 0),
                          'image_id': self.image_id, 'status': 'running', 'fitness': None}
                self.append('evaluation_started', result)
                self.evaluations.append(result)
                self.job = EvaluationJob(self.task, self.snapshots / (attempt['attempt_id'] + '.py'),
                                         self.directory / 'evaluations' / result['evaluation_id'],
                                         self.image_id, self.image_path, self.config, result['seed'], timeout)
            score = self.job.run()
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('Fitness must be a finite number')
            with self.lock:
                result.update(status='ok', fitness=float(score))
        except Exception as error:
            if result is None:
                raise
            with self.lock:
                result.update(status='error', error=type(error).__name__ + ': ' + str(error)[:2000])
        finally:
            with self.lock:
                if result is not None:
                    self.append('evaluation_finished', result)
                self.job = None
            self.evaluation_done.set()
            self.evaluation_lock.release()
        return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            job = self.job
        if job is not None:
            job.cancel()
        self.evaluation_done.wait(timeout=60)
        self.server.close()
        with self.lock:
            self.finished = time.monotonic()
            self.append('run_closed', {'lifetime': self.lifetime(), 'best': self.best()})
