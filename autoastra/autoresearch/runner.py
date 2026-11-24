"""Supervise one coding agent conversation; the agent owns the experiment cycle."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import time

from . import claude, opencode
from .config import RunnerConfig
from ..sandbox import SandboxConfig, create_sandbox
from ..sandbox.runtime import validate_run_config


INSTRUCTION = '''Read /task/{instructions}, /task/context/task_spec.md and
/task/context/fitness_api.md. Carry out the research task in /task/workspace.
You own the complete edit, evaluate, and record cycle in this single session.
Use fitness-eval for official scores, experiment-ledger for provenance, and
lifetime to check your budgets. Iterate while useful and within those budgets.
Do not start other agents. When no evaluations remain, record your conclusion
promptly and finish. Follow the final selection instructions in /task/{instructions}
and explain the result and its limitations. Do not ask for interactive approval.
'''


def _agent_paths(config):
    selected = getattr(config, f'{config.agent}_binary')
    binary = selected or shutil.which(config.agent)
    if binary is not None:
        binary = Path(binary).expanduser().resolve(strict=True)
        with binary.open('rb') as stream:
            magic = stream.read(4)
        if magic != b'\x7fELF' or not os.access(binary, os.X_OK):
            raise ValueError(f'--{config.agent}-binary must be an executable Linux native binary, not an npm wrapper')
    if config.agent == 'claude':
        claude.credential_name()
        return binary, None
    if config.agent == 'opencode':
        return binary, None
    home = config.profile_path.resolve()
    if config.provider == 'codex' and not (home / 'auth.json').is_file():
        raise ValueError(f'Sign in first: CODEX_HOME={home} codex -c cli_auth_credentials_store=\'"file"\' login')
    if config.provider == 'openrouter' and not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError('Set OPENROUTER_API_KEY on the execution host')
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    return binary, home


def _command(sandbox, config, binary):
    if config.agent == 'claude':
        return claude.command(sandbox.container_id, config.model, binary, INSTRUCTION.format(instructions='CLAUDE.md'))
    if config.agent == 'opencode':
        return opencode.command(sandbox.container_id, config.model, binary, INSTRUCTION.format(instructions='AGENTS.md'))
    command = ['docker', 'exec', '-i', '--env', 'CODEX_HOME=/opt/autoastra/codex-home']
    if config.provider == 'openrouter':
        # Docker inherits the value from its environment; the key is never in argv.
        command += ['--env', 'OPENROUTER_API_KEY']
    command += [sandbox.container_id, '/usr/local/bin/autoastra-codex' if binary else 'codex',
                '-c', 'cli_auth_credentials_store="file"',
                '-c', 'features.multi_agent=false',
                '-c', 'web_search="disabled"']
    if config.provider == 'openrouter':
        command += ['-c', 'model_provider="openrouter"',
                    '-c', 'model_providers.openrouter.name="OpenRouter"',
                    '-c', 'model_providers.openrouter.base_url="https://openrouter.ai/api/v1"',
                    '-c', 'model_providers.openrouter.env_key="OPENROUTER_API_KEY"',
                    '-c', 'model_providers.openrouter.wire_api="responses"']
    else:
        command += ['-c', 'model_provider="openai"']
    return command + ['app-server', '--listen', 'stdio://']


def _send(process, message):
    process.stdin.write((json.dumps(message) + '\n').encode())
    process.stdin.flush()


def _session(process, sandbox, config, events, cancel_event=None):
    """Stream agent events without blocking supervision on an idle pipe."""
    runtime = sandbox.runtime
    thread_id = turn_id = None
    reported_tokens = 0
    startup_deadline = time.monotonic() + config.startup_timeout_seconds
    conclusion_deadline = None
    buffer = b''
    reason, error = None, None
    stream = None
    if config.agent == 'opencode':
        stream = opencode.OpenCodeStream(sandbox, config.model)
    elif config.agent == 'claude':
        stream = claude.ClaudeStream(sandbox)
    else:
        _send(process, {'id': 1, 'method': 'initialize', 'params': {
            'clientInfo': {'name': 'autoastra', 'version': '0.1.0'},
            'capabilities': {'experimentalApi': False},
        }})
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while reason is None:
            now = time.monotonic()
            with runtime.lock:
                budget = runtime.lifetime()
                evaluating = not runtime.evaluation_done.is_set()
            if cancel_event is not None and cancel_event.is_set():
                reason = 'interrupted'
            elif budget['remaining_seconds'] is not None and budget['remaining_seconds'] <= 0:
                reason = 'time_budget'
            elif budget['max_tokens'] is not None and (budget['tokens_used'] or 0) >= budget['max_tokens']:
                reason = 'token_budget'
            elif startup_deadline is not None and now >= startup_deadline:
                reason, error = 'failed', f'{config.agent} did not start a turn before the startup timeout'
            elif budget['evaluations_remaining'] == 0 and not evaluating:
                if conclusion_deadline is None:
                    conclusion_deadline = now + config.conclusion_seconds
                if now >= conclusion_deadline:
                    reason = 'evaluation_budget'
            if reason is not None:
                break
            if not selector.select(timeout=0.1):
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                if isinstance(stream, opencode.OpenCodeStream):
                    code = process.poll()
                    if code is None:
                        time.sleep(0.05)
                        continue
                    reason, error = stream.finish(code)
                    break
                reason, error = 'failed', f'{config.agent} exited before completing its turn; see {config.agent}.stderr.log'
                break
            events.write(chunk)
            events.flush()
            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line.strip():
                    continue
                message = json.loads(line)
                if stream is not None:
                    reason, error = stream.receive(message)
                    thread_id = stream.session_id
                    if thread_id:
                        startup_deadline = None
                    if reason:
                        break
                    continue
                if 'id' in message and 'method' in message:
                    # An unattended run cannot answer questions or approval requests.
                    _send(process, {'id': message['id'], 'error': {
                        'code': -32601, 'message': 'AutoAstra does not support interactive requests'}})
                    raise RuntimeError('Codex requested an interactive operation: ' + message['method'])
                if 'error' in message:
                    raise RuntimeError('Codex request failed: ' + json.dumps(message['error']))
                if message.get('id') == 1:
                    with runtime.lock:
                        runtime.append('codex_initialized', {'server': message['result']})
                    _send(process, {'method': 'initialized', 'params': {}})
                    params = {'cwd': '/task/workspace', 'approvalPolicy': 'never',
                              'sandbox': 'danger-full-access', 'ephemeral': True}
                    if config.model:
                        params['model'] = config.model
                    _send(process, {'id': 2, 'method': 'thread/start', 'params': params})
                elif message.get('id') == 2:
                    thread_id = message['result']['thread']['id']
                    with runtime.lock:
                        runtime.append('codex_session_started', {'thread_id': thread_id,
                            'model': message['result'].get('model', config.model), 'provider': config.provider})
                    _send(process, {'id': 3, 'method': 'turn/start', 'params': {
                        'threadId': thread_id, 'input': [{'type': 'text', 'text': INSTRUCTION.format(instructions='AGENTS.md')}],
                        'approvalPolicy': 'never',
                        'sandboxPolicy': {'type': 'externalSandbox', 'networkAccess': 'enabled'},
                    }})
                elif message.get('id') == 3:
                    turn_id = message['result']['turn']['id']
                    startup_deadline = None
                method, params = message.get('method'), message.get('params', {})
                if method == 'thread/tokenUsage/updated' and params.get('threadId') == thread_id:
                    total = params['tokenUsage']['total']['totalTokens']
                    if type(total) is not int or total < 0:
                        raise ValueError('Codex returned invalid token usage')
                    if runtime.tokens_used is None or total > reported_tokens:
                        sandbox.record_token_usage(total - reported_tokens)
                        reported_tokens = total
                elif method == 'turn/completed' and params.get('threadId') == thread_id:
                    reason = params['turn']['status']
                    error = params['turn'].get('error')
            # Process every usage event already in this chunk before stopping.
    if reason in ('time_budget', 'token_budget', 'evaluation_budget', 'interrupted') and turn_id:
        try:
            _send(process, {'id': 4, 'method': 'turn/interrupt',
                            'params': {'threadId': thread_id, 'turnId': turn_id}})
        except (BrokenPipeError, OSError):
            pass
    return reason, error, thread_id


def run(task, destination, *, config=None, sandbox_config=None, run_config=None, cancel_event=None):
    """Run one session and retain the sandbox directory in its finished state.

    Task defaults override the fallback of ten minutes and twenty evaluations.
    Explicit run settings override task defaults. Null token/evaluation budgets
    mean no limit. Token limits act on reported
    cumulative usage, so an in-flight model request can exceed the limit.
    Run this on the Docker execution host, never through a remote Docker socket.
    """
    config = RunnerConfig() if config is None else config
    if not isinstance(config, RunnerConfig):
        raise TypeError('config must be a RunnerConfig')
    limits = validate_run_config({'time_seconds': 600, 'max_evaluations': 20,
                                  **task.resolve_run_config(run_config)})
    if limits.get('provider', config.provider) != config.provider:
        raise ValueError('The run configuration and RunnerConfig must use the same provider')
    limits['provider'] = config.provider
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError('Batch interrupted before seed startup')
    binary, home = _agent_paths(config)
    resources = sandbox_config or SandboxConfig(memory_mb=2048, pids_limit=256, tmp_mb=256)
    resources = replace(resources, network_access=True)
    sandbox = create_sandbox(task, destination, config=resources, run_config=limits,
                             codex_home=home, **{f'{config.agent}_binary': binary})
    runtime = sandbox.runtime
    process = None
    reason, error, thread_id = 'failed', None, None
    try:
        with (sandbox.directory / f'{config.agent}.events.jsonl').open('wb') as events, \
             (sandbox.directory / f'{config.agent}.stderr.log').open('wb') as stderr:
            process = subprocess.Popen(_command(sandbox, config, binary),
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
            reason, error, thread_id = _session(process, sandbox, config, events, cancel_event)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        # Removing the container also stops shell children and any active evaluator.
        # Do this before touching final_solution.py on the host.
        try:
            sandbox.close()
        finally:
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()
    best = runtime.best()
    if thread_id is None:
        thread_id = next((event['thread_id'] for event in runtime.events
                          if event['type'] == f'{config.agent}_session_started'), None)
    solution = None
    mode = limits.get('selection', 'best-fitness')
    selected = runtime.selected if mode == 'model' else best
    if selected is not None:
        source = (runtime.snapshots / (selected['attempt_id'] + '.py')).read_bytes()
        if hashlib.sha256(source).hexdigest() != selected['source_sha256']:
            raise RuntimeError('The best source snapshot failed its hash check')
        # Replace a possible symlink; never follow an agent-controlled final path.
        target = sandbox.workspace / 'final_solution.py'
        if target.is_dir() and not target.is_symlink():
            raise RuntimeError('final_solution.py is a directory')
        target.unlink(missing_ok=True)
        target.write_bytes(source)
        solution = str(target)
    if reason == 'completed' and best is None:
        reason, error = 'failed', 'The session finished without a successful fitness evaluation'
    if reason == 'completed' and selected is None and best is not None:
        reason, error = 'failed', 'No model selection was recorded'
    if reason == 'completed':
        status = 'completed'
    elif reason == 'failed':
        status = 'failed'
    else:
        status = 'stopped'
    result = {'provider': config.provider, 'seed': limits.get('seed', 0), 'task': task.task_spec['name'],
              'zsc': task.task_spec.get('zsc'), 'image_id': sandbox.image_id,
              'selection': mode, 'selected': selected, 'status': status,
              'stop_reason': reason, 'thread_id': thread_id, 'best': best,
              'sandbox': str(sandbox.directory),
              'solution': solution, 'lifetime': runtime.lifetime(), 'error': error}
    runtime.append('research_finished', result)
    (sandbox.directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    sandbox.save_state()
    return result
