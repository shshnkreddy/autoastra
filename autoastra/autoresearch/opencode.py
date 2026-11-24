"""Run one OpenCode CLI session and account for each completed model step."""

import json
import os


def command(container, model, binary, instruction):
    config = {'instructions': ['/task/AGENTS.md'], 'permission': {
        'external_directory': {'/task/*': 'allow'}, 'task': 'deny'}}
    args = ['docker', 'exec', '--env', 'OPENCODE_CONFIG_CONTENT=' + json.dumps(config)]
    key = {'openrouter': 'OPENROUTER_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY',
           'openai': 'OPENAI_API_KEY', 'opencode': 'OPENCODE_API_KEY',
           'opencode-go': 'OPENCODE_API_KEY'}.get(model.partition('/')[0])
    if key and os.environ.get(key):
        args += ['--env', key]
    return args + [container, '/usr/local/bin/opencode' if binary else 'opencode',
                   'run', '--format', 'json', '--title', 'AutoAstra research', '--model', model, instruction]


class OpenCodeStream:
    def __init__(self, sandbox, model):
        self.sandbox = sandbox
        self.model = model
        self.session_id = None
        self.steps = set()
        self.last_finish = None

    def receive(self, event):
        session_id = event['sessionID']
        if self.session_id is None:
            self.session_id = session_id
            runtime = self.sandbox.runtime
            with runtime.lock:
                runtime.append('opencode_session_started', {'thread_id': session_id,
                               'model': self.model, 'provider': 'opencode'})
        elif session_id != self.session_id:
            raise ValueError('OpenCode reported a different session ID')
        if event['type'] == 'error':
            return 'failed', event['error']
        if event['type'] == 'step_start':
            self.last_finish = None
        if event['type'] == 'step_finish':
            part = event['part']
            self.last_finish = part['reason']
            if part['id'] not in self.steps:
                self.steps.add(part['id'])
                tokens = part['tokens']
                total = tokens.get('total')
                if total is None:
                    cache = tokens.get('cache', {})
                    total = sum((tokens['input'], tokens['output'], tokens.get('reasoning', 0),
                                 cache.get('read', 0), cache.get('write', 0)))
                if type(total) is not int or total < 0:
                    raise ValueError('OpenCode reported invalid token usage')
                self.sandbox.record_token_usage(total)
        return None, None

    def finish(self, exit_code):
        if exit_code == 0 and self.last_finish == 'stop':
            return 'completed', None
        return 'failed', (f'OpenCode exited with code {exit_code} and last step {self.last_finish!r}. '
                          'See opencode.stderr.log')
