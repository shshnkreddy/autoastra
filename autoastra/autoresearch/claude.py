"""Launch Claude Code in print mode and interpret its stream-json events."""

import os


def credential_name():
    for name in ('ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN'):
        if os.environ.get(name):
            return name
    raise ValueError('Set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN on the execution host')


def command(container, model, binary, instruction):
    tools = 'Bash,Edit,Read,Write,Glob,Grep'
    args = ['docker', 'exec', '--env', credential_name(), container,
            '/usr/local/bin/autoastra-claude' if binary else 'claude',
            '-p', '--output-format', 'stream-json', '--verbose', '--no-session-persistence',
            '--tools', tools, '--allowedTools', tools, '--permission-mode', 'dontAsk']
    if model:
        args += ['--model', model]
    return args + ['--', instruction]


def token_count(usage):
    counts = [usage.get(key, 0) for key in (
        'input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')]
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Claude Code returned invalid token usage')
    return sum(counts)


class ClaudeStream:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.session_id = None
        self.message_tokens = {}

    def receive(self, event):
        kind = event.get('type')
        runtime = self.sandbox.runtime
        if kind == 'system' and event.get('subtype') == 'init':
            self.session_id = event['session_id']
            with runtime.lock:
                runtime.append('claude_session_started', {'thread_id': self.session_id,
                               'model': event['model'], 'provider': 'anthropic'})
        elif kind == 'assistant':
            message = event['message']
            if message.get('id') and message.get('usage'):
                # Multiple tool blocks can report usage for the same API response.
                key = message['id']
                previous = self.message_tokens.get(key, 0)
                count = token_count(message['usage'])
                if count > previous:
                    self.message_tokens[key] = count
                    self.sandbox.record_token_usage(count - previous)
        elif kind == 'result':
            if event.get('usage'):
                total = token_count(event['usage'])
                self.sandbox.record_token_usage(max(0, total - (runtime.tokens_used or 0)))
            if event.get('is_error') or event.get('subtype') != 'success':
                return 'failed', event.get('errors') or event.get('result') or event.get('subtype')
            if not self.session_id:
                return 'failed', 'Claude Code returned a result without starting a session'
            return 'completed', None
        return None, None
