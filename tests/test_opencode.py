"""OpenCode's JSON stream reports model steps separately from the final CLI exit."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from autoastra.autoresearch import RunnerConfig
from autoastra.autoresearch.opencode import OpenCodeStream, command


class OpenCodeTests(unittest.TestCase):
    def test_counts_steps_once_and_requires_clean_exit(self):
        runtime = SimpleNamespace(lock=threading.Lock(), append=Mock())
        sandbox = SimpleNamespace(runtime=runtime, record_token_usage=Mock())
        stream = OpenCodeStream(sandbox, 'openrouter/example/model')
        finish = dict(type='step_finish', sessionID='s', part={
            'reason': 'tool-calls', 'id': '1', 'tokens': {'total': 17}})
        self.assertEqual(stream.receive(finish), (None, None))
        stream.receive(finish)
        sandbox.record_token_usage.assert_called_once_with(17)
        self.assertEqual(stream.finish(0)[0], 'failed')
        stream.receive({**finish, 'part': {'id': '2', 'reason': 'stop', 'tokens': {
            'input': 5, 'output': 3, 'reasoning': 2, 'cache': {'read': 7, 'write': 1}}}})
        sandbox.record_token_usage.assert_called_with(18)
        self.assertEqual(stream.finish(0), ('completed', None))
        self.assertEqual(stream.finish(1)[0], 'failed')
        self.assertEqual(stream.receive(dict(type='error', sessionID='s', error='model failed')),
                         ('failed', 'model failed'))
        runtime.append.assert_called_once()

    def test_selects_only_the_requested_model_credential(self):
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fixture-secret', 'ANTHROPIC_API_KEY': 'unused'}):
            args = command('container', 'openrouter/provider/model', None, 'Read /task/AGENTS.md')
        self.assertIn('OPENROUTER_API_KEY', args)
        self.assertNotIn('ANTHROPIC_API_KEY', args)
        self.assertNotIn('fixture-secret', ' '.join(args))
        with self.assertRaisesRegex(ValueError, 'provider/model'):
            RunnerConfig(provider='opencode', model='missing-provider')
