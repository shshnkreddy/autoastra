"""Exercise supervision with real containers and a deterministic Codex protocol peer."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from autoastra.autoresearch import RunnerConfig, run
from autoastra.autoresearch.runner import _agent_paths, _command
from autoastra.sandbox.sandbox import _container_options, SandboxConfig
from tasks.affine import task


class AutoresearchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='autoresearch-test-'))
        self.home = self.root / 'codex-home'
        self.home.mkdir()
        self.binary = self.root / 'codex'
        shutil.copyfile(Path(__file__).parent / 'fixtures/codex_stub.py', self.binary)
        self.binary.chmod(0o755)
        self.addCleanup(self.remove_files)

    def remove_files(self):
        for current, _, _ in os.walk(self.root):
            Path(current).chmod(0o700)
        shutil.rmtree(self.root)

    def research(self, mode, **limits):
        (self.home / 'mode').write_text(mode)
        with patch('autoastra.autoresearch.runner._agent_paths', return_value=(self.binary, self.home)):
            result = run(task, self.root / 'run', config=RunnerConfig(conclusion_seconds=1),
                         run_config={'time_seconds': 30, **limits})
        directory = Path(result['sandbox'])
        self.assertEqual(directory, self.root / 'run')
        self.assertEqual((directory / 'task/ledger/ledger.jsonl').read_bytes(),
                         (directory / '.runtime/ledger.jsonl').read_bytes())
        report = (directory / 'task/ledger/ledger.md').read_text()
        self.assertIn('**Status:** ' + result['status'].capitalize(), report)
        return result

    def test_one_session_restores_best_and_counts_cumulative_usage(self):
        result = self.research('success')
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['best']['fitness'], 0)
        self.assertEqual(result['best']['attempt_id'], 'A00')
        self.assertEqual(result['lifetime']['tokens_used'], 40)
        self.assertEqual(result['lifetime']['evaluations_used'], 2)
        solution = Path(result['solution'])
        self.assertFalse(solution.is_symlink())
        self.assertIn('2*x+1', solution.read_text())
        requests = [json.loads(line) for line in (self.home / 'requests.jsonl').read_text().splitlines()]
        self.assertEqual(sum(item['method'] == 'thread/start' for item in requests), 1)
        self.assertEqual(sum(item['method'] == 'turn/start' for item in requests), 1)
        self.assertEqual(json.loads((self.root / 'run/result.json').read_text()), result)
        ledger = [json.loads(line) for line in (self.root / 'run/.runtime/ledger.jsonl').read_text().splitlines()]
        self.assertEqual(ledger[-1]['type'], 'research_finished')
        started = next(event for event in ledger if event['type'] == 'run_started')
        self.assertFalse(started['sandbox_config']['network_access'])
        self.assertIn('api.openai.com', started['sandbox_config']['model_hosts'])
        self.assertTrue(any(item['type'] == 'codex_session_started' for item in ledger))
        # The complete task tree is inspectable after copying it away from .runtime.
        saved = self.root / 'saved-task'
        shutil.copytree(self.root / 'run/task', saved)
        self.assertEqual(json.loads((saved / 'ledger/ledger.jsonl').read_text().splitlines()[-1])['type'],
                         'research_finished')
        self.assertEqual((saved / 'ledger/snapshots/A00.py').read_bytes(), solution.read_bytes())
        self.assertTrue((saved / 'ledger/snapshots/A01.py').is_file())
        self.assertIn('Informal work', (saved / 'workspace/scratch.py').read_text())
        self.assertIn('Research notes', (saved / 'workspace/notes/findings.md').read_text())
        for name in ('AGENTS.md', 'context/task_spec.md', 'context/fitness_api.md',
                     'bin/fitness-eval', 'bin/experiment-ledger', 'bin/lifetime'):
            self.assertTrue((saved / name).is_file(), name)
        self.assertEqual((saved / 'ledger/ledger.jsonl').stat().st_mode & 0o777, 0o444)
        report = (saved / 'ledger/ledger.md').read_text()
        self.assertIn('**Best development fitness:** 0.0 (A00, E00)', report)
        self.assertIn('Measured fitness', report)
        self.assertIn('snapshots/A00.py', report)
        self.assertEqual((saved / 'ledger/ledger.md').stat().st_mode & 0o777, 0o444)

    def test_model_can_select_a_lower_scoring_revision(self):
        result = self.research('choose_second', selection='model')
        self.assertEqual(result['best']['attempt_id'], 'A00')
        self.assertEqual(result['selected']['attempt_id'], 'A01')
        self.assertIn('3*x+1', Path(result['solution']).read_text())

    def test_model_selection_requires_nomination(self):
        result = self.research('success', selection='model')
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['solution'])
        self.assertIsNotNone(result['best'])

    def test_cli_prints_finished_sandbox_path(self):
        import contextlib
        import io
        from autoastra.autoresearch.__main__ import main
        result = {'sandbox': str(self.root / 'run'), 'solution': 'final_solution.py',
                  'status': 'completed', 'error': None}
        output = io.StringIO()
        with patch('autoastra.autoresearch.__main__.run', return_value=result), contextlib.redirect_stdout(output):
            status = main(['tasks.affine:task', '--output', str(self.root / 'run')])
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), result['sandbox'])

    def test_general_network_access_requires_explicit_cli_flag(self):
        from autoastra.autoresearch.__main__ import main
        result = {'sandbox': str(self.root / 'run'), 'solution': 'final_solution.py',
                  'status': 'completed', 'error': None}
        for flags, expected in (([], False), (['--allow-internet'], True)):
            with patch('autoastra.autoresearch.__main__.run', return_value=result) as launch:
                main(['tasks.affine:task', '--output', str(self.root / 'run'), *flags])
            self.assertEqual(launch.call_args.kwargs['sandbox_config'].network_access, expected)
        with patch('autoastra.autoresearch.__main__.run', return_value=result) as launch:
            main(['tasks.affine:task', '--output', str(self.root / 'run'),
                  '--model-host', 'models.example.org'])
        self.assertEqual(launch.call_args.kwargs['sandbox_config'].model_hosts, ('models.example.org',))

    def test_evaluation_budget_allows_conclusion_and_restores(self):
        result = self.research('quota', max_evaluations=1)
        self.assertEqual(result['stop_reason'], 'evaluation_budget', result)
        self.assertEqual(result['best']['fitness'], 0)
        self.assertEqual(result['lifetime']['evaluations_used'], 1)
        self.assertIn('Measured fitness', (self.root / 'run/.runtime/ledger.jsonl').read_text())

    def test_null_budgets_allow_evaluations_and_report_no_cap(self):
        result = self.research('success', max_evaluations=None, max_tokens=None)
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['lifetime']['evaluations_used'], 2)
        for key in ('max_evaluations', 'evaluations_remaining', 'max_tokens', 'tokens_remaining'):
            self.assertIsNone(result['lifetime'][key])

    def test_time_limit_stops_idle_session(self):
        result = self.research('idle', time_seconds=2)
        self.assertEqual(result['stop_reason'], 'time_budget', result)
        self.assertTrue(result['lifetime']['closed'])
        self.assertIsNone(result['solution'])

    def test_token_limit_stops_session(self):
        result = self.research('tokens', max_tokens=30)
        self.assertEqual(result['stop_reason'], 'token_budget', result)
        self.assertEqual(result['lifetime']['tokens_used'], 40)

    def test_model_failure_is_not_success(self):
        result = self.research('startup_error')
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['error']['message'], 'test failure')

    def test_early_exit_is_not_success(self):
        result = self.research('crash')
        self.assertEqual(result['status'], 'failed', result)
        self.assertIn('exited before', result['error'])

    def test_interactive_request_fails_instead_of_hanging(self):
        result = self.research('interactive')
        self.assertEqual(result['status'], 'failed', result)
        self.assertIn('interactive operation', result['error'])
        self.assertEqual(result['thread_id'], 'thread-test')

    def test_unscored_finish_is_not_success(self):
        result = self.research('no_score')
        self.assertEqual(result['status'], 'failed', result)
        self.assertIsNone(result['solution'])

    def test_evaluator_options_remain_offline(self):
        options = _container_options(SandboxConfig(network_access=True), '/usr/bin')
        self.assertIn('--network=none', options)

    def test_native_binary_validation_rejects_wrapper(self):
        with self.assertRaisesRegex(ValueError, 'native binary'):
            _agent_paths(RunnerConfig(codex_binary=self.binary, codex_home=self.home))

    def claude_run(self, model, **limits):
        shutil.copyfile(Path(__file__).parent / 'fixtures/claude_stub.py', self.binary)
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-only'}), patch(
            'autoastra.autoresearch.runner._agent_paths', return_value=(self.binary, None)
        ):
            return run(task, self.root / 'run', config=RunnerConfig(provider='anthropic', model=model),
                       run_config={'time_seconds': 30, **limits})

    def test_claude_evaluates_with_its_project_instructions(self):
        result = self.claude_run('sonnet')
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(result['thread_id'], 'claude-test')
        self.assertEqual(result['best']['fitness'], 0)
        self.assertEqual(result['lifetime']['tokens_used'], 19)
        self.assertIn('sonnet', (self.root / 'run/task/ledger/ledger.md').read_text())
        self.assertTrue((self.root / 'run/claude.events.jsonl').is_file())

    def test_claude_token_budget_and_duplicate_usage(self):
        result = self.claude_run('idle', max_tokens=15)
        self.assertEqual(result['stop_reason'], 'token_budget', result)
        self.assertEqual(result['lifetime']['tokens_used'], 15)

    def test_claude_error_is_retained(self):
        result = self.claude_run('failure')
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['error'], ['test failure'])

    def test_openrouter_requires_model(self):
        with self.assertRaisesRegex(ValueError, 'model'):
            RunnerConfig(provider='openrouter')

    def test_openrouter_key_is_not_in_command_arguments(self):
        from types import SimpleNamespace
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-secret'}):
            command = _command(SimpleNamespace(container_id='container'),
                               RunnerConfig(provider='openrouter', model='provider/model'), self.binary)
        self.assertIn('OPENROUTER_API_KEY', command)
        self.assertNotIn('test-secret', ' '.join(command))
        self.assertIn('model_providers.openrouter.wire_api="responses"', command)


if __name__ == '__main__':
    unittest.main()
