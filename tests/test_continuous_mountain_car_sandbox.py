"""Opt-in Docker integration: AUTOASTRA_TEST_MOUNTAIN_CAR=1 python -m unittest ..."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from autoastra.sandbox import SandboxConfig, create_sandbox
from autoastra.wrappers.mountain_car import score_runs
from tasks.continuous_mountain_car import task


@unittest.skipUnless(os.environ.get('AUTOASTRA_TEST_MOUNTAIN_CAR') == '1',
                     'Requires the Mountain Car image on a Linux Docker host')
class MountainCarSandboxTests(unittest.TestCase):
    def test_official_commands_and_final_test_wrapper(self):
        with tempfile.TemporaryDirectory(prefix='mountain-car-integration-') as temporary:
            root = Path(temporary)
            config = SandboxConfig(cpus=2, memory_mb=4096, pids_limit=256, tmp_mb=256,
                                   model_hosts=('api.openai.com',))
            with create_sandbox(task, root / 'sandbox', config=config) as sandbox:
                hidden = sandbox.run(['python', '-c',
                    'from pathlib import Path; assert not Path("/task/context/environment.json").exists()'])
                self.assertEqual(hidden.returncode, 0, hidden.stderr)
                def command(*args):
                    response = sandbox.run(list(args), timeout=600)
                    self.assertEqual(response.returncode, 0, response.stdout + response.stderr)
                    return json.loads(response.stdout)

                def register(title):
                    return command('experiment-ledger', 'register', 'final_solution.py',
                                   '--title', title, '--hypothesis', 'The controller matches its reference behaviour',
                                   '--planned-change', title, '--success-criterion', 'Valid official fitness')

                register('Zero-action controller')
                zero = command('fitness-eval', 'final_solution.py')
                self.assertEqual(zero['fitness'], 0)
                self.assertEqual(zero['status'], 'ok')
                source = task.starter.replace('jnp.zeros((1,), dtype=jnp.float32)',
                    'jnp.asarray([jnp.where(obs[1] >= 0, 1., -1.)], dtype=jnp.float32)')
                (sandbox.workspace / 'final_solution.py').write_text(source)
                register('Momentum controller')
                selected = command('fitness-eval', 'final_solution.py')
                self.assertGreater(selected['fitness'], 70)
                self.assertEqual(selected['status'], 'ok')
                private_config = (root / 'sandbox/.runtime/evaluations' / selected['evaluation_id'] /
                                  'code/tasks/continuous_mountain_car/environment.json')
                self.assertTrue(private_config.is_file())
                command('lifetime')
                self.assertEqual(command('fitness-eval', 'final_solution.py')['fitness'], selected['fitness'])
                image_id = sandbox.image_id

            # Construct a finished-run fixture from real official evaluations.
            result = {'task': task.task_spec['name'], 'seed': 0, 'image_id': image_id,
                      'selected': selected, 'status': 'completed'}
            (root / 'sandbox/result.json').write_text(json.dumps(result))
            report = score_runs([root / 'sandbox'], root / 'report')
            self.assertEqual(report[0]['episodes'], 1024)
            self.assertEqual(report[0]['success_rate'], 1)
            self.assertGreater(report[0]['mean_return'], 70)
            self.assertTrue((root / 'report/summary.csv').is_file())
            self.assertTrue((root / 'sandbox/task/ledger/ledger.jsonl').is_file())
            print(json.dumps({'development_return': selected['fitness'],
                              'test_return': report[0]['mean_return'],
                              'test_success_rate': report[0]['success_rate'],
                              'image_id': image_id}), flush=True)


if __name__ == '__main__':
    unittest.main()
