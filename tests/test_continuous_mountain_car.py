"""Run inside the Mountain Car image; Docker integration is a separate host check."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import jax
    import gymnax
except ModuleNotFoundError as error:
    raise unittest.SkipTest('Run these tests inside the Mountain Car task image') from error
import jax.numpy as jnp
import numpy as np
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar

from tasks.continuous_mountain_car import task
from tasks.continuous_mountain_car.evaluate import evaluate
from tasks.continuous_mountain_car.rollouts import episode_keys, evaluate_policy
from autoastra.wrappers.mountain_car import _selected


MOMENTUM = '''import jax.numpy as jnp
def init_state():
    return ()
def act(obs, state, rng):
    return jnp.asarray([jnp.where(obs[1] >= 0, 1., -1.)], dtype=jnp.float32), state
'''


class MountainCarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.solution = self.directory / 'final_solution.py'
        self.solution.write_text(task.starter)

    def test_task_budget_and_zero_policy(self):
        self.assertEqual(task.run_config['time_seconds'], 7200)
        self.assertIsNone(task.run_config['max_tokens'])
        self.assertIsNone(task.run_config['max_evaluations'])
        result = evaluate_policy(self.solution, split='dev', seed=0, episodes=4)
        self.assertEqual(result['mean_return'], 0)
        self.assertEqual(result['success_rate'], 0)
        self.assertEqual(result['mean_episode_length'], 999)
        self.assertEqual(evaluate(self.solution, split='dev', seed=0), 0)
        with self.assertRaises(ValueError):
            evaluate(self.solution, split='test', seed=0)

    def test_momentum_policy_and_seed_reproducibility(self):
        self.solution.write_text(MOMENTUM)
        result = evaluate_policy(self.solution, split='dev', seed=2, episodes=16)
        self.assertGreater(result['mean_return'], 70)
        self.assertEqual(result['success_rate'], 1)
        self.assertEqual(result, evaluate_policy(self.solution, split='dev', seed=2, episodes=16))
        for split, seed in [('test', 2), ('dev', 3)]:
            self.assertFalse(np.array_equal(episode_keys('dev', 2, 16)[0],
                                            episode_keys(split, seed, 16)[0]))
        np.testing.assert_array_equal(episode_keys('dev', 2, 4)[0], episode_keys('dev', 2, 16)[0][:4])

    def test_first_terminal_reward_only(self):
        self.solution.write_text(MOMENTUM)
        params = ContinuousMountainCar().default_params.replace(goal_position=-0.7, goal_velocity=-1.)
        result = evaluate_policy(self.solution, split='dev', seed=0, episodes=4, params=params)
        self.assertEqual(result['mean_episode_length'], 1)
        self.assertEqual(result['success_rate'], 1)
        self.assertAlmostEqual(result['mean_return'], 99.9, places=4)
        self.assertEqual(result['mean_control_effort'], 1)

    def test_matches_single_episode_reference(self):
        self.solution.write_text(MOMENTUM)
        result = evaluate_policy(self.solution, split='dev', seed=9, episodes=2)
        reset_keys, policy_keys = episode_keys('dev', 9, 2)
        env = ContinuousMountainCar()
        params = env.default_params
        for index in range(2):
            obs, state = env.reset(reset_keys[index], params)
            key = policy_keys[index]
            total = 0.
            for length in range(1, 1000):
                key, _, step_key = jax.random.split(key, 3)
                action = jnp.asarray([jnp.where(obs[1] >= 0, 1., -1.)], jnp.float32)
                obs, state, reward, terminated, truncated, _ = env.step(step_key, state, action, params)
                total += float(reward)
                if bool(terminated | truncated):
                    break
            self.assertAlmostEqual(result['episode_returns'][index], total, places=3)
            self.assertEqual(result['episode_lengths'][index], length)

    def test_invalid_policies(self):
        for expression in ('jnp.array([2.])', 'jnp.array([jnp.nan])',
                           'jnp.array(0.)', 'jnp.array([0])', 'jnp.zeros(2)'):
            with self.subTest(expression=expression):
                self.solution.write_text('import jax.numpy as jnp\ndef init_state(): return ()\n'
                                         f'def act(obs,state,rng): return {expression}, state\n')
                with self.assertRaises(ValueError):
                    evaluate_policy(self.solution, split='dev', seed=0, episodes=2)
        self.solution.write_text('import jax.numpy as jnp\ndef init_state(): return jnp.zeros(1)\n'
                                 'def act(obs,state,rng): return jnp.zeros(1), jnp.zeros(2)\n')
        with self.assertRaises(ValueError):
            evaluate_policy(self.solution, split='dev', seed=0, episodes=2)
        self.solution.write_text('import jax.numpy as jnp\ndef init_state(): return jnp.zeros(1)\n'
                                 'def act(obs,state,rng): return jnp.zeros(1), jnp.array([jnp.inf])\n')
        with self.assertRaises(ValueError):
            evaluate_policy(self.solution, split='dev', seed=0, episodes=2)

    def test_wrapper_rejects_changed_solution(self):
        workspace = self.directory / 'run/task/workspace'
        workspace.mkdir(parents=True)
        source = task.starter.encode()
        (workspace / 'final_solution.py').write_bytes(source)
        (self.directory / 'run/result.json').write_text(json.dumps({
            'task': 'continuous_mountain_car', 'image_id': 'sha256:' + 'a' * 64,
            'selected': {'status': 'ok', 'source_sha256': hashlib.sha256(source).hexdigest()}}))
        _selected(self.directory / 'run')
        (workspace / 'final_solution.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            _selected(self.directory / 'run')


if __name__ == '__main__':
    unittest.main()
