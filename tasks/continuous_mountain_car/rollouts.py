"""Batched rollouts for the pinned Gymnax environment and policy contract."""

import argparse
import contextlib
import json
from pathlib import Path
import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar

SETTINGS = json.loads(Path(__file__).with_name("environment.json").read_text())
EPISODES = {"dev": SETTINGS["development_episodes"], "test": SETTINGS["test_episodes"]}
_SPLIT_KEYS = {"dev": 0x444556, "test": 0x54455354}


def episode_keys(split, seed, episodes):
    if split not in EPISODES:
        raise ValueError("split must be 'dev' or 'test'")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    root = jax.random.fold_in(jax.random.PRNGKey(seed), _SPLIT_KEYS[split])
    reset_root, policy_root = jax.random.split(root)
    indices = jnp.arange(episodes, dtype=jnp.uint32)
    return (jax.vmap(lambda i: jax.random.fold_in(reset_root, i))(indices),
            jax.vmap(lambda i: jax.random.fold_in(policy_root, i))(indices))


def _policy(path):
    module = runpy.run_path(str(path))
    init_state, act = module.get("init_state"), module.get("act")
    if not callable(init_state) or not callable(act):
        raise ValueError("Define init_state() and act(obs, state, rng)")
    memory = init_state()
    for leaf in jax.tree_util.tree_leaves(memory):
        if not isinstance(leaf, jax.Array):
            raise ValueError("Policy memory leaves must be JAX arrays")
        if leaf.dtype.kind not in "biuf" or not np.all(np.isfinite(np.asarray(leaf))):
            raise ValueError("Policy memory must contain finite numeric or boolean arrays")
    result = jax.eval_shape(act, jnp.zeros(2, jnp.float32), memory, jax.random.PRNGKey(0))
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("act must return (action, next_state)")
    action, next_memory = result
    if not hasattr(action, "dtype") or action.shape != (1,) or action.dtype.kind != "f":
        raise ValueError("Action must be a floating-point array of shape (1,)")
    if jax.tree_util.tree_structure(memory) != jax.tree_util.tree_structure(next_memory):
        raise ValueError("Policy memory structure must remain fixed")
    for before, after in zip(jax.tree_util.tree_leaves(memory), jax.tree_util.tree_leaves(next_memory)):
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError("Policy memory shapes and dtypes must remain fixed")
    return act, memory


def evaluate_policy(path, *, split, seed, episodes=None, params=None):
    """Score one episode per key. `params` is only for evaluator regression tests."""
    episodes = EPISODES.get(split) if episodes is None else episodes
    reset_keys, keys = episode_keys(split, seed, episodes)
    act, memory = _policy(path)
    env = ContinuousMountainCar()
    params = env.default_params if params is None else params
    memory = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (episodes,) + x.shape), memory)

    def rollout():
        obs, env_state = jax.vmap(lambda k: env.reset(k, params))(reset_keys)
        zeros = jnp.zeros(episodes, jnp.float32)
        flags = jnp.zeros(episodes, bool)
        initial = (obs, env_state, memory, keys, ~flags, zeros,
                   jnp.zeros(episodes, jnp.int32), zeros, flags, flags)

        def step(carry, _):
            obs, env_state, memory, keys, alive, returns, lengths, effort, success, invalid = carry
            parts = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
            keys, policy_keys, step_keys = parts[:, 0], parts[:, 1], parts[:, 2]
            actions, next_memory = jax.vmap(act)(obs, memory, policy_keys)
            valid = jnp.all(jnp.isfinite(actions) & (jnp.abs(actions) <= 1), axis=1)
            for leaf in jax.tree_util.tree_leaves(next_memory):
                valid &= jnp.all(jnp.isfinite(leaf).reshape((episodes, -1)), axis=1)
            invalid |= alive & ~valid
            safe_actions = jnp.where(valid[:, None], actions, 0)
            obs, env_state, reward, terminated, truncated, _ = jax.vmap(
                lambda k, s, a: env.step(k, s, a, params))(step_keys, env_state, safe_actions)
            returns += jnp.where(alive, reward, 0)
            lengths += alive.astype(jnp.int32)
            effort += jnp.where(alive, jnp.sum(safe_actions**2, axis=1), 0)
            success |= alive & terminated
            alive &= ~(terminated | truncated)
            return (obs, env_state, next_memory, keys, alive, returns,
                    lengths, effort, success, invalid), None

        final, _ = jax.lax.scan(step, initial, None, length=int(params.max_steps_in_episode))
        return final[5:]

    returns, lengths, effort, success, invalid = map(np.asarray, jax.jit(rollout)())
    if np.any(invalid):
        raise ValueError("Policy produced an invalid action or non-finite memory during an episode")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Non-finite episode return")
    return {
        "split": split, "seed": seed, "episodes": episodes,
        "mean_return": float(np.mean(returns, dtype=np.float64)),
        "return_std": float(np.std(returns, dtype=np.float64)),
        "return_standard_error": float(np.std(returns, dtype=np.float64) / np.sqrt(episodes)),
        "success_rate": float(np.mean(success)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_control_effort": float(np.mean(effort, dtype=np.float64)),
        "episode_returns": returns.tolist(), "episode_lengths": lengths.tolist(),
        "episode_control_effort": effort.tolist(), "episode_success": success.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--split", choices=EPISODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        result = evaluate_policy(args.solution, split=args.split, seed=args.seed, episodes=args.episodes)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
