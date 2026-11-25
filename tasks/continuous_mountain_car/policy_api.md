# Policy API

Implement these two functions in `final_solution.py`:

```python
import jax.numpy as jnp

def init_state():
    return jnp.asarray(0, dtype=jnp.int32)

def act(obs, state, rng):
    position, velocity = obs[0], obs[1]
    action = jnp.zeros((1,), dtype=jnp.float32)
    return action, state
```

`obs` is a float32 JAX array of shape `(2,)`, containing position and velocity.
`rng` is a fresh JAX PRNG key for this step. Use it if your policy is stochastic.
`state` is your own memory, initialised separately for each episode. It can be
an array, a tuple or dictionary of arrays, or an empty structure such as `()`.
All leaves must be finite numeric or boolean JAX arrays. Their shapes and dtypes
must stay fixed. Return the updated memory as the second item of the result.

`action` must be a floating-point JAX array of shape `(1,)`, with a finite value
in `[-1, 1]`. Negative values push left; positive values push right.

Both functions must be pure and compatible with JAX transformations. Use
`jax.numpy`, `jax.lax.cond`, or `jnp.where` for operations that depend on the
observation. A Python `if` on a traced observation will fail. Avoid side effects,
Python random generators, and file access. The evaluator batches episodes with
`jax.vmap` and compiles the rollout with `jax.jit` and `jax.lax.scan`.

For informal tests in `scratch.py`, construct the same public environment:

```python
import jax
from gymnax.environments.classic_control.continuous_mountain_car import ContinuousMountainCar
from final_solution import init_state, act

env = ContinuousMountainCar()
params = env.default_params
key = jax.random.PRNGKey(42)  # Choose your own exploratory seed.
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, params)
memory = init_state()
episode_return = 0.0
for _ in range(params.max_steps_in_episode):
    key, policy_key, step_key = jax.random.split(key, 3)
    action, memory = act(obs, memory, policy_key)
    obs, env_state, reward, terminated, truncated, info = env.step(
        step_key, env_state, action, params)
    episode_return += float(reward)
    if bool(terminated | truncated):
        break
print(episode_return)
```

This pinned Gymnax version returns six values from `step` and automatically
resets the environment at an episode boundary. Always stop counting rewards
at the first termination or truncation. Informal tests are exploratory;
register the revision and call `fitness-eval` for its official score.
