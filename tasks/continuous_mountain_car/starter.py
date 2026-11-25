"""A valid starting policy. Replace this with your own controller."""

import jax.numpy as jnp


def init_state():
    return jnp.asarray(0, dtype=jnp.int32)


def act(obs, state, rng):
    return jnp.zeros((1,), dtype=jnp.float32), state
