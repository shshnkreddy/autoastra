# Environment

This task uses Gymnax `ContinuousMountainCar`, pinned to commit
`77d4863333e7ec203d0b4a43f998b2b691fdb997`.
[Read the environment source](https://github.com/RobertTLange/gymnax/blob/77d4863333e7ec203d0b4a43f998b2b691fdb997/gymnax/environments/classic_control/continuous_mountain_car.py).
The same public source is installed in the Docker image.

The observation is `[position, velocity]`. Position is bounded by `[-1.2, 0.6]`
and velocity by `[-0.07, 0.07]`. Each episode starts with position uniform in
`[-0.6, -0.4]` and velocity zero.

At each step, the environment adds
`0.0015 * action - 0.0025 * cos(3 * position)` to the velocity, clips it to its
bounds, and adds it to the position, which is also clipped. Use the pinned
implementation for exact boundary behaviour; other Mountain Car libraries
need not implement identical dynamics.

Reaching position `0.45` with nonnegative velocity ends the episode successfully.
The maximum episode length is 999 steps. Reward at each step is
`100 * reached_goal - 0.1 * action**2`. The successful step counts, and no later
rewards count. Fitness is the mean undiscounted episode return, not success
rate or speed alone. Success rate, episode length, and summed squared actions
are additional diagnostics in the final test report.

The image includes Python, JAX (CPU), NumPy, Gymnax, Flax, and Matplotlib.
CPU execution is sufficient for this task. No GPU is required.
