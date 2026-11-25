"""Keep JAX's native thread pools within the sandbox's CPU allowance."""

import os

# Docker's CPU quota does not reduce the CPU count seen by XLA. On large hosts,
# XLA can otherwise exhaust the container's process limit before its first step.
# AutoAstra sets OMP_NUM_THREADS from --cpus. This also applies to scratch.py.
if hasattr(os, "sched_getaffinity"):
    available = sorted(os.sched_getaffinity(0))
    try:
        threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    except ValueError:
        threads = 1
    os.sched_setaffinity(0, available[:threads])
