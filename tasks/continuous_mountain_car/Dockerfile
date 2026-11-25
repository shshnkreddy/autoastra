FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV JAX_PLATFORMS=cpu \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /opt/mountain-car/requirements.txt
RUN pip install --no-cache-dir -r /opt/mountain-car/requirements.txt \
    && pip check \
    && pip freeze > /opt/mountain-car/installed-requirements.txt

COPY sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py

# The image contains dependencies and public Gymnax source, not the evaluator.
WORKDIR /task/workspace
