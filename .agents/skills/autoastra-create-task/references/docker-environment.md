# Prepare the Docker environment

Use this guide when a task needs a new image, additional dependencies, datasets, or GPUs. Follow the user's chosen execution machine; task definitions and instructions should remain portable.

## Build or pull

The same task image supplies dependencies for the development container and each temporary evaluator. It must contain Python and the libraries needed to execute the submitted solution and evaluator. Put public environment code and required public data in the image when they cannot be represented as text context.

For a custom environment, adapt a Dockerfile such as:

```dockerfile
FROM python:3.12-slim
WORKDIR /opt/task
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY environment/ /opt/task/environment/
```

Add package installation or explicit absolute paths as required by the evaluator. Do not assume the evaluator runs in `/opt/task`; AutoAstra starts it in a separate working directory. Pin important dependencies for repeatable experiments.

On the execution machine:

```sh
docker build -t my-task:latest .
docker image inspect my-task:latest
```

Alternatively, use a registry reference in `Task.image` and pull it there:

```sh
docker pull registry.example.com/team/my-task:version
```

Replace the example registry with the user's actual image. AutoAstra reuses a local image or pulls it when absent; it does not build Dockerfiles. To complete task setup, ensure the required image is locally available on the execution machine. An image built only on a laptop is not automatically present on a remote Docker host. Authenticate to private registries on that host when necessary.

## Data and runtime constraints

- Evaluation is offline, with a read-only filesystem and writable temporary space. Download dependencies and datasets ahead of time; do not require network installation during evaluation.
- The evaluator receives the frozen solution and its evaluator package, not the editable development directory. Check paths from inside the actual image.
- Image contents are visible to the development agent. Do not bake private answers, held-out labels, or credentials into the shared image. Keep private evaluation assets with the evaluator package, or implement a deliberate evaluator-only data mechanism if that package is insufficient.
- CLI binaries and credentials are configured by AutoAstra on the host; do not bake account credentials into images. See the repository README for supported authentication.

Validate the baseline through the actual evaluator, including package imports and data access. For a sandbox integration check, use the documented `create_sandbox` interface or CLI help for the installed version, then register and score a candidate through the sandbox commands. A successful image build alone does not validate those paths.

## GPUs and concurrency

Use a CUDA-compatible task image and a Docker host with GPU support. Select resources explicitly:

```sh
autoastra run my_tasks.example:task --output runs/example-gpu --gpus 1
```

`--gpu-device` can select device IDs; check current CLI help. CPU, memory, and GPU settings are shared by the run's development and evaluation containers. Multiple seeds do not automatically receive distinct GPUs. If exclusive GPU allocation matters, have the wrapper assign devices per run and limit concurrency to available capacity.
