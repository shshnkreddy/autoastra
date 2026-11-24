"""Create a Docker sandbox on the machine running this module."""

import argparse
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from ..task.layout import create_layout
from ..task import Task


@dataclass(frozen=True)
class SandboxConfig:
    python_executable: str = "python"
    gpus: int | str = 0
    gpu_devices: tuple[str, ...] = ()
    cpus: float = 1
    memory_mb: int = 512
    pids_limit: int = 64
    tmp_mb: int = 64
    command_timeout_seconds: int = 60
    network_access: bool = False  # Development only; evaluations stay offline.

    def __post_init__(self):
        if type(self.network_access) is not bool:
            raise ValueError("network_access must be a boolean")
        if not isinstance(self.python_executable, str) or not self.python_executable.strip() or "\0" in self.python_executable:
            raise ValueError("python_executable must name the Python executable inside the image")
        if not ((type(self.gpus) is int and self.gpus >= 0) or self.gpus == "all"):
            raise ValueError("gpus must be a nonnegative integer or 'all'")
        if not isinstance(self.gpu_devices, tuple) or any(
            not isinstance(device, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", device)
            for device in self.gpu_devices
        ):
            raise ValueError("gpu_devices must be a tuple of device indices or GPU/MIG identifiers")
        if len(set(self.gpu_devices)) != len(self.gpu_devices):
            raise ValueError("gpu_devices must not contain duplicates")
        if self.gpus and self.gpu_devices:
            raise ValueError("Choose either gpus or gpu_devices")
        if type(self.cpus) not in (int, float) or not math.isfinite(self.cpus) or self.cpus <= 0:
            raise ValueError("cpus must be a positive finite number")
        for name in ("memory_mb", "pids_limit", "tmp_mb", "command_timeout_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def gpu_arguments(self) -> list[str]:
        if self.gpu_devices:
            # Docker parses this argument as CSV even without a shell.
            selection = '"device=' + ','.join(self.gpu_devices) + '"'
        elif self.gpus:
            selection = str(self.gpus)
        else:
            # CUDA images can default to exposing all GPUs with an NVIDIA runtime.
            return ["--env", "NVIDIA_VISIBLE_DEVICES=void"]
        return ["--gpus", selection, "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]


def _docker(*arguments, timeout=30, check=True):
    try:
        return subprocess.run(["docker", *arguments], capture_output=True, text=True,
                              check=check, timeout=timeout)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Docker {arguments[0]} failed: {error.stderr.strip()}") from error


def _resolve_image(reference):
    """Use a host image if present; otherwise pull it before creating a sandbox."""
    result = _docker("image", "inspect", reference, check=False)
    if result.returncode:
        if "No such image:" not in result.stderr:
            raise RuntimeError(f"Docker image inspect failed: {result.stderr.strip()}")
        _docker("pull", reference, timeout=600)
        result = _docker("image", "inspect", reference)
    return json.loads(result.stdout)[0]


def _container_options(config, image_path):
    memory = f'{config.memory_mb}m'
    threads = max(1, math.floor(config.cpus))
    return [
        '--pull=never', '--init', '--network=none', '--read-only',
        '--cap-drop=ALL', '--security-opt=no-new-privileges=true',
        '--user', f'{os.getuid()}:{os.getgid()}',
        '--cpus', str(config.cpus), '--memory', memory, '--memory-swap', memory,
        '--pids-limit', str(config.pids_limit), '--ulimit', 'nofile=256:256',
        '--tmpfs', f'/tmp:rw,nosuid,nodev,noexec,size={config.tmp_mb}m,mode=1777',
        '--env', 'HOME=/tmp', '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--env', f'OMP_NUM_THREADS={threads}', '--env', f'OPENBLAS_NUM_THREADS={threads}',
        '--env', f'MKL_NUM_THREADS={threads}', '--env', f'PATH=/task/bin:{image_path}',
        *config.gpu_arguments(),
    ]


@dataclass
class Sandbox:
    directory: Path
    container_id: str
    image_id: str
    config: SandboxConfig
    runtime: object = None
    closed: bool = False

    @property
    def workspace(self) -> Path:
        return self.directory / "task/workspace"

    def run(self, command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
        """Execute an argument array in /task/workspace and capture its output.

        A timeout stops the entire container so child processes cannot keep
        running. The working files remain available on the execution host.
        """
        if self.closed:
            raise RuntimeError("The sandbox is closed")
        if not isinstance(command, list) or not command or any(
            not isinstance(arg, str) or "\0" in arg for arg in command
        ) or not command[0]:
            raise ValueError("command must be a nonempty argument array")
        timeout = self.config.command_timeout_seconds if timeout is None else timeout
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        try:
            return subprocess.run(
                ["docker", "exec", self.container_id, *command],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            self.close()
            raise

    def close(self):
        """Remove this container and save its complete task directory."""
        if not self.closed:
            try:
                _docker("rm", "--force", self.container_id)
            finally:
                if self.runtime is not None:
                    self.runtime.close()
            self.closed = True
            self.save_state()

    def save_state(self) -> Path:
        """Materialize the ledger mounts in the finished sandbox directory.

        Workspace, context and tools already live on the host. Copy the final
        ledger and snapshots into their visible task paths after shutdown so
        the saved task tree can be inspected without Docker or the runtime.
        Calling again includes any final records appended by the runner.
        """
        if not self.closed:
            raise RuntimeError('Stop the sandbox before saving its final state')
        if self.runtime is not None:
            ledger = self.directory / 'task/ledger'
            snapshots = ledger / 'snapshots'
            files = [(self.runtime.ledger, ledger / 'ledger.jsonl')]
            files += [(source, snapshots / source.name)
                      for source in sorted(self.runtime.snapshots.glob('*.py'))]
            ledger.chmod(0o755)
            snapshots.chmod(0o755)
            try:
                for source, target in files:
                    temporary = target.with_name('.' + uuid.uuid4().hex)
                    try:
                        shutil.copyfile(source, temporary)
                        temporary.chmod(0o444)
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
                from .ledger_report import write_ledger_report
                write_ledger_report(self.directory)
            finally:
                snapshots.chmod(0o555)
                ledger.chmod(0o555)
        return self.directory

    def record_token_usage(self, count: int):
        """Record tokens consumed by the trusted model runner."""
        if self.runtime is None:
            raise RuntimeError('No runtime is attached')
        self.runtime.record_tokens(count)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_sandbox(task: Task, destination: str | Path, *,
                   config: SandboxConfig | None = None, run_config=None,
                   codex_home: Path | None = None, codex_binary: Path | None = None,
                   claude_binary: Path | None = None, opencode_binary: Path | None = None) -> Sandbox:
    """Create the task bundle and start an isolated container on this host.

    The destination must be new, on the Docker daemon's host.
    The task's image is reused if present, or pulled from its registry if missing.
    It must contain Python and the task's dependencies.
    run_config controls time and evaluation allowances and evaluation timeout.
    Optional agent mounts are used only by the development container.
    """
    from .runtime import Runtime, validate_run_config
    config = SandboxConfig() if config is None else config
    if not isinstance(config, SandboxConfig):
        raise TypeError("config must be a SandboxConfig")
    if not isinstance(task, Task):
        raise TypeError("task must be a Task")
    run_config = validate_run_config(task.resolve_run_config(run_config))
    agent_mounts = []
    mounts = [
        (opencode_binary, '/usr/local/bin/opencode', True),
        (claude_binary, '/usr/local/bin/autoastra-claude', True),
        (codex_home, '/opt/autoastra/codex-home', False),
        (codex_binary, '/usr/local/bin/autoastra-codex', True),
    ]
    if codex_binary is not None:
        binary = Path(codex_binary).resolve(strict=True)
        # Current standalone Codex releases launch these sibling executables.
        # Mount only package assets, never the surrounding host directory.
        for name in ('codex-code-mode-host', 'codex-exec-server'):
            companion = binary.parent / name
            if companion.is_file():
                mounts.append((companion, '/usr/local/bin/' + name, True))
        package = binary.parent.parent
        if (package / 'codex-package.json').is_file():
            for name in ('codex-package.json', 'codex-path', 'codex-resources'):
                asset = package / name
                if asset.exists():
                    mounts.append((asset, '/usr/local/' + name, True))
    for source, target, readonly in mounts:
        if source is not None:
            source = Path(source).resolve(strict=True)
            if any(char in str(source) for char in (',', '"', '\n', '\r')):
                raise ValueError('Agent mount paths cannot contain commas, quotes, or newlines')
            agent_mounts += ['--mount', f'type=bind,src={source},dst={target},bind-recursive=disabled' + (',readonly' if readonly else '')]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"The sandbox destination already exists: {destination}")
    if any(character in str(destination) for character in (",", '"', "\n", "\r")):
        raise ValueError("The sandbox path cannot contain commas, double quotes, or newlines")
    if not hasattr(os, "getuid"):
        raise RuntimeError("Run sandbox creation on a POSIX host with Linux Docker containers")
    if os.getuid() == 0:
        raise RuntimeError("Run sandbox creation as a non-root user")
    if not shutil.which("docker"):
        raise RuntimeError("Docker is required on the execution host")
    _docker("info", "--format", "{{.ServerVersion}}")
    image_info = _resolve_image(task.image)
    image_id = image_info["Id"]
    image_path = "/usr/local/bin:/usr/bin:/bin"
    for variable in image_info["Config"].get("Env") or []:
        if variable.startswith("PATH="):
            image_path = variable[5:]
    directory = create_layout(task, destination, run_config=run_config or None)
    name = "autoastra-" + uuid.uuid4().hex
    task_directory = directory / "task"
    workspace = task_directory / "workspace"
    options = _container_options(config, image_path)
    if config.network_access:
        options[options.index('--network=none')] = '--network=bridge'
    command = [
        "create", "--name", name, "--label", "autoastra.managed=true",
        *options, *agent_mounts,
        "--workdir", "/task/workspace",
        "--mount", f"type=bind,src={task_directory},dst=/task,readonly,bind-recursive=disabled",
        "--mount", f"type=bind,src={workspace},dst=/task/workspace,bind-recursive=disabled",
        "--mount", f"type=bind,src={directory}/.runtime/ipc,dst=/run/autoastra,readonly,bind-recursive=disabled",
        "--mount", f"type=bind,src={directory}/.runtime/ledger.jsonl,dst=/task/ledger/ledger.jsonl,readonly",
        "--mount", f"type=bind,src={directory}/.runtime/snapshots,dst=/task/ledger/snapshots,readonly,bind-recursive=disabled",
        "--entrypoint", config.python_executable, image_id,
        "-I", "-c", "import time; time.sleep(31536000)",
    ]
    runtime = None
    try:
        runtime = Runtime(task, directory, image_id, image_path, config, run_config)
        container_id = _docker(*command).stdout.strip()
        _docker("start", container_id)
        sandbox = Sandbox(directory, container_id, image_id, config, runtime=runtime)
        probe = sandbox.run([config.python_executable, "-I", "-c", "from pathlib import Path; assert Path.cwd() == Path('/task/workspace')"])
        if probe.returncode:
            raise RuntimeError("The sandbox failed its startup check: " + probe.stderr.strip())
        return sandbox
    except BaseException:
        if runtime is not None:
            runtime.close()
        # The unique name also covers a lost reply from docker create.
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=30)
        # Only the directory exclusively created by this call is removed.
        for current, _, _ in os.walk(directory):
            Path(current).chmod(0o700)
        shutil.rmtree(directory)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="Import reference such as tasks.affine:task")
    parser.add_argument("--output", required=True, type=Path, help="New task directory on the execution host")
    parser.add_argument("--python-executable", default="python")
    gpu_selection = parser.add_mutually_exclusive_group()
    gpu_selection.add_argument("--gpus", default="0", help="GPU count or 'all'; default 0")
    gpu_selection.add_argument("--gpu-device", action="append", default=[], help="Device index or UUID; repeat to select several")
    parser.add_argument("--cpus", type=float, default=SandboxConfig.cpus)
    parser.add_argument("--memory-mb", type=int, default=SandboxConfig.memory_mb)
    parser.add_argument("--run-config", type=Path, help="Optional research run configuration JSON")
    parser.add_argument("--timeout", type=int, default=60, help="Command timeout in seconds")
    # Split explicitly so command flags are passed to the sandbox unchanged.
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv and any(flag in argv for flag in ("-h", "--help")):
        parser.parse_args(argv)
    if "--" not in argv:
        parser.error("Supply a command after --")
    separator = argv.index("--")
    args = parser.parse_args(argv[:separator])
    command = argv[separator + 1:]
    if not command:
        parser.error("Supply a command after --")
    try:
        module, delimiter, attribute = args.task.partition(":")
        if not delimiter or not module or not attribute:
            raise ValueError("Use a task reference such as tasks.affine:task")
        task = getattr(importlib.import_module(module), attribute)
        run_config = json.loads(args.run_config.read_text()) if args.run_config else None
        config = SandboxConfig(
            python_executable=args.python_executable,
            gpus="all" if args.gpus == "all" else int(args.gpus),
            gpu_devices=tuple(args.gpu_device), cpus=args.cpus,
            memory_mb=args.memory_mb, command_timeout_seconds=args.timeout,
        )
        with create_sandbox(task, args.output, config=config, run_config=run_config) as sandbox:
            result = sandbox.run(command)
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, AttributeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Sandbox operation failed. {error}\n")
