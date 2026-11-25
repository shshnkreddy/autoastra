"""Run a task with one supervised coding agent session."""

import argparse
import importlib
import json
from pathlib import Path
import sys

from . import RunnerConfig, run, run_seeds
from ..sandbox import SandboxConfig
from ..sandbox.runtime import validate_run_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', help='Import reference, such as tasks.affine:task')
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory to retain the complete finished sandbox')
    parser.add_argument('--model')
    parser.add_argument('--provider', choices=('codex', 'openrouter', 'anthropic', 'opencode'), default='codex')
    parser.add_argument('--opencode-binary', type=Path, help='Native Linux OpenCode executable on the execution host')
    parser.add_argument('--claude-binary', type=Path, help='Native Linux Claude Code executable on the execution host')
    parser.add_argument('--codex-binary', type=Path, help='Native Linux Codex executable on the execution host')
    parser.add_argument('--codex-home', type=Path, help='Dedicated Codex authentication and state directory')
    parser.add_argument('--run-config', type=Path, help='JSON overrides for the task run configuration')
    parser.add_argument('--time-seconds', type=int, help='Override the task time limit')
    parser.add_argument('--max-evaluations', type=int, help='Override the task evaluation limit')
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--evaluation-timeout-seconds', type=int)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seeds', nargs='+', type=int, help='Launch one independent run per seed')
    parser.add_argument('--jobs', type=int, default=1, help='Maximum concurrent seed sessions (default: 1)')
    parser.add_argument('--paired-paraphrases', action='store_true',
                        help='Pair successive seeds with p0 through p5 from the task module')
    parser.add_argument('--selection', choices=('best-fitness', 'model'))
    parser.add_argument('--cpus', type=float, default=1)
    parser.add_argument('--memory-mb', type=int, default=2048)
    parser.add_argument('--allow-internet', action='store_true',
                        help='Allow general development internet access instead of model-provider connections only')
    parser.add_argument('--model-host', action='append', default=[],
                        help='Replace the provider domain allowlist with these exact HTTPS hosts (repeatable)')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--gpus', default='0')
    selection.add_argument('--gpu-device', action='append', default=[])
    args = parser.parse_args(argv)
    try:
        module, separator, attribute = args.task.partition(':')
        if not separator or not module or not attribute:
            raise ValueError('Use a task reference such as tasks.affine:task')
        task = getattr(importlib.import_module(module), attribute)
        limits = validate_run_config(json.loads(args.run_config.read_text())) if args.run_config else {}
        limits.update({name: getattr(args, name) for name in (
            'time_seconds', 'max_evaluations', 'max_tokens', 'evaluation_timeout_seconds', 'seed'
        ) if getattr(args, name) is not None})
        if args.selection is not None:
            limits['selection'] = args.selection
        config = RunnerConfig(
            model=args.model, provider=args.provider, codex_binary=args.codex_binary,
            codex_home=args.codex_home, claude_binary=args.claude_binary, opencode_binary=args.opencode_binary,
        )
        resources = SandboxConfig(
            cpus=args.cpus, memory_mb=args.memory_mb, pids_limit=256, tmp_mb=256,
            network_access=args.allow_internet,
            model_hosts=tuple(args.model_host),
            gpus='all' if args.gpus == 'all' else int(args.gpus), gpu_devices=tuple(args.gpu_device),
        )
        if args.jobs < 1 or (args.jobs > 1 and not args.seeds):
            raise ValueError('--jobs must be positive and parallel runs require --seeds')
        if args.paired_paraphrases and (args.seeds is None or len(args.seeds) != 6):
            raise ValueError('--paired-paraphrases requires six seeds in pairing order')
        if args.seeds:
            if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
                raise ValueError('Supply distinct nonnegative seeds')
            module_obj = importlib.import_module(module)
            tasks = [getattr(module_obj, f'p{i}') if args.paired_paraphrases else task
                     for i in range(len(args.seeds))]
            success = True
            for _, result in run_seeds(dict(zip(args.seeds, tasks)), args.output, jobs=args.jobs,
                                       config=config, sandbox_config=resources, run_config=limits):
                print(result['sandbox'], flush=True)
                if result['error']:
                    print(result['error'], file=sys.stderr)
                success &= bool(result['solution']) and result['status'] != 'failed'
                if result['stop_reason'] == 'interrupted':
                    return 130
            return 0 if success else 1
        result = run(task, args.output, config=config, sandbox_config=resources, run_config=limits)
        print(result['sandbox'])
        if result['error']:
            print('Autoresearch: ' + str(result['error']), file=sys.stderr)
        return 0 if result['solution'] and result['status'] != 'failed' else 1
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, AttributeError) as error:
        parser.exit(1, f'Autoresearch failed. {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
