#!/usr/bin/env python3
"""Client installed under each of the three task/bin command names."""

import argparse
import base64
import json
from pathlib import Path
import socket
import sys

LIMIT = 4 * 1024 * 1024
WORKSPACE = Path('/task/workspace')


def source_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('Source must be inside /task/workspace')
    return path


def main():
    name = Path(sys.argv[0]).name
    parser = argparse.ArgumentParser(prog=name)
    if name == 'fitness-eval':
        parser.description = 'Evaluate an exact registered source revision on the development split.'
        parser.add_argument('source', nargs='?', default='final_solution.py')
        parser.add_argument('--attempt-id')
    elif name == 'experiment-ledger':
        parser.description = 'Record experiments and inspect immutable source and evaluation history.'
        commands = parser.add_subparsers(dest='action', required=True)
        register = commands.add_parser('register')
        register.add_argument('source', nargs='?', default='final_solution.py')
        for field in ('title', 'hypothesis', 'planned-change', 'success-criterion'):
            register.add_argument('--' + field, required=True)
        register.add_argument('--parent')
        select = commands.add_parser('select')
        select.add_argument('attempt_id')
        select.add_argument('--reason', required=True)
        commands.add_parser('history')
        inspect = commands.add_parser('inspect')
        inspect.add_argument('attempt_id')
        restore = commands.add_parser('restore')
        restore.add_argument('attempt_id')
        restore.add_argument('--output', default='final_solution.py')
        observe = commands.add_parser('observe')
        observe.add_argument('--attempt-id', required=True)
        observe.add_argument('--kind', choices=['exploratory'], default='exploratory')
        observe.add_argument('--summary', required=True)
        conclude = commands.add_parser('conclude')
        conclude.add_argument('--attempt-id', required=True)
        conclude.add_argument('--decision', choices=['keep', 'reject', 'follow_up'], required=True)
        conclude.add_argument('--summary', required=True)
    elif name == 'lifetime':
        parser.description = 'Show elapsed time, evaluation usage, and available budget accounting.'
    else:
        parser.error('Unknown command')
    args = vars(parser.parse_args())
    try:
        payload = {'command': name, **args}
        if 'source' in args:
            path = source_path(args['source'])
            with path.open('rb') as stream:
                source = stream.read(1024 * 1024 + 1)
            if len(source) > 1024 * 1024:
                raise ValueError('Source must be at most 1 MiB')
            source.decode('utf-8')
            payload['source'] = base64.b64encode(source).decode()
        output = source_path(args['output']) if args.get('action') == 'restore' else None
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3600)
            connection.connect('/run/autoastra/agent.sock')
            connection.sendall(json.dumps(payload, allow_nan=False).encode() + b'\n')
            with connection.makefile('rb') as stream:
                line = stream.readline(LIMIT + 1)
        if len(line) > LIMIT or not line.endswith(b'\n'):
            raise RuntimeError('Invalid service response')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['error'])
        result = response['result']
        if output is not None:
            output.write_bytes(base64.b64decode(result.pop('source'), validate=True))
            result['restored_to'] = str(output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result.get('status') == 'error' else 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'{name}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
