"""Entrypoint for a temporary container evaluating one frozen solution."""

import contextlib
import importlib
import json
import math
import numbers
from pathlib import Path
import sys


def main():
    try:
        config = json.loads(Path('/input/request.json').read_text())
        with contextlib.redirect_stdout(sys.stderr):
            sys.path[:0] = ['/runtime', '/evaluator/code']
            module = importlib.import_module(config['module'])
            function = module
            for name in config['qualname'].split('.'):
                function = getattr(function, name)
            result = function(Path('/submission/final_solution.py'), split='dev', seed=config['seed'])
            if isinstance(result, bool) or not isinstance(result, numbers.Real) or not math.isfinite(result):
                raise ValueError('Fitness must be a finite number')
            result = float(result)
        response = {'ok': True, 'result': result}
        encoded = json.dumps(response, allow_nan=False)
        if len(encoded.encode()) > 1024 * 1024:
            raise ValueError('Result exceeds 1 MiB')
    except BaseException as error:
        encoded = json.dumps({'ok': False, 'error': type(error).__name__ + ': ' + str(error)[:2000]})
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
