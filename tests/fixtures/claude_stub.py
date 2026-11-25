#!/usr/local/bin/python
"""Deterministic Claude stream-json peer for container tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def emit(**event):
    print(json.dumps(event), flush=True)


assert '-p' in sys.argv and 'stream-json' in sys.argv
assert '/task/CLAUDE.md' in sys.argv[-1]
assert Path('/task/CLAUDE.md').stat().st_mode & 0o777 == 0o444
assert not Path('/task/AGENTS.md').exists()
assert os.environ['ANTHROPIC_API_KEY'] == 'test-only'
model = sys.argv[sys.argv.index('--model') + 1]
emit(type='system', subtype='init', model=model, session_id='claude-test')
for _ in range(2):
    emit(type='assistant', message={'id': 'a', 'usage': {'input_tokens': 10, 'cache_read_input_tokens': 5}})
if model == 'idle':
    time.sleep(60)
if model == 'failure':
    emit(type='result', subtype='error_during_execution', is_error=True, errors=['test failure'])
    sys.exit(1)
Path('final_solution.py').write_text('def predict(xs):\n    return [2*x+1 for x in xs]\n')
subprocess.run(['experiment-ledger', 'register', '--title', 'Affine', '--hypothesis', 'Linear',
                '--planned-change', 'Fit slope', '--success-criterion', 'Zero error'], check=True, capture_output=True)
score = subprocess.check_output(['fitness-eval'], text=True)
assert json.loads(score)['fitness'] == 0
emit(type='result', subtype='success', is_error=False, usage={'input_tokens': 10,
     'cache_read_input_tokens': 5, 'output_tokens': 4}, session_id='claude-test')
