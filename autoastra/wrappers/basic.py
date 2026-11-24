"""Print a finished solution and fitness, and plot its official evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def print_result(sandbox_directory: str | Path) -> None:
    """Print saved source and fitness, then save the official fitness plot."""
    directory = Path(sandbox_directory)
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    source = (directory / 'task/workspace/final_solution.py').read_text(encoding='utf-8')
    best = result.get('selected', result.get('best'))
    score = best['fitness'] if best is not None else 'unavailable (no successful evaluation)'
    print(f'Development fitness: {score}')
    print('\nfinal_solution.py:\n')
    print(source, end='' if source.endswith('\n') else '\n')
    print(f'\nFitness plot: {plot_fitness(directory)}')


def plot_fitness(sandbox_directory: str | Path) -> Path:
    """Save fitness.png using only completed official evaluations in the ledger.

    Each evaluation is one point on the x-axis, labelled with its attempt ID.
    Re-evaluations of the same revision remain separate entries. Failed
    evaluations have no numeric score and leave a gap in the fitness line.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(sandbox_directory)
    ledger = directory / 'task/ledger/ledger.jsonl'
    records = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    evaluations = [record for record in records if record.get('type') == 'evaluation_finished']
    x = list(range(1, len(evaluations) + 1))
    scores = [record['fitness'] if record['status'] == 'ok' else float('nan')
              for record in evaluations]
    failed = [index for index, record in zip(x, evaluations) if record['status'] != 'ok']

    figure = Figure(figsize=(8, 4.5), dpi=150)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(x, scores, marker='o', color='#2563eb', linewidth=1.7, label='Official fitness')
    if failed:
        axes.scatter(failed, [0.04] * len(failed), transform=axes.get_xaxis_transform(),
                     marker='x', color='#b91c1c', label='Failed (no score)')
        axes.legend()
    if not any(record['status'] == 'ok' for record in evaluations):
        axes.text(0.5, 0.5, 'No successful official evaluations',
                  transform=axes.transAxes, ha='center', va='center')
    if evaluations:
        step = max(1, (len(x) + 11) // 12)
        ticks = sorted(set(x[::step] + [x[-1]]))
        axes.set_xticks(ticks)
        axes.set_xticklabels([evaluations[index - 1]['attempt_id'] for index in ticks])
        axes.set_xlim(0.5, len(x) + 0.5)
    else:
        axes.set_xticks([])
    axes.set(title='Official attempts vs fitness', xlabel='Official evaluation (attempt ID)',
             ylabel='Development fitness')
    axes.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    output = directory / 'fitness.png'
    figure.savefig(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sandbox_directory', type=Path, help='Finished sandbox output directory')
    args = parser.parse_args(argv)
    try:
        print_result(args.sandbox_directory)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Could not read the finished sandbox: {error}\n')


if __name__ == '__main__':
    main()
