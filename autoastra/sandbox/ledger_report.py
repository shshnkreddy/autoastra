"""Render a human-readable experiment ledger from the saved JSONL records."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import stat
import uuid


def _text(value):
    """Keep recorded prose readable without allowing it to break Markdown."""
    value = html.escape(' '.join(str(value).split()), quote=False)
    for character in ('\\', '`', '*', '_', '[', ']', '|', '#'):
        value = value.replace(character, '\\' + character)
    return value


def _elapsed(event):
    value = event.get('elapsed_seconds')
    return f'{value:.1f}s' if value is not None else 'Not recorded'


def render_ledger(events):
    """Summarize official outcomes and preserve every recorded research note."""
    start = next((event for event in events if event['type'] == 'run_started'), {})
    end = next((event for event in reversed(events)
                if event['type'] in ('research_finished', 'run_closed')), {})
    session = next((event for event in events if event['type'] in ('codex_session_started', 'claude_session_started', 'opencode_session_started')), {})
    spec = start.get('task_spec', {})
    attempts = [event for event in events if event['type'] == 'registered']
    evaluations = {}
    for event in events:
        if event['type'] in ('evaluation_started', 'evaluation_finished'):
            evaluations[event['evaluation_id']] = event

    title = spec.get('title', spec.get('name', 'Research run'))
    lines = [f'# Experiment ledger: {_text(title)}', '',
             'Generated from [ledger.jsonl](ledger.jsonl) after the run. '
             'Official fitness and exploratory observations are kept separate.', '',
             '## Run summary', '']
    status = end.get('status', 'closed' if end else 'unfinished')
    lines += [f'- **Status:** {_text(status.capitalize())}.']
    if end.get('stop_reason'):
        lines += [f'- **Stop reason:** {_text(end["stop_reason"].replace("_", " "))}.']
    if start.get('timestamp'):
        timestamp = datetime.fromisoformat(start['timestamp']).astimezone(timezone.utc)
        lines += [f'- **Started:** {timestamp:%Y-%m-%d %H:%M:%S} UTC.']
    if session:
        lines += [f'- **Model:** {_text(session.get("model", "Not recorded"))}.']
    if spec.get('metric'):
        preference = 'Lower' if spec.get('direction') == 'minimize' else 'Higher'
        label = 'Development score' if spec.get('development_proxy') else 'Metric'
        metric = spec.get('metric_label', spec['metric'].replace('_', ' '))
        lines += [f'- **{label}:** {_text(metric)}. {preference} is better.']
    lifetime = end.get('lifetime', {})
    if lifetime:
        lines += [f'- **Time used:** {_elapsed(lifetime)}.',
                  f'- **Official evaluations:** {len(evaluations)}.']
        tokens = lifetime.get('tokens_used')
        lines += [f'- **Reported tokens:** {tokens:,} (including cached input).' if tokens is not None
                  else '- **Reported tokens:** Not available.']
    best = end.get('best')
    if best:
        lines += [f'- **Best development fitness:** {best["fitness"]} '
                  f'({_text(best["attempt_id"])}, {_text(best["evaluation_id"])}).']
    elif any(event['status'] == 'ok' for event in evaluations.values()):
        lines += ['- **Best development fitness:** No final selection recorded.']
    else:
        lines += ['- **Best development fitness:** No successful official evaluation recorded.']
    selected = end.get('selected')
    if selected:
        lines.append(f'- **Selected attempt:** {_text(selected["attempt_id"])} '
                     f'({_text(end["selection"])}, development fitness {selected["fitness"]}).')
        if selected.get('reason'):
            lines.append(f'- **Selection reason:** {_text(selected["reason"])}.')
    if end.get('solution'):
        lines += ['- **Selected solution:** [final_solution.py](../workspace/final_solution.py).']
    if end.get('error'):
        error = end['error']
        if isinstance(error, dict):
            error = error.get('message', str(error))
        lines += ['', '**Run error:** ' + _text(error)]
    if not attempts:
        lines += ['', 'No attempts were registered.']

    for attempt in attempts:
        attempt_id = attempt['attempt_id']
        lines += ['', f'## {_text(attempt_id)}: {_text(attempt["title"])}', '',
                  f'Registered at {_elapsed(attempt)}. '
                  f'[Source snapshot](snapshots/{attempt_id}.py).']
        if attempt.get('parent'):
            lines += ['', '**Parent attempt:** ' + _text(attempt['parent'])]
        for key, label in (('hypothesis', 'Hypothesis'), ('planned_change', 'Change'),
                           ('success_criterion', 'Success criterion')):
            lines += ['', f'**{label}:** {_text(attempt[key])}']
        scores = [event for event in evaluations.values() if event['attempt_id'] == attempt_id]
        lines += ['', '### Official evaluations', '']
        if not scores:
            lines += ['This revision was not officially evaluated.']
        else:
            lines += ['| Evaluation | Result | Fitness | Elapsed time |',
                      '| --- | --- | --- | --- |']
            for score in scores:
                outcome = {'ok': 'Succeeded', 'error': 'Failed'}.get(score['status'], 'No result recorded')
                fitness = str(score['fitness']) if score['status'] == 'ok' else '—'
                lines += [f'| {_text(score["evaluation_id"])} | {outcome} | {fitness} | {_elapsed(score)} |']
            for score in scores:
                if score.get('error'):
                    lines += ['', f'**{_text(score["evaluation_id"])} error:** {_text(score["error"])}']
        notes = [event for event in events if event.get('attempt_id') == attempt_id]
        for note in notes:
            if note['type'] == 'observation':
                lines += ['', f'### Exploratory observation ({_elapsed(note)})', '', _text(note['summary'])]
            elif note['type'] == 'conclusion':
                decision = note['decision'].replace('_', ' ')
                lines += ['', f'### Conclusion: {_text(decision)} ({_elapsed(note)})', '', _text(note['summary'])]
            elif note['type'] == 'restore_requested':
                lines += ['', f'Restoration of this revision was requested at {_elapsed(note)}.']
        if not any(note['type'] == 'conclusion' for note in notes):
            lines += ['', 'No conclusion was recorded for this attempt.']

    lines += ['', '## Provenance', '',
              'The [JSONL ledger](ledger.jsonl) retains the full event sequence, timestamps, '
              'source hashes, evaluation settings, and hash chain. '
              'The source snapshots above identify the exact revisions evaluated. '
              'This report renders those records without adding new evaluations or interpretations.', '']
    return '\n'.join(lines)


def write_ledger_report(sandbox_directory):
    """Create or refresh task/ledger/ledger.md without changing the JSONL ledger."""
    directory = Path(sandbox_directory) / 'task/ledger'
    events = [json.loads(line) for line in (directory / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()
              if line.strip()]
    report = render_ledger(events)
    mode = stat.S_IMODE(directory.stat().st_mode)
    temporary = directory / ('.' + uuid.uuid4().hex)
    target = directory / 'ledger.md'
    directory.chmod(mode | stat.S_IWUSR)
    try:
        temporary.write_text(report, encoding='utf-8')
        temporary.chmod(0o444)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
        directory.chmod(mode)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sandbox_directory', type=Path)
    args = parser.parse_args()
    try:
        print(write_ledger_report(args.sandbox_directory))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Could not render the ledger: {error}\n')
