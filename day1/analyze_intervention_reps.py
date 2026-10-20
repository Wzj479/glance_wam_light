"""Intervention outcomes over repeated runs, against the no-intervention baseline.

An intervention verdict drawn from one run is the same mistake as drawing a
model attribution from one run: a single closed-loop episode flips. This script
reads a directory of repeated intervention runs (one directory per run, named
`state<N>_<model>_r<rep>`), counts how often the intervention succeeded, and
compares that with the no-intervention success count for the same state.

The three verdicts are deliberately narrow:

  recovered      the intervention succeeds where the baseline always failed
  unchanged      the intervention leaves the baseline outcome alone
  damaged        the intervention turns a baseline success into a failure

`mixed` covers everything else, including an intervention that improves an
unstable baseline: without an always-failing baseline there is nothing for the
intervention to recover, so the comparison is not well defined.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RUN_PATTERN = re.compile(r'state(?P<state>\d+)_(?P<model>[A-Za-z][A-Za-z0-9_]*)_r(?P<rep>\d+)$')


def read_run(run_dir):
    """Read one intervention run's driver results."""
    path = Path(run_dir) / 'driver' / 'results.json'
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    episodes = payload.get('episode_results') or []
    if not episodes:
        return None
    record = episodes[0]
    return dict(
        episode_id=int(record['episode_id']),
        steps=int(record['steps']),
        success=bool(record['success']),
        overridden_steps=record.get('gripper_overridden_steps'),
    )


def collect(root):
    """Group intervention runs by (episode, model)."""
    root = Path(root)
    grouped = defaultdict(dict)
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        match = RUN_PATTERN.search(path.name)
        if not match:
            continue
        run = read_run(path)
        if run is None:
            continue
        key = (int(match['state']), match['model'])
        grouped[key][int(match['rep'])] = run
    return grouped


def verdict(successes, runs, baseline_successes, baseline_runs):
    """Compare intervention outcomes with the no-intervention baseline."""
    if runs == 0:
        return 'no_data'
    if baseline_runs and baseline_successes == 0:
        if successes == runs:
            return 'recovered'
        if successes == 0:
            return 'still_failing'
        return 'partial'
    if baseline_runs and baseline_successes == baseline_runs:
        if successes == runs:
            return 'unchanged'
        if successes == 0:
            return 'damaged'
        return 'mixed'
    return 'baseline_unstable'


def summarise(grouped, baseline=None):
    baseline = baseline or {}
    out = {}
    for (episode, model), reps in sorted(grouped.items()):
        runs = [reps[rep] for rep in sorted(reps)]
        successes = sum(1 for r in runs if r['success'])
        base = (baseline.get(str(episode)) or {}).get(model) or {}
        base_successes = base.get('successes')
        base_runs = base.get('runs')
        out[f'{episode}:{model}'] = dict(
            episode_id=episode,
            model=model,
            runs=len(runs),
            successes=successes,
            success_rate=successes / len(runs),
            steps=[r['steps'] for r in runs],
            overridden_steps=sorted({r['overridden_steps'] for r in runs
                                     if r['overridden_steps'] is not None}),
            baseline_successes=base_successes,
            baseline_runs=base_runs,
            verdict=verdict(successes, len(runs), base_successes, base_runs),
        )
    return out


def load_baseline(path):
    """Per-(episode, model) no-intervention success counts from a flip-rate report."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    out = {}
    for model, episodes in (payload.get('episodes') or {}).items():
        for episode, value in episodes.items():
            out.setdefault(str(episode), {})[model] = dict(
                successes=value['successes'], runs=value['runs'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, help='Directory of repeated intervention runs')
    parser.add_argument('--baseline', help='flip-rate report for the same states, no intervention')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    grouped = collect(args.runs)
    baseline = load_baseline(args.baseline)
    report = dict(
        runs_root=str(args.runs),
        baseline=str(args.baseline) if args.baseline else None,
        results=summarise(grouped, baseline),
    )
    report['verdict_counts'] = {}
    for value in report['results'].values():
        report['verdict_counts'][value['verdict']] = \
            report['verdict_counts'].get(value['verdict'], 0) + 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
