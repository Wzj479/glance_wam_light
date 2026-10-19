"""Classify archived closed-loop failures by which models fail them, kinematically.

Reads only results.json and actions.npy from finished evaluations. No environment,
model or physics execution, so it runs without a GPU and cannot perturb anything.

The point is attribution. A state that every model fails is the base policy being
fragile under the colour shift, not a regression caused by fine-tuning. A state only
one model fails is the only place a fine-tuning effect can be claimed at all, and
which states those are determines what a follow-up diagnostic is worth running on.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Command magnitudes follow the action parameterisation, whose physical scale was
# not verified here. Only relative comparisons between models are reported.
UNIT_NOTE = ('Magnitudes are in the action parameterisation of world_vector, not a '
             'verified physical distance; only relative comparisons are meaningful.')


def read_results(source, model, condition):
    path = Path(source) / model / condition / 'results.json'
    return json.loads(path.read_text())


def action_path(source, model, condition, task_id, episode_id):
    return Path(source) / model / condition / f'task_{task_id:02d}_episode_{episode_id:03d}_actions.npy'


def gripper_timeline(actions):
    """Indices where the gripper command flips; +1 close, -1 open."""
    grip = actions[:, 6]
    close = [int(i) for i in range(1, len(grip)) if grip[i] > 0 and grip[i - 1] < 0]
    opened = [int(i) for i in range(1, len(grip)) if grip[i] < 0 and grip[i - 1] > 0]
    return close, opened


def motion_profile(actions):
    """Median and peak per-step translation command magnitude."""
    steps = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    if not len(steps):
        return 0.0, 0.0
    return float(np.median(steps) * 1000), float(steps.max() * 1000)


def collect(source, models, conditions):
    """Per-episode records plus the (condition, episode) -> failing models map."""
    episodes, failures = [], defaultdict(list)
    for model in models:
        for condition in conditions:
            results = read_results(source, model, condition)
            for record in results['episode_results']:
                actions = np.load(action_path(source, model, condition,
                                              record['task_id'], record['episode_id']))
                close, opened = gripper_timeline(actions)
                median, peak = motion_profile(actions)
                episodes.append(dict(model=model, condition=condition,
                                     task_id=record['task_id'], episode_id=record['episode_id'],
                                     success=bool(record['success']), steps=int(record['steps']),
                                     gripper_close_steps=close, gripper_open_steps=opened,
                                     median_command_mag=median, peak_command_mag=peak))
                if not record['success']:
                    failures[(condition, record['episode_id'])].append(model)
    return episodes, failures


def attribute(failures, models):
    """Split failing states into shared across all models vs model-specific."""
    shared, specific = [], {}
    for (condition, episode), failed in sorted(failures.items()):
        if len(failed) == len(models):
            shared.append(dict(condition=condition, episode_id=episode, models=failed))
        else:
            specific[(condition, episode)] = sorted(failed)
    return shared, specific


def motion_separation(episodes):
    """Compare failure and success motion magnitudes inside each model/condition cell.

    Reported per cell rather than pooled, because pooling would let one condition
    dominate. A cell with too few episodes on either side is listed as undecided.
    """
    cells, pooled = [], {True: [], False: []}
    groups = defaultdict(list)
    for row in episodes:
        groups[(row['model'], row['condition'])].append(row)
    for (model, condition), rows in sorted(groups.items()):
        good = [r for r in rows if r['success']]
        bad = [r for r in rows if not r['success']]
        pooled[True].extend(good)
        pooled[False].extend(bad)
        cell = dict(model=model, condition=condition, successes=len(good), failures=len(bad))
        if len(bad) >= 2 and len(good) >= 2:
            cell.update(separated=bool(
                np.mean([r['median_command_mag'] for r in bad]) <
                np.mean([r['median_command_mag'] for r in good]) and
                np.mean([r['peak_command_mag'] for r in bad]) >
                np.mean([r['peak_command_mag'] for r in good])))
        else:
            cell['separated'] = None
            cell['undecided_because'] = 'fewer than two episodes on one side'
        cells.append(cell)
    overall = dict(
        successes=len(pooled[True]), failures=len(pooled[False]),
        success_median_command_mag=float(np.mean([r['median_command_mag'] for r in pooled[True]])) if pooled[True] else None,
        failure_median_command_mag=float(np.mean([r['median_command_mag'] for r in pooled[False]])) if pooled[False] else None,
        success_peak_command_mag=float(np.mean([r['peak_command_mag'] for r in pooled[True]])) if pooled[True] else None,
        failure_peak_command_mag=float(np.mean([r['peak_command_mag'] for r in pooled[False]])) if pooled[False] else None)
    return cells, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Archived evaluation directory')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--conditions', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    episodes, failures = collect(args.source, args.models, args.conditions)
    shared, specific = attribute(failures, args.models)
    cells, overall = motion_separation(episodes)
    report = dict(label=args.label, source=str(args.source), models=args.models,
                  conditions=args.conditions, unit_note=UNIT_NOTE,
                  episodes=episodes, failures_by_state={f'{c}:{e}': m for (c, e), m in sorted(failures.items())},
                  shared_failures=shared,
                  model_specific_failures={f'{c}:{e}': m for (c, e), m in sorted(specific.items())},
                  motion_cells=cells, motion_overall=overall,
                  unattributed_note=('A state failing for every model is base-policy fragility, not a '
                                     'fine-tuning effect. A state failing for one model is the only '
                                     'candidate regression, and each is a single episode.'))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()