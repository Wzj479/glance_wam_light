"""Failure mechanism per (state, model), from a physics replay of saved actions.

The replay writes one directory per case and model with the replayed actions
(`replay.npz`), the per-state physics (`states.csv`) and a metadata file. This
script turns those into the facts a mechanism claim needs -- when the gripper
closed and opened, whether the bowl was ever between both finger pads, how far
it was lifted, and where it ended up -- and then labels the mechanism.

The labels are deliberately factual rather than narrative: a claim such as
"the gripper released before the grasp settled" is only supportable if the
release action precedes the first both-pad contact, and that ordering is
recomputed from the logs here rather than taken on trust.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

LIFT_MARKER_MM = 20.0
PLATE_TOLERANCE_M = 0.03


def read_states(source):
    """Read states.csv into a list of dicts with typed values."""
    path = Path(source)
    rows = []
    with path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            typed = {}
            for key, value in row.items():
                if value in ('True', 'False'):
                    typed[key] = value == 'True'
                elif key.endswith(('_mm', '_m', '_deg')) or key in (
                        'state_step', 'time_s', 'eef_x', 'eef_y', 'eef_z',
                        'bowl_x', 'bowl_y', 'bowl_z', 'plate_x', 'plate_y', 'plate_z'):
                    try:
                        typed[key] = float(value)
                    except ValueError:
                        typed[key] = value
                else:
                    typed[key] = value
            rows.append(typed)
    return rows


def gripper_events(actions):
    """First close and first release action index from the gripper channel.

    The channel is binary (+1 closed, -1 open), so a transition is the first
    index where the sign changes.
    """
    grip = np.asarray(actions)[:, 6]
    closed = grip > 0
    first_close = None
    first_release = None
    for index in range(1, len(grip)):
        if first_close is None and closed[index] and not closed[index - 1]:
            first_close = index
        if first_release is None and not closed[index] and closed[index - 1]:
            first_release = index
    return first_close, first_release


def mechanism_facts(states, actions):
    """Per-episode facts and the mechanism label."""
    bowl_z = np.asarray([row['bowl_z'] for row in states], dtype=float)
    lift_mm = (bowl_z - bowl_z[0]) * 1000.0
    both = np.asarray([bool(row['both_pad_contact']) for row in states])
    any_finger = np.asarray([bool(row['any_finger_contact']) for row in states])
    first_close, first_release = gripper_events(actions)

    contact_states = np.flatnonzero(both)
    first_contact = int(contact_states.min()) if len(contact_states) else None
    last_contact = int(contact_states.max()) if len(contact_states) else None
    lifted = np.flatnonzero(lift_mm > LIFT_MARKER_MM)
    first_lift = int(lifted.min()) if len(lifted) else None

    release_state = None
    if first_release is not None:
        release_state = min(first_release, len(states) - 1)
    contact_at_release = bool(both[release_state]) if release_state is not None else None
    contact_after_release = bool(any(both[release_state:])) if release_state is not None else None

    facts = dict(
        steps=len(states),
        first_close_action=first_close,
        first_release_action=first_release,
        first_both_pad_contact_state=first_contact,
        last_both_pad_contact_state=last_contact,
        max_lift_mm=float(lift_mm.max()),
        first_lift_20mm_state=first_lift,
        contact_at_release=contact_at_release,
        any_pad_contact_after_release=contact_after_release,
        finger_contact_states=int(any_finger.sum()),
        final_xy_mm=float(states[-1]['xy_distance_m']) * 1000.0,
        final_bowl_above_plate=bool(states[-1]['above_plate']),
        final_bowl_plate_contact=bool(states[-1]['bowl_plate_contact']),
        success=bool(states[-1]['success']),
    )

    if facts['success']:
        label = 'success'
    elif first_release is not None and (first_contact is None or first_release <= first_contact):
        # The gripper opened before the bowl was ever between both pads: a release
        # that happened without a grasp, whether or not a grasp was attempted.
        label = 'released_before_both_pad_contact'
    elif first_contact is None:
        # The gripper never opened and the bowl was never between both pads.
        label = 'never_both_pad_contact'
    elif first_lift is None:
        label = 'grasped_never_lifted'
    elif (not facts['final_bowl_plate_contact'] and last_contact is not None
            and last_contact < len(states) - 1):
        label = 'lifted_then_lost_contact'
    elif facts['final_xy_mm'] > PLATE_TOLERANCE_M * 1000.0:
        label = 'placed_but_misaligned'
    else:
        label = 'other'
    facts['mechanism'] = label
    return facts


def collect(run_dir):
    """Read every (case, model) replay under a replay output directory.

    The replay writes one `replay_plan.json` for the whole run and one
    directory per case named `<condition>_episode<NNN>`, each holding a
    directory per model.
    """
    root = Path(run_dir)
    plan_path = root / 'replay_plan.json'
    if not plan_path.is_file():
        return {}
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    # The case directories are named `<condition>_episode<N>` without zero padding,
    # so they are matched by prefix and the number is parsed back out.
    directories = {}
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        marker = '_episode'
        if marker not in path.name:
            continue
        condition, _, number = path.name.rpartition(marker)
        if not number.isdigit():
            continue
        directories[(condition, int(number))] = path
    out = {}
    for condition, episode in plan['cases']:
        case_dir = directories.get((condition, int(episode)))
        if case_dir is None or not case_dir.is_dir():
            continue
        for model_dir in sorted(p for p in case_dir.iterdir() if p.is_dir()):
            states_path = model_dir / 'states.csv'
            replay_path = model_dir / 'replay.npz'
            if not (states_path.is_file() and replay_path.is_file()):
                continue
            actions = np.load(replay_path)['actions']
            facts = mechanism_facts(read_states(states_path), actions)
            out[(condition, int(episode), model_dir.name)] = facts
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, help='Replay output directory')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    facts = collect(args.replay)
    report = {
        'cases': sorted({(c, e) for c, e, _ in facts}),
        'models': sorted({m for _, _, m in facts}),
        'mechanisms': {
            f'{condition}:{episode}:{model}': value
            for (condition, episode, model), value in sorted(facts.items())
        },
        'label_counts': {},
    }
    for value in facts.values():
        label = value['mechanism']
        report['label_counts'][label] = report['label_counts'].get(label, 0) + 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
