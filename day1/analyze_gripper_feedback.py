"""Decide the state33 gripper-intervention diagnostic from its recorded logs.

Reads only files written by diagnose_gripper_feedback.py. No environment, model or
physics execution, so the verdict is reproducible off the GPU host.

The question is narrow: does holding the gripper closed through the window let the
bowl be transported, or not. The archived run never lifted the bowl, so transport is
judged on end-anchored contact and lift rather than on the fingerpad proxy that the
offline review already showed to be inadequate.

Two guards keep the reading honest. The zero-intervention control must reproduce the
archived actions before any difference is attributed to the gripper override, and the
divergence step is reported next to the intervention window because the window's first
action block is shared with the archived run while later blocks are re-queried.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# The archived run never raised the bowl past this marker and a successful transport
# must. Fixing the bar here avoids re-tuning it against the result.
LIFT_MARKER_MM = 20.0


def read_csv(path):
    # A run with nothing to report in a channel writes no file, so absence is normal
    # rather than an error; reporting it as an empty channel keeps the reader usable.
    path = Path(path)
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def driver_dir(run_dir):
    """Locate the driver output; the runner nests it one level down."""
    run_dir = Path(run_dir)
    return run_dir / 'driver' if (run_dir / 'driver').exists() else run_dir


def booleans(rows, name):
    """Accept both CSV text and native bools so the same code serves logs and tests."""
    return np.asarray([row[name] if isinstance(row[name], bool) else row[name] == 'True' for row in rows])


def numbers(rows, name):
    return np.asarray([float(row[name]) for row in rows])


def gripper_commands(rows, name):
    """Gripper channel as integers, from CSV text or native numbers.

    The writer stores the channel as a float, so a round trip through the CSV
    turns -1 into the text '-1.0'; int() on that text raises. The channel is
    binary by construction -- override_gripper rejects anything else -- so
    rounding recovers the intended value and a value outside {-1, 1} means the
    log and the run disagree.
    """
    values = []
    for row in rows:
        raw = row[name]
        value = int(round(float(raw)))
        if value not in (-1, 1):
            raise ValueError(f'{name} must be -1 or +1, got {raw!r}')
        values.append(value)
    return values


def contact_tail(flags):
    """(first, last, trailing contact-free states). Independent of where an early gap fell."""
    index = np.flatnonzero(flags)
    if not len(index):
        return None, None, None
    return int(index.min()), int(index.max()), int(len(flags) - 1 - index.max())


def analyse_run(run_dir):
    run_dir = driver_dir(run_dir)
    results = json.loads((run_dir / 'results.json').read_text())
    episode = results['episode_results'][0]
    stem = f'task_{episode["task_id"]:02d}_episode_{episode["episode_id"]:03d}'
    rows = read_csv(run_dir / f'{stem}_state_log.csv')
    override = read_csv(run_dir / f'{stem}_override_log.csv')
    chunk = read_csv(run_dir / f'{stem}_chunk_log.csv')
    flagged = booleans(override, 'overridden')
    overridden = [int(r['action_index']) for r, flag in zip(override, flagged) if flag]
    model_commands = set(gripper_commands(override, 'model_gripper_command'))
    grasp = grasp_profile(rows)
    overlay = overlay_profile(rows)
    transported = grasp['reached_lift_marker_before_last_finger_contact']
    return dict(
        model=results['model'], condition=results['condition'],
        hold_close_inclusive=results['hold_close_inclusive'],
        protocol_sha256=results['protocol_sha256'],
        steps=episode['steps'], success=episode['success'],
        action_sha256=episode['action_sha256'],
        init_state_sha256=episode['init_state_sha256'],
        post_settle_state_sha256=episode['post_settle_state_sha256'],
        divergence_step_vs_reference=episode.get('divergence_step_vs_reference'),
        gripper_overridden_steps=len(overridden),
        gripper_override_first_step=overridden[0] if overridden else None,
        gripper_override_last_step=overridden[-1] if overridden else None,
        model_gripper_commands_seen=sorted(model_commands),
        queries=len(chunk),
        goal_sources=sorted({r['goal_source'] for r in chunk}),
        goal_refresh_steps=[int(r['query_step']) for r in chunk if r['goal_source'] == 'generate'],
        **grasp, **overlay, **slip_profile(rows),
        # The environment's termination flag is authoritative; the state log derives
        # the same predicate independently, so disagreement is worth surfacing.
        success_predicates_agree=bool(episode['success'] == overlay['state_log_success']),
        verdict='transport_restored' if transported else 'no_transport_under_intervention',
        verdict_basis=('bowl cleared the 20mm lift marker while gripper geometry was still in contact'
                       if transported else
                       'bowl never cleared the 20mm lift marker before gripper contact was lost'),
        # Transport and placement are separate axes. The 39/aug replay lifted the bowl
        # 100mm and still failed, so a restored transport must never be read as success.
        outcome=('transported_and_placed' if transported and episode['success'] else
                 'transported_not_placed' if transported else
                 'not_transported'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Output directory of run_gripper_feedback.py')
    parser.add_argument('--control-run', help='Zero-intervention control run directory')
    parser.add_argument('--reference-actions', help='Archived actions.npy for the same state')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = analyse_run(args.run)
    if args.control_run or args.reference_actions:
        report['control'] = actions_match(args.control_run, args.reference_actions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def grasp_profile(rows):
    """End-anchored transport evidence: lift reached while contact was still held."""
    z = numbers(rows, 'bowl_z')
    lift = (z - z[0]) * 1000
    any_finger = booleans(rows, 'any_finger_contact')
    first, last, tail = contact_tail(any_finger)
    reached = np.flatnonzero(lift > LIFT_MARKER_MM)
    regions = int(np.count_nonzero(any_finger[1:] & ~any_finger[:-1]) + (1 if any_finger[0] else 0))
    return dict(
        max_lift_mm=float(lift.max()),
        lift_at_last_finger_contact_mm=float(lift[last]) if last is not None else None,
        first_state_above_lift_marker=int(reached[0]) if len(reached) else None,
        reached_lift_marker_before_last_finger_contact=(
            bool(len(reached) and last is not None and int(reached[0]) <= last)),
        finger_contact_states=int(any_finger.sum()),
        finger_contact_regions=regions,
        first_finger_contact_state=first, last_finger_contact_state=last,
        trailing_contact_free_states=tail)


def overlay_profile(rows):
    plate = np.asarray([[float(row[f'plate_{a}']) for a in 'xyz'] for row in rows])
    bowl = np.asarray([[float(row[f'bowl_{a}']) for a in 'xyz'] for row in rows])
    xy = np.linalg.norm(bowl[:, :2] - plate[:, :2], axis=1) * 1000
    return dict(min_xy_mm=float(xy.min()), final_xy_mm=float(xy[-1]),
                final_within_30mm=bool(xy[-1] < 30),
                final_bowl_above_plate=bool(bowl[-1, 2] >= plate[-1, 2]),
                final_bowl_plate_contact=bool(booleans(rows, 'bowl_plate_contact')[-1]),
                state_log_success=bool(booleans(rows, 'success')[-1]))


def slip_profile(rows):
    logged = numbers(rows, 'relative_slip_mm')
    return dict(max_relative_slip_mm=float(logged.max()), mean_relative_slip_mm=float(logged.mean()))


def actions_match(run_dir, reference_path):
    """True only when a run reproduces the archived actions bit for bit.

    The control run is the gate for the whole experiment: if a zero-intervention run
    cannot reproduce the archive, a difference in the intervention run cannot be
    attributed to the gripper override.
    """
    if reference_path is None:
        return None
    produced = sorted(driver_dir(run_dir).glob('*_actions.npy'))
    if not produced:
        raise FileNotFoundError(f'No actions.npy under {driver_dir(run_dir)}')
    produced = produced[0]
    left = np.load(produced).astype(np.float64)
    right = np.load(reference_path).astype(np.float64)
    if left.shape != right.shape:
        return dict(reproduced=False, reason='step_count_differs',
                    produced_steps=int(left.shape[0]), reference_steps=int(right.shape[0]))
    identical = bool(np.array_equal(left, right))
    result = dict(reproduced=identical, produced_steps=int(left.shape[0]))
    if not identical:
        step = first_divergence_step(left, right)
        result.update(reason='actions_differ', first_divergence_step=step,
                      max_abs_difference_mm=float(np.abs(left - right).max()))
    return result


def first_divergence_step(actions, reference):
    for step in range(min(len(actions), len(reference))):
        if not np.array_equal(actions[step], reference[step]):
            return int(step)
    return None


if __name__ == '__main__':
    main()
