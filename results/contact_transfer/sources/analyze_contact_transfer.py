"""Contact-transfer and relative-slip metrics from saved-action replay logs.

Reads the CSV traces written by replay_failure_cases.py. No environment, model or
physics execution, so it also covers replays produced before the finger-level
contact columns existed: those runs still yield pad transitions and slip, and the
finger columns are reported only when present.

The earlier reports treated a lost fingerpad pair as a lost grasp. The counterfactual
state inspection showed contact migrating to the non-pad finger geoms, so this
script separates the two and never infers object loss from pad loss alone.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAD_COLUMNS = ('left_pad_contact', 'right_pad_contact', 'both_pad_contact')
FINGER_COLUMNS = ('left_finger_contact', 'right_finger_contact', 'any_finger_contact',
                  'finger_contact_without_both_pads')
# Traces end at each model's original stop time, so whole-trace averages mix
# different horizons. This leading window is common to every compared trace.
WINDOW_STATES = 100


def read_states(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty trace: {path}')
    return rows


def column(rows, name):
    if name not in rows[0]:
        return None
    if rows[0][name] in ('True', 'False'):
        return np.asarray([row[name] == 'True' for row in rows])
    return np.asarray([float(row[name]) for row in rows])


def eef_relative_offset(rows):
    """Bowl body origin expressed in the EEF frame, in metres."""
    eef = np.asarray([[float(row[f'eef_{axis}']) for axis in 'xyz'] for row in rows])
    bowl = np.asarray([[float(row[f'bowl_{axis}']) for axis in 'xyz'] for row in rows])
    quats = np.asarray([[float(row[f'eef_q{axis}']) for axis in 'xyzw'] for row in rows])
    return np.einsum('nji,nj->ni', Rotation.from_quat(quats).as_matrix(), bowl - eef)


def first_true(flags):
    return int(np.flatnonzero(flags)[0]) if np.any(flags) else None


def first_falling_edge(flags):
    edges = np.flatnonzero(flags[:-1] & ~flags[1:]) + 1
    return int(edges[0]) if len(edges) else None


def contact_regions(flags):
    """Contiguous contact intervals; contact commonly returns after an early gap."""
    return int(np.count_nonzero(flags[1:] & ~flags[:-1]) + (1 if flags[0] else 0))


def analyze(path):
    rows = read_states(path)
    offsets = eef_relative_offset(rows)
    slip = np.zeros(len(rows))
    slip[1:] = np.linalg.norm(np.diff(offsets, axis=0), axis=1) * 1000
    logged = column(rows, 'relative_slip_mm')

    both_pads = column(rows, 'both_pad_contact')
    any_pad = column(rows, 'left_pad_contact') | column(rows, 'right_pad_contact')
    lift = np.asarray([float(row['bowl_z']) for row in rows])
    lift = (lift - lift[0]) * 1000
    metrics = dict(trace=str(path.parent), states=len(rows), model=path.parent.name,
                   max_lift_mm=float(lift.max()),
                   max_relative_slip_mm=float(slip.max()),
                   mean_relative_slip_mm=float(slip.mean()),
                   max_relative_slip_state=int(np.argmax(slip)),
                   first_any_pad_contact_state=first_true(any_pad),
                   last_any_pad_contact_state=int(np.flatnonzero(any_pad)[-1]) if np.any(any_pad) else None,
                   states_with_any_pad_contact=int(any_pad.sum()),
                   any_pad_contact_regions=contact_regions(any_pad),
                   first_both_pad_contact_state=first_true(both_pads),
                   last_both_pad_contact_state=int(np.flatnonzero(both_pads)[-1]) if np.any(both_pads) else None,
                   states_with_both_pad_contact=int(both_pads.sum()),
                   both_pad_contact_regions=contact_regions(both_pads),
                   first_both_pad_contact_gap_state=first_falling_edge(both_pads))
    if logged is not None:
        metrics['logged_world_frame_max_relative_slip_mm'] = float(logged.max())
    metrics['both_pad_contact_is_adequate_grasp_proxy'] = bool(
        metrics['states_with_both_pad_contact'] or metrics['max_lift_mm'] <= 20)
    # Traces end at the original stop time, so whole-trace averages are not
    # comparable across models; the shared leading window is.
    window = min(WINDOW_STATES, len(rows))
    metrics['window_states'] = window
    metrics['window_max_relative_slip_mm'] = float(slip[:window].max())
    metrics['window_mean_relative_slip_mm'] = float(slip[:window].mean())
    metrics['window_max_lift_mm'] = float(lift[:window].max())
    metrics['window_states_with_any_pad_contact'] = int(any_pad[:window].sum())
    metrics['window_states_with_both_pad_contact'] = int(both_pads[:window].sum())

    gap = metrics['first_both_pad_contact_gap_state']
    if gap is not None:
        metrics['slip_around_first_both_pad_gap_mm'] = dict(
            at_state=gap, step_into_state_mm=float(slip[gap]),
            next_step_mm=float(slip[gap + 1]) if gap + 1 < len(slip) else None,
            cumulative_next_three_mm=float(slip[gap:gap + 3].sum()),
            max_from_gap_on_mm=float(slip[gap:].max()),
            both_pad_contact_resumed_at_state=next(
                (int(i) for i in np.flatnonzero(both_pads[gap:]) if i), None))

    missing = [name for name in FINGER_COLUMNS if column(rows, name) is None]
    metrics['finger_contact_columns_present'] = not missing
    if missing:
        metrics['finger_contact_columns_missing'] = missing
        metrics['finger_contact_interpretation'] = (
            'This replay predates finger-level logging, so contact with the non-pad finger '
            'geometry cannot be read here. A lost fingerpad pair therefore cannot be called '
            'an object release from these columns alone.')
        return metrics

    any_finger = column(rows, 'any_finger_contact')
    non_pad_only = column(rows, 'finger_contact_without_both_pads')
    release = first_falling_edge(any_finger)
    metrics.update(states_with_any_finger_contact=int(any_finger.sum()),
                   states_finger_contact_without_both_pads=int(non_pad_only.sum()),
                   first_finger_contact_without_both_pads_state=first_true(non_pad_only),
                   first_all_finger_contact_loss_state=release,
                   any_finger_contact_regions=contact_regions(any_finger))
    if gap is not None:
        metrics['finger_contact_at_first_both_pad_gap'] = bool(any_finger[gap])
        metrics['finger_body_contact_kept_after_pad_gap'] = bool(any_finger[gap:].any())
    if release is not None:
        metrics['finger_contact_ends_before_last_state'] = release < len(rows) - 1
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='append', required=True,
                        help='A model directory holding states.csv; repeat per model')
    parser.add_argument('--out', required=True)
    parser.add_argument('--figure', help='Optional PNG comparing pad contact, offset and slip')
    parser.add_argument('--title', default='Contact retention and relative slip')
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = []
    for name in args.replay:
        path = Path(name) / 'states.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        metrics.append(analyze(path))
    out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    columns = ['model', 'states', 'first_any_pad_contact_state', 'last_any_pad_contact_state',
               'states_with_any_pad_contact', 'any_pad_contact_regions',
               'states_with_both_pad_contact', 'both_pad_contact_regions',
               'first_both_pad_contact_gap_state', 'window_states_with_both_pad_contact',
               'first_finger_contact_without_both_pads_state', 'first_all_finger_contact_loss_state',
               'window_max_relative_slip_mm', 'window_mean_relative_slip_mm',
               'max_relative_slip_mm', 'mean_relative_slip_mm', 'max_lift_mm',
               'finger_contact_columns_present']
    with out.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics)
    print(out.with_suffix('.csv').read_text(encoding='utf-8'))

    if args.figure:
        figure, panels = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for entry, name in zip(metrics, args.replay):
            rows = read_states(Path(name) / 'states.csv')
            offsets = eef_relative_offset(rows)
            slip = np.zeros(len(rows))
            slip[1:] = np.linalg.norm(np.diff(offsets, axis=0), axis=1) * 1000
            steps = np.arange(len(rows))
            pads = column(rows, 'left_pad_contact').astype(int) + column(rows, 'right_pad_contact').astype(int)
            line, = panels[0].step(steps, pads, where='post', label=entry['model'])
            panels[1].plot(steps, np.linalg.norm(offsets, axis=1) * 1000, color=line.get_color(), label=entry['model'])
            panels[2].plot(steps, slip, color=line.get_color(), label=entry['model'])
            if entry['first_both_pad_contact_gap_state'] is not None:
                for panel in panels:
                    panel.axvline(entry['first_both_pad_contact_gap_state'], color=line.get_color(), ls=':', lw=.8)
        for panel, title, ylabel in zip(panels,
                ['Contacting fingerpad groups (0/1/2)',
                 'Bowl origin offset from EEF (mm)',
                 'Relative slip per step, EEF frame (mm)'],
                ['count', 'mm', 'mm']):
            panel.set(title=title, ylabel=ylabel)
            panel.grid(alpha=.2)
        panels[2].set_xlabel('State/action index (20Hz)')
        panels[0].set_yticks([0, 1, 2])
        panels[2].set_yscale('symlog')
        panels[0].legend(fontsize=8, loc='lower left')
        panels[0].text(.01, .04, 'Dotted vertical: first lost fingerpad pair, which later returns',
                          transform=panels[0].transAxes, fontsize=8)
        figure.suptitle(args.title)
        figure.savefig(args.figure, dpi=140, bbox_inches='tight')
        plt.close(figure)


if __name__ == '__main__':
    main()