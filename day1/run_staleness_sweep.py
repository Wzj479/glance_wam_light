"""Drive the goal-staleness sweep and aggregate it into a decay curve.

For each refresh interval in the grid, this runs the same closed-loop diagnostic the task
runs use (``run_gripper_feedback.py``), but forces the eval client to hold its goal for that
many native rows via ``GLANCEWAM_GOAL_REFRESH_ROWS`` before regenerating. It then reads each
arm's ``driver/results.json`` and folds the per-episode outcomes into one curve with
``goal_staleness.aggregate_staleness``.

This must run on the GPU server (it spawns the policy server + the MuJoCo client). It mirrors
``results/task1/run_task1.sh``'s process setup: it runs ``run_gripper_feedback.py`` from the
GlanceWAM work tree so the client can find ``day1/diagnose_gripper_feedback.py`` there.

Usage (on the GPU server):
    python day1/run_staleness_sweep.py \
      --protocol results/task1/protocols/proto-a.json --condition color_strong \
      --ckpt results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt --model-name base \
      --out results/staleness/base --H-g 48 --multipliers 1,1.5,2,3,6 \
      --driver-args --target-body flat_stove_1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from goal_staleness import aggregate_staleness, load_driver_results, refresh_grid  # noqa: E402

REPO = Path('/root/autodl-tmp/wam_glance_project/glance_wam_light')
WORK = Path('/root/autodl-tmp/wam_glance_project/GlanceWAM')
LIBERO = Path('/root/autodl-tmp/wam_glance_project/LIBERO')


def run_arm(refresh_rows, args, arm_out, port):
    """Run one refresh-interval arm; return its per-episode records tagged with the arm."""
    env = os.environ.copy()
    env['GLANCEWAM_GOAL_REFRESH_ROWS'] = str(refresh_rows)
    cmd = [
        sys.executable, str(REPO / 'day1/run_gripper_feedback.py'),
        '--protocol', str(args.protocol), '--condition', args.condition,
        '--ckpt', str(args.ckpt), '--model-name', args.model_name,
        '--out', str(arm_out), '--port', str(port),
        '--libero-python', args.libero_python, '--libero-home', str(LIBERO),
        '--driver-args', *args.driver_args,
    ]
    print(f'=== staleness R={refresh_rows} -> {arm_out} (port {port}) ===', flush=True)
    subprocess.run(cmd, cwd=str(WORK), env=env, check=True)
    records = load_driver_results(arm_out / 'driver' / 'results.json')
    for record in records:
        record['refresh_rows'] = refresh_rows
    return records


def plot_curve(curve, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs = [p['staleness_rows'] for p in curve]
    ys = [p['success_rate'] for p in curve]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(xs, ys, 'o-')
    ax.set_xlabel('extra staleness beyond H_g (native rows)')
    ax.set_ylabel('closed-loop success rate')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model-name', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--H-g', type=float, default=48.0, help='trained goal horizon (goal_horizon_rows)')
    p.add_argument('--multipliers', default='1,1.5,2,3,6',
                   help='refresh intervals as multiples of H_g; a large value = frozen goal')
    p.add_argument('--port', type=int, default=6694)
    p.add_argument('--libero-python', default='/root/wam_sim_env/bin/python')
    p.add_argument('--driver-args', nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Run on the Linux GPU server')

    multipliers = [float(x) for x in a.multipliers.split(',') if x.strip()]
    grid = refresh_grid(a.H_g, multipliers)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new --out directory; partial sweeps are not merged')
    out.mkdir(parents=True, exist_ok=True)

    all_records = []
    for index, rows in enumerate(grid):
        records = run_arm(rows, a, out / f'R{rows}', a.port + index)
        all_records.extend(records)
        with (out / 'records.jsonl').open('a', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record) + '\n')

    summary = aggregate_staleness(all_records, H_g=a.H_g)
    (out / 'staleness_curve.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_curve(summary['curve'], out / 'staleness_curve.png')

    print(json.dumps(summary, indent=2))
    print(f'wrote {out / "staleness_curve.json"} (+ .png)')


if __name__ == '__main__':
    main()
