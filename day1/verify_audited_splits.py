"""Exercise the trainer's explicit-split selector against actual Parquet rows."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from finetune_utils import load_episode_partition, split_step_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    table = pq.read_table(args.root / 'data/chunk-000/file-000.parquet',
                          columns=['episode_index', 'frame_index', 'index'])
    rows = table.to_pylist()
    steps = [(r['episode_index'], r['frame_index']) for r in rows]
    name = 'libero_goal_no_noops_1.0.0_lerobot'
    tr, va = load_episode_partition(args.split, name, {e for e, _ in steps})
    ti, vi = split_step_indices(steps, tr, va, 8)
    assert {steps[i][0] for i in ti} == set(tr)
    assert {steps[i][0] for i in vi} == set(va)
    assert not {rows[i]['index'] for i in ti} & {rows[i]['index'] for i in vi}
    result = dict(status='passed', split=str(args.split), train_episodes=tr, val_episodes=va,
                  train_windows=len(ti), validation_windows_stride8=len(vi),
                  scope='Real Parquet + trainer split helpers; full official loader/GPU not executed')
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
