"""Build a real grasp audit plan and export explicit trainer episode splits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from finetune_utils import split_episodes, stable_seed

NAME = 'libero_goal_no_noops_1.0.0_lerobot'
TASK = 'put the bowl on the plate'


def transitions(gripper):
    x = np.asarray(gripper)
    return [dict(frame=int(k), before=float(x[k - 1]), after=float(x[k]),
                 event='close' if x[k] == 0 else 'open')
            for k in np.flatnonzero(x[1:] != x[:-1]) + 1]


def plan(root, out):
    episodes = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    candidates = sorted([e for e in episodes if TASK in e['tasks']], key=lambda e: e['episode_index'])[:10]
    summaries = []
    for e in candidates:
        rows = pq.read_table(root / 'data/chunk-000/file-000.parquet',
                             filters=[('episode_index', '=', e['episode_index'])]).to_pylist()
        rows.sort(key=lambda r: r['frame_index'])
        g = np.asarray([r['action'][6] for r in rows])
        assert set(np.unique(g)) == {0., 1.}
        events = transitions(g)
        assert {x['event'] for x in events} == {'close', 'open'}
        summaries.append(dict(episode=e['episode_index'], length=len(rows),
                              open_rows=int(np.sum(g == 1)), closed_rows=int(np.sum(g == 0)),
                              transitions=events))
    ids = [e['episode_index'] for e in candidates]
    tr, va = split_episodes(ids, .2, stable_seed(42, NAME))
    specs = []
    for prefix, eid in [('train', tr[0]), ('validation', va[0])]:
        summary = next(x for x in summaries if x['episode'] == eid)
        for event, tag in [('close', prefix), ('open', prefix + '_release')]:
            edge = next(x['frame'] for x in summary['transitions'] if x['event'] == event)
            specs.append(dict(tag=tag, episode=eid, frame=max(0, edge - 3), event=event, event_frame=edge))
    specs.append(dict(tag='tail', episode=tr[0], frame=-3, event='tail'))
    result = dict(schema_version=1, dataset=NAME, task=TASK, seed=42, val_fraction=.2,
                  selected_episodes=ids, train_episodes=tr, val_episodes=va,
                  samples=specs, gripper_episodes=summaries)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def combine(paths, out):
    datasets, sources = {}, []
    for path in paths:
        source = json.loads(path.read_text(encoding='utf-8'))
        target = datasets.setdefault(source['dataset'], dict(train_episode_ids=[], val_episode_ids=[]))
        target['train_episode_ids'].extend(source['train_episodes'])
        target['val_episode_ids'].extend(source['val_episodes'])
        sources.append(dict(task=source['task'], selected_episodes=source['selected_episodes']))
    for item in datasets.values():
        tr, va = item['train_episode_ids'], item['val_episode_ids']
        assert len(tr) == len(set(tr)) and len(va) == len(set(va))
        assert not set(tr) & set(va)
        item['train_episode_ids'], item['val_episode_ids'] = sorted(tr), sorted(va)
    result = dict(schema_version=1, datasets=datasets, sources=sources,
                  dataset_revision='0b9208dff852914f94be879f702cd40118255bce',
                  scope='Adaptation holdout; possible pretrained overlap')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest='command', required=True)
    a = commands.add_parser('grasp')
    a.add_argument('--root', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    a = commands.add_parser('combine')
    a.add_argument('--splits', type=Path, nargs='+', required=True)
    a.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'grasp':
        plan(args.root, args.out)
    else:
        combine(args.splits, args.out)


if __name__ == '__main__':
    main()
