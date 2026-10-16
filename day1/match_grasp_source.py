"""Find a LeRobot episode in pinned original LIBERO HDF5 by its entire action sequence.

Run on networked host. HTTP Range fetches metadata/actions/state, not image arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import fsspec
import h5py
import numpy as np
import pyarrow.parquet as pq

REVISION = 'f13aa24a3da8c43c7225569f28c562979fa0e35a'
LFS_SHA = 'e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406'
URL = ('https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/' +
       REVISION + '/libero_goal/put_the_bowl_on_the_plate_demo.hdf5')


def nonnoop_indices(actions):
    kept = []
    for i, action in enumerate(actions):
        noop = np.linalg.norm(action[:6]) < 1e-4 and (
            not kept or action[6] == actions[kept[-1], 6])
        if not noop:
            kept.append(i)
    return np.asarray(kept)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--local-hdf5', type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.data_root / 'data/chunk-000/file-000.parquet',
                          filters=[('episode_index', '=', args.episode)]).to_pylist()
    rows.sort(key=lambda r: r['frame_index'])
    target = np.asarray([r['action'] for r in rows], np.float64)
    assert len(target) and rows[0]['task_index'] == 0
    target_env = target.copy()
    target_env[:, 6] = 1 - 2 * (target[:, 6] > .5)
    path = args.local_hdf5 or fsspec.open(URL, 'rb', block_size=65536, cache_type='blockcache').open()
    examined, matches = [], []
    with h5py.File(path, 'r') as h:
        names = sorted(h['data'].keys(), key=lambda x: int(x.split('_')[-1]))
        for name in names:
            group = h['data'][name]
            action = np.asarray(group['actions'])
            kept = nonnoop_indices(action)
            best = None
            for mode, candidate in [('original', action), ('nonnoop', action[kept])]:
                if candidate.shape == target_env.shape:
                    error = float(np.max(np.abs(candidate - target_env)))
                    best = error if best is None else min(best, error)
                    if error < 2e-6:
                        matches.append((name, mode, error, kept))
            examined.append(dict(demo=name, original_length=len(action), filtered_length=len(kept),
                                 max_abs_error_if_shape_matches=best))
            print('EXAMINED', examined[-1], flush=True)
        assert len({m[0] for m in matches}) == 1, matches
        name, mode, error, kept = matches[0]
        group = h['data'][name]
        original_actions = np.asarray(group['actions'])
        original_states = np.asarray(group['states'])
        xml = group.attrs.get('model_file', '')
        joint_names = [x.attrib.get('name') for x in ET.fromstring(xml).find('worldbody').iter('joint')]
        np.savez_compressed(args.out / 'matched_demo.npz',
            initial_state=original_states[0], original_actions=original_actions,
            original_states=original_states, nonnoop_indices=kept,
            lerobot_actions=target, env_actions=target_env,
            lerobot_states=np.asarray([r['observation.state'] for r in rows]),
            source_ee_states=np.asarray(group['obs/ee_states']),
            source_gripper_states=np.asarray(group['obs/gripper_states']))
        (args.out / 'source_model.xml').write_text(xml, encoding='utf-8')
        meta = dict(source_url=URL, source_revision=REVISION, published_full_file_lfs_sha256=LFS_SHA,
            access='local HDF5' if args.local_hdf5 else 'HTTP Range; full file SHA not recomputed',
            lerobot_dataset_revision='0b9208dff852914f94be879f702cd40118255bce',
            lerobot_episode=args.episode, matched_original_demo=name, action_match_mode=mode,
            full_action_max_abs_error=error, original_steps=len(original_actions),
            retained_nonnoop_steps=len(kept), target_steps=len(target),
            hdf5_env_args=json.loads(h['data'].attrs['env_args']), joint_names=joint_names,
            examined=examined)
        meta['extracted_npz_sha256'] = hashlib.sha256((args.out / 'matched_demo.npz').read_bytes()).hexdigest()
        (args.out / 'match.json').write_text(json.dumps(meta, indent=2))
        print('UNIQUE_MATCH', name, mode, error, flush=True)


if __name__ == '__main__':
    main()
