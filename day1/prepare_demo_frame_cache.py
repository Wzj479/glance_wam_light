"""Network fallback: extract a few lossless frames from verified official files.

Run on the download host; transfer the resulting cache plus original Parquet/meta
to the CPU server. No trimming/re-encoding of shared video timestamps is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from finetune_utils import stable_seed, split_episodes


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--plan', type=Path)
    args = p.parse_args()
    root = Path(args.root)
    name = 'libero_goal_no_noops_1.0.0_lerobot'
    episodes = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    plan = json.loads(args.plan.read_text()) if args.plan else None
    task = plan['task'] if plan else 'open the middle drawer of the cabinet'
    episodes = sorted([e for e in episodes if task in e['tasks']], key=lambda e: e['episode_index'])[:10]
    train, val = split_episodes([e['episode_index'] for e in episodes], .2, stable_seed(42, name))
    if plan:
        assert [e['episode_index'] for e in episodes] == plan['selected_episodes']
        assert train == plan['train_episodes'] and val == plan['val_episodes']
    byid = {e['episode_index']: e for e in episodes}
    needed = {cam: set() for cam in ['observation.images.image', 'observation.images.wrist_image']}
    specs = [(s['episode'], s['frame']) for s in plan['samples']] if plan else [(train[0], 10), (val[0], 10), (train[0], -3)]
    for eid, t in specs:
        e = byid[eid]
        if t < 0:
            t += e['length']
        g = int(np.random.default_rng(stable_seed(42, 0, name, eid, t)).integers(1, 49))
        for cam in needed:
            for frame in [t, min(t + 48, e['length'] - 1), min(t + g, e['length'] - 1)]:
                needed[cam].add(round((e[f'videos/{cam}/from_timestamp'] + frame / 20) * 20))
    receipt = dict(method='OpenCV indexed AV1 decode, original shared-file 20Hz frame indices',
                   data_revision='0b9208dff852914f94be879f702cd40118255bce',
                   opencv_version=cv2.__version__, source_videos={}, frames={})
    official = json.loads((root / 'download_receipt.json').read_text())
    for cam, indices in needed.items():
        relvideo = f'videos/{cam}/chunk-000/file-000.mp4'
        video = root / relvideo
        expected = next(x['official_lfs_sha256'] for x in official['files'] if x['path'] == relvideo)
        actual = sha(video)
        assert expected == actual
        receipt['source_videos'][cam] = dict(path=relvideo, sha256=actual)
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 20
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 52042
        for frame in sorted(indices):
            assert cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            assert ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            assert abs(timestamp - frame / 20) < .001, (frame, timestamp)
            rel = f'frames/{cam}/{frame:08d}.png'
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(dest)
            receipt['frames'][rel] = dict(sha256=sha(dest), global_frame=frame,
                                         decoded_pts_seconds=timestamp, source_camera=cam)
        cap.release()
    receipt_path = root / 'frame_receipt.json'
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        prior['frames'].update(receipt['frames'])
        prior['source_videos'].update(receipt['source_videos'])
        receipt = prior
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print('VERIFIED_FRAMES', len(receipt['frames']), flush=True)


if __name__ == '__main__':
    main()
