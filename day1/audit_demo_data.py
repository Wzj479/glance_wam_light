"""Bounded-memory audit of real, pinned LeRobot v3 LIBERO demonstrations.

No model, CUDA, simulator, pickle or text encoder is loaded. This is a CPU
reference for the source data contract, not a claim of full WAM loader parity.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import csv
import hashlib
import json
import resource
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from finetune_utils import photometric_window, split_episodes, stable_seed
from demo_audit_plan import transitions

NAME = 'libero_goal_no_noops_1.0.0_lerobot'
KEYS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
CAMS = ['observation.images.image', 'observation.images.wrist_image']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def normalize(actions, stats):
    result = actions.copy().astype(np.float32)
    lo, hi = np.array(stats['min'], np.float32), np.array(stats['max'], np.float32)
    # Official min_max does NOT clamp, and leaves gripper untouched.
    for i in range(6):
        result[:, i] = 2 * ((result[:, i] - lo[i]) / (hi[i] - lo[i])) - 1 if hi[i] != lo[i] else 0
    return result


def decode(root, episode, cam, frame, cache):
    timestamp = episode[f'videos/{cam}/from_timestamp'] + frame / 20
    global_frame = round(timestamp * 20)
    rel = f'frames/{cam}/{global_frame:08d}.png'
    cached = root / rel
    if cached.exists():
        receipts = cache[rel]
        assert sha(cached) == receipts['sha256']
        assert receipts['global_frame'] == global_frame
        return Image.open(cached).convert('RGB'), timestamp, global_frame
    import imageio_ffmpeg
    video = root / f'videos/{cam}/chunk-000/file-000.mp4'
    # Accurate output seeking; constant-20Hz stream, metadata supplies shared-file offset.
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '1',
           '-ss', f'{timestamp:.6f}', '-i', str(video), '-frames:v', '1',
           '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    data = subprocess.run(cmd, check=True, capture_output=True).stdout
    assert len(data) == 256 * 256 * 3, (cam, frame, len(data))
    return Image.fromarray(np.frombuffer(data, np.uint8).reshape(256, 256, 3)), timestamp, global_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--task', default='open the middle drawer of the cabinet')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--plan', type=Path, help='Explicit, source-derived grasp windows and episodes')
    args = p.parse_args()
    root, out = Path(args.root), Path(args.out)
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if plan:
        args.task = plan['task']
        assert args.seed == plan['seed'] and args.val_fraction == plan['val_fraction']
    out.mkdir(parents=True, exist_ok=True)
    input_receipt = json.loads((root / 'download_receipt.json').read_text())
    verified_inputs, absent_videos = [], []
    for item in input_receipt['files']:
        path = root / item['path']
        if path.exists():
            assert path.stat().st_size == item['size'] and sha(path) == item['sha256']
            verified_inputs.append(item['path'])
        else:
            assert item['path'].startswith('videos/'), item['path']
            absent_videos.append(item['path'])
    meta = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet')
    cols = [c for c in meta.column_names if not c.startswith('stats/')]
    candidates = sorted([e for e in meta.select(cols).to_pylist() if args.task in e['tasks']],
                        key=lambda e: e['episode_index'])[:args.episodes]
    assert len(candidates) == args.episodes and len(candidates) >= 2
    ids = [e['episode_index'] for e in candidates]
    train, val = split_episodes(ids, args.val_fraction, stable_seed(args.seed, NAME))
    if plan:
        assert ids == plan['selected_episodes']
        assert train == plan['train_episodes'] and val == plan['val_episodes']
    train_rows, val_rows, actions, ep_data = [], [], [], {}
    task_table = pq.read_table(root / 'meta/tasks.parquet').to_pylist()
    task_map = {t['task_index']: t['task'] for t in task_table}
    for e in candidates:
        eid = e['episode_index']
        table = pq.read_table(root / 'data/chunk-000/file-000.parquet',
                              filters=[('episode_index', '=', eid)])
        rows = sorted(table.to_pylist(), key=lambda r: r['frame_index'])
        n = e['length']
        assert len(rows) == n
        assert [r['frame_index'] for r in rows] == list(range(n))
        assert [r['index'] for r in rows] == list(range(e['dataset_from_index'], e['dataset_to_index']))
        assert all(task_map[r['task_index']] == args.task for r in rows)
        np.testing.assert_allclose([r['timestamp'] for r in rows], np.arange(n) / 20, atol=2e-5, rtol=0)
        for cam in CAMS:
            duration = e[f'videos/{cam}/to_timestamp'] - e[f'videos/{cam}/from_timestamp']
            assert abs(duration - n / 20) < 1e-4
        a = np.asarray([r['action'] for r in rows], np.float32)
        assert a.shape == (n, 7) and np.isfinite(a).all()
        ep_data[eid] = (e, rows, a)
        actions.append(a)
        (train_rows if eid in train else val_rows).extend(r['index'] for r in rows)
    assert not set(train_rows) & set(val_rows)
    split = dict(seed=args.seed, dataset=NAME, task=args.task, scope='selected_subset_adaptation_holdout',
                 train_episodes=train, val_episodes=val, train_global_rows=train_rows,
                 val_global_rows=val_rows, selected_episodes=ids,
                 episode_lengths={str(e['episode_index']): e['length'] for e in candidates},
                 pretraining_overlap='Unknown; do not claim unseen-task generalization')
    dump(out / 'split.json', split)
    stats = json.loads((root / 'meta/stats_gr00t.json').read_text())['statistics']['action']
    all_a = np.concatenate(actions)
    gripper_episodes = []
    for eid, (e, rows, a) in ep_data.items():
        grip = a[:, 6]
        edges = transitions(grip)
        if plan:
            assert set(np.unique(grip)) == {0., 1.}
            assert {edge['event'] for edge in edges} == {'open', 'close'}
        chunks = grip[np.minimum(np.arange(len(a))[:, None] + np.arange(8), len(a) - 1)]
        gripper_episodes.append(dict(episode=eid, split='train' if eid in train else 'validation',
            length=len(a), open_rows=int(np.sum(grip == 1)), closed_rows=int(np.sum(grip == 0)),
            transitions=edges, windows_spanning_switch=int(np.sum(np.any(chunks[:, 1:] != chunks[:, :-1], axis=1)))))
    gripper_summary = {}
    for part, allowed in [('train', train), ('validation', val)]:
        subset = [e for e in gripper_episodes if e['episode'] in allowed]
        gripper_summary[part] = {key: sum(e[key] for e in subset)
            for key in ['length', 'open_rows', 'closed_rows', 'windows_spanning_switch']}
        gripper_summary[part]['close_events'] = sum(x['event'] == 'close' for e in subset for x in e['transitions'])
        gripper_summary[part]['open_events'] = sum(x['event'] == 'open' for e in subset for x in e['transitions'])
    dump(out / 'gripper_events.json', dict(encoding={'open': 1, 'closed': 0},
                                         summary=gripper_summary, episodes=gripper_episodes))
    with (out / 'action_stats.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dimension', 'min', 'max', 'mean', 'std', 'p01', 'p99', 'zero_fraction'])
        for i, name in enumerate(KEYS):
            x = all_a[:, i]
            writer.writerow([name, float(x.min()), float(x.max()), float(x.mean()), float(x.std()),
                             float(np.quantile(x, .01)), float(np.quantile(x, .99)), float(np.mean(x == 0))])
    cache = {}
    if (root / 'frame_receipt.json').exists():
        cache = json.loads((root / 'frame_receipt.json').read_text())['frames']
    samples = []
    # One ordinary training sample, one validation sample and a trajectory-tail sample.
    specs = [('train', train[0], 10), ('validation', val[0], 10), ('tail', train[0], -3)]
    if plan:
        specs = [(s['tag'], s['episode'], s['frame']) for s in plan['samples']]
    canvas = Image.new('RGB', (3 * 448, len(specs) * 270 + 40), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f'{args.task} | each tile: primary (left), wrist (right)', fill='black')
    for row_number, (tag, eid, t) in enumerate(specs):
        e, rows, a = ep_data[eid]
        t = t if t >= 0 else len(a) + t
        rng = np.random.default_rng(stable_seed(args.seed, 0, NAME, eid, t))
        g = int(rng.integers(1, 49))
        requested_actions = np.arange(t, t + 8)
        ai = requested_actions.clip(0, len(a) - 1)
        image_indices = [t, min(t + 48, len(a) - 1), min(t + g, len(a) - 1)]
        raw = a[ai]
        normalized = normalize(raw, stats).astype(np.float16)
        if tag == 'tail':
            np.testing.assert_array_equal(raw[3:], np.repeat(a[-1:], 5, axis=0))
        np.testing.assert_array_equal(normalized[:, 6], raw[:, 6].astype(np.float16))
        if plan and tag != 'tail':
            spec = next(s for s in plan['samples'] if s['tag'] == tag)
            assert {0., 1.} == set(raw[:, 6])
            assert rows[spec['event_frame']]['frame_index'] == spec['event_frame']
            assert raw[2, 6] == (1 if spec['event'] == 'close' else 0)
            assert raw[3, 6] == (0 if spec['event'] == 'close' else 1)
        images, sources = [], []
        for slot, fi in enumerate(image_indices):
            views = []
            for cam in CAMS:
                image, ts, global_frame = decode(root, e, cam, fi, cache)
                views.append(np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR)))
                sources.append(dict(slot=slot, camera=cam, episode_frame=fi,
                                    shared_video_frame=global_frame, timestamp=ts))
            merged = np.concatenate(views, axis=1)
            images.append(merged)
            x, y = slot * 448, 40 + row_number * 270
            canvas.paste(Image.fromarray(merged), (x, y + 34))
            draw.text((x + 5, y), f'{tag} ep={eid} t={t} slot={slot} frame={fi}', fill='black')
        dark = photometric_window(images, brightness=.55)
        color = photometric_window(images, saturation=.65, gains=(1.18, .85, .75))
        original_hash = hashlib.sha256(normalized.tobytes()).hexdigest()
        for variant, window in [('clean', images), ('dark', dark), ('color', color)]:
            np.savez_compressed(out / f'{tag}_{variant}.npz', images=np.asarray(window),
                                action=normalized, raw_action=raw, language=np.array(args.task),
                                goal_offset_rows=np.array(g))
        assert hashlib.sha256(normalized.tobytes()).hexdigest() == original_hash
        trace = dict(tag=tag, episode=eid, split='validation' if eid in val else 'train',
                     base_frame=t, episode_length=len(a), language=args.task,
                     requested_goal_offset=g, effective_goal_offset=image_indices[2] - t,
                     image_episode_indices=image_indices, requested_action_indices=requested_actions.tolist(),
                     action_episode_indices=ai.tolist(), tail_padding=(requested_actions >= len(a)).tolist(),
                     action_shape=[8, 7], action_dtype='float16', image_shape=[3, 224, 448, 3],
                     normalization='published full-dataset min_max for first 6, no clamp; gripper unchanged',
                     action_sha256=original_hash, sources=sources)
        trace['gripper_action_labels'] = raw[:, 6].tolist()
        if plan:
            trace['event'] = next(s['event'] for s in plan['samples'] if s['tag'] == tag)
        samples.append(trace)
    canvas.save(out / 'sample_windows.png')
    comparison = Image.new('RGB', (448 * 3, 250), 'white')
    painter = ImageDraw.Draw(comparison)
    for i, variant in enumerate(('clean', 'dark', 'color')):
        z = np.load(out / f'train_{variant}.npz')
        comparison.paste(Image.fromarray(z['images'][0]), (i * 448, 26))
        painter.text((i * 448 + 8, 6), variant, fill='black')
        np.testing.assert_array_equal(z['action'], np.load(out / 'train_clean.npz')['action'])
    comparison.save(out / 'appearance_comparison.png')
    dump(out / 'sample_trace.json', samples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(7, 1, figsize=(10, 12))
    for i, ax in enumerate(axes):
        ax.hist(all_a[:, i], bins=40)
        ax.set_ylabel(KEYS[i])
    fig.suptitle('Real demonstration actions: selected 10 episodes; simulator command units')
    fig.tight_layout()
    fig.savefig(out / 'action_histograms.png', dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(len(ep_data), 1, figsize=(11, 1.6 * len(ep_data)))
    for ax, (eid, (_, rows, a)) in zip(np.atleast_1d(axes), ep_data.items()):
        ax.step(np.arange(len(a)), a[:, 6], where='post')
        ax.set_yticks([0, 1], ['closed', 'open'])
        ax.set_ylim(-.15, 1.15)
        ax.set_ylabel(f'ep {eid}')
        ax.set_title('train' if eid in train else 'validation', fontsize=8)
    axes[-1].set_xlabel('Episode frame index (20Hz processed demonstration)')
    fig.tight_layout()
    fig.savefig(out / 'gripper_timeline.png', dpi=110)
    plt.close(fig)
    receipt = input_receipt
    report = dict(status='passed', dataset_revision='0b9208dff852914f94be879f702cd40118255bce',
                  code_revision='6a51bdc5139e72e8ad525779db0c989e1af5abab', task=args.task,
                  episodes=len(ids), real_action_rows=len(all_a), train_rows=len(train_rows), val_rows=len(val_rows),
                  train_episodes=train, val_episodes=val, finite_actions=True, episode_overlap=0,
                  gripper_values=np.unique(all_a[:, 6]).tolist(), raw_action_min=all_a.min(0).tolist(),
                  gripper_summary=gripper_summary,
                  raw_action_max=all_a.max(0).tolist(), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  input_files_sha256_verified=verified_inputs, missing_full_videos=absent_videos,
                  image_input_mode='verified_sparse_png_cache' if absent_videos else 'original_shared_videos',
                  checks=['contiguous episode frames', 'global row bounds', '20Hz timestamps',
                          'video episode durations', 'task IDs', 'finite 7D actions', 'episode split isolation',
                          'tail repeat padding', 'appearance variants preserve action labels'],
                  limits=['No WAM forward/backward or official complete loader run',
                          'PIL bilinear reference resize; pixel-exact upstream resize parity untested',
                          'roll/pitch/yaw are source field names; LIBERO uses axis-angle command components',
                          'Official data are simulator demonstrations, not physical-robot captures'],
                  download_receipt=receipt)
    dump(out / 'report.json', report)
    if plan:
        dump(out / 'audit_plan.json', plan)
    dump(out / 'SHA256SUMS.json', {f.name: sha(f) for f in sorted(out.iterdir())
                                  if f.is_file() and f.name != 'SHA256SUMS.json'})
    print(json.dumps({k: v for k, v in report.items() if k != 'download_receipt'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
