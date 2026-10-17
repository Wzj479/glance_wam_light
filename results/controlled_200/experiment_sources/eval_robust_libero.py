"""Run under LIBERO's Python; use official policy client/action processing.

Applies a fixed camera perturbation each episode and sends paired inference seeds.
No original repository file is modified. Run via run_robust_eval.py.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import imageio.v2 as imageio
import numpy as np

from eval_protocol import MAX_STEPS, protocol_hash, validate_episode_results
from finetune_utils import photometric_window, stable_seed


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model-name', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=6694)
    p.add_argument('--save-videos', action='store_true')
    a = p.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    fingerprint = protocol_hash(protocol)
    corruption = protocol['conditions'][a.condition]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() or (out / 'episodes.jsonl').exists():
        raise FileExistsError('Use a new evaluation output directory')
    # Import official helpers only after parsing protocol and environment setup.
    from examples.LIBERO.eval_files.eval_libero import (
        LIBERO_DUMMY_ACTION, _binarize_gripper_open, _get_libero_env, benchmark)
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    np.random.seed(protocol['seed'])
    suite = benchmark.get_benchmark_dict()[protocol['suite']]()
    if max(protocol['task_ids']) >= suite.n_tasks:
        raise ValueError('Task ID outside this suite')
    # Validate all initial-state ranges before any rollout.
    states = {tid: suite.get_task_init_states(tid) for tid in protocol['task_ids']}
    if any(max(protocol['episode_ids']) >= len(v) for v in states.values()):
        raise ValueError('Episode ID exceeds available LIBERO initial states')
    client = ModelClient(policy_ckpt_path=a.ckpt, host='127.0.0.1', port=a.port, image_size=[224, 224])
    results = []
    for task_id in protocol['task_ids']:
        task = suite.get_task(task_id)
        for episode_id in protocol['episode_ids']:
            # New env per episode: reset RNG and avoid previous rollout history effects.
            env_seed = stable_seed(protocol['seed'], protocol['suite'], task_id, episode_id) % (2**31)
            env, instruction = _get_libero_env(task, 256, env_seed)
            record = dict(task_id=task_id, episode_id=episode_id, env_seed=env_seed,
                          instruction=instruction, task_name=task.name,
                          condition=a.condition, success=False)
            images, actions, query_times = [], [], []
            start_episode = time.perf_counter()
            try:
                client.reset(task_description=instruction)
                env.reset()
                initial = states[task_id][episode_id]
                state_bytes = np.asarray(initial).tobytes()
                record['init_state_sha256'] = hashlib.sha256(state_bytes).hexdigest()
                obs = env.set_init_state(initial)
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                record['post_settle_state_sha256'] = hashlib.sha256(
                    np.asarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
                for step in range(MAX_STEPS[protocol['suite']]):
                    # Preserve official 180-degree camera rotation and 224px resize.
                    pair = [np.ascontiguousarray(obs[key][::-1, ::-1])
                            for key in ('agentview_image', 'robot0_eye_in_hand_image')]
                    pair = [client._resize_image(im) for im in pair]
                    viewed = photometric_window(pair, corruption['brightness'],
                                                corruption['saturation'], corruption['gains'])
                    viewed = [np.asarray(im).copy() for im in viewed]
                    if a.save_videos:
                        images.append(np.concatenate([
                            np.concatenate([pair[0], viewed[0]], axis=1),
                            np.concatenate([pair[1], viewed[1]], axis=1)], axis=0))
                    seed = stable_seed(protocol['seed'], task_id, episode_id, step // client.action_chunk_size)
                    example = dict(image=viewed, lang=str(instruction), seed=seed)
                    is_query = step % client.action_chunk_size == 0
                    refresh = (client._cached_goal is None or
                        client._steps_since_refresh >= (client.goal_horizon_rows or float('inf')))
                    started = time.perf_counter()
                    response = client.step(example=example, step=step)
                    if is_query:
                        query_times.append(dict(step=step, refresh=refresh,
                                                seconds=time.perf_counter() - started))
                    raw = response['raw_action']
                    translation = np.asarray(raw['world_vector'], dtype=np.float32).reshape(-1)
                    rotation = np.asarray(raw['rotation_delta'], dtype=np.float32).reshape(-1)
                    gripper = _binarize_gripper_open(raw['open_gripper'])
                    action = np.concatenate([translation, rotation, gripper])
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError('Invalid robot action')
                    actions.append(action)
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        record['success'] = True
                        break
                if a.save_videos:
                    # Include the returned terminal state, not just pre-action frames.
                    pair = [np.ascontiguousarray(obs[key][::-1, ::-1])
                            for key in ('agentview_image', 'robot0_eye_in_hand_image')]
                    pair = [client._resize_image(im) for im in pair]
                    viewed = photometric_window(pair, corruption['brightness'],
                                                corruption['saturation'], corruption['gains'])
                    images.append(np.concatenate([
                        np.concatenate([pair[0], np.asarray(viewed[0])], axis=1),
                        np.concatenate([pair[1], np.asarray(viewed[1])], axis=1)], axis=0))
                record.update(steps=len(actions), seconds=time.perf_counter() - start_episode,
                              query_times=query_times, video_frames=len(images),
                              video_fps=20, video_time_base='simulation',
                              video_layout='rows: primary/wrist; columns: original/observed')
            finally:
                env.close()
            stem = f'task_{task_id:02d}_episode_{episode_id:03d}'
            np.save(out / (stem + '_actions.npy'), np.stack(actions))
            if a.save_videos:
                imageio.mimwrite(out / (stem + '.mp4'), images, fps=20)
            results.append(record)
            with (out / 'episodes.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    validate_episode_results(results, protocol)
    successes = sum(r['success'] for r in results)
    summary = dict(schema_version=1, model=a.model_name, checkpoint=str(Path(a.ckpt).resolve()),
        protocol_sha256=fingerprint, condition=a.condition, corruption=corruption,
        successes=successes, episodes=len(results), success_rate=successes / len(results),
        episode_results=results, timing_scope='synchronous official client, cached goal cadence')
    atomic_json(out / 'results.json', summary)


if __name__ == '__main__':
    main()
