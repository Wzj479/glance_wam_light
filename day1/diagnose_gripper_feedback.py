"""Gripper-intervention closed-loop diagnostic on one held-back initial state.

The frozen-action counterfactual could not separate a bad transport plan from a
limited replay: steps after the intervention still executed the six arm channels
of the original failure trajectory, which had been generated from the original
observations. This driver re-queries the policy with the observation it actually
has after the intervention starts, so every executed arm command comes from a real
inference on the state that was reached.

Only the gripper channel is overridden inside the window. Nothing else is masked,
blended or gated, and no release rule is added after the window.

Inference cadence is the deployed one and is recorded rather than assumed: the
official client queries once every ``action_chunk_size`` steps and replays the
returned block in between, so a block's arm channels come from one inference.
Actions before the window are untouched, therefore the observation entering the
first window query matches the archived run and that query reproduces the archived
block; the divergence step is then derived from the recorded blocks.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np

GRIPPER_CLOSE = 1.0
FINGER_GROUPS = ('left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad')


def parse_window(values):
    """Validate an inclusive [START, END] action interval; None disables intervention."""
    if not values:
        return None
    if len(values) != 2:
        raise ValueError('Use --hold-close START END as an inclusive action interval')
    start, end = int(values[0]), int(values[1])
    if not (0 <= start <= end):
        raise ValueError('Hold-close interval must satisfy 0 <= START <= END')
    return (start, end)


def override_gripper(action, step, window):
    """Return (executed_action, overridden). The six arm channels are copied through.

    The deployed gripper sign is +1 close / -1 open, as produced by the official
    client's _binarize_gripper_open; forcing the wrong sign would run the opposite
    experiment silently, so the constant is asserted against the action below.
    """
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,):
        raise ValueError('Expected a 7-dim action')
    if window is None or not (window[0] <= step <= window[1]):
        return action, False
    if action[6] not in (GRIPPER_CLOSE, -GRIPPER_CLOSE):
        raise ValueError(f'Unexpected gripper command {action[6]}; the official client binarises to +/-1')
    executed = action.copy()
    executed[6] = GRIPPER_CLOSE
    return executed, True


def first_divergence_step(actions, reference):
    """First action index whose arm channels differ from the archived trajectory.

    Only the six arm channels are compared. The gripper is expected to differ
    inside the window by construction, so including it would report the
    intervention itself rather than the first genuine behavioural divergence.
    """
    if reference is None:
        return None
    actions = np.asarray(actions, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape[1] != 7:
        raise ValueError('Reference actions must be (steps, 7)')
    for step in range(min(len(actions), len(reference))):
        if not np.array_equal(actions[step][:6], reference[step][:6]):
            return step
    return None


def subset_protocol(protocol, episode_id, condition=None):
    """Derive a one-state protocol from a parent protocol.

    Every per-episode input is a function of (seed, suite, task, episode) and of
    the condition, none of the episode list, so the derived protocol reproduces
    the archived episode exactly. The protocol hash necessarily changes, which is
    why a subset run is recorded separately instead of being mixed into the suite.
    """
    from eval_protocol import validate_protocol
    if episode_id not in protocol['episode_ids']:
        raise ValueError(f'Episode {episode_id} is not part of the parent protocol')
    derived = dict(protocol, episode_ids=[int(episode_id)])
    if condition is not None:
        if condition not in protocol['conditions']:
            raise ValueError(f'Unknown condition {condition}')
        derived['conditions'] = {condition: protocol['conditions'][condition]}
    validate_protocol(derived)
    return derived


def chunk_records(chunks):
    """Per-query rows: full action block plus goal provenance and seed."""
    rows = []
    for index, entry in enumerate(chunks):
        rows.append(dict(query_index=index, query_step=entry['step'],
                         goal_source=entry['goal_source'], goal_horizon_rows=entry['goal_horizon_rows'],
                         seed=entry['seed'], chunk_position=entry['chunk_position'],
                         chunk_actions=json.dumps(entry['actions'].tolist())))
    return rows


def slip_series(offsets):
    """Per-step change of the bowl offset from the EEF origin, in millimetres."""
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    slip = np.zeros(len(offsets))
    if len(offsets) > 1:
        slip[1:] = np.linalg.norm(np.diff(offsets, axis=0), axis=1) * 1000
    return slip


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_record(step, obs, env, inner, bowl, plate, bid, pid, grip, offset, slip):
    """One post-settle-indexed state: poses, contact groups, offset and slip.

    The EEF quaternion is logged even though bowl_in_eef_* is already derived from
    it, so analyze_contact_transfer.py can read this log unchanged. That keeps one
    set of metric definitions for the offline replays and the closed-loop runs,
    and the redundancy lets the rotation be cross-checked instead of trusted.
    """
    bowl_pos = env.sim.data.body_xpos[bid].copy()
    plate_pos = env.sim.data.body_xpos[pid].copy()
    eef_pos = np.asarray(obs['robot0_eef_pos'], dtype=float)
    pads = {name: bool(inner.check_contact(grip.important_geoms[name], bowl.contact_geoms))
            for name in FINGER_GROUPS}
    any_finger = pads['left_finger'] or pads['right_finger']
    both_pads = pads['left_fingerpad'] and pads['right_fingerpad']
    xy = float(np.linalg.norm(bowl_pos[:2] - plate_pos[:2]))
    contact = bool(inner.check_contact(bowl, plate))
    above = bool(bowl_pos[2] >= plate_pos[2])
    return dict(state_step=int(step), time_s=step / 20,
                **{f'eef_{a}': float(v) for a, v in zip('xyz', eef_pos)},
                **{f'eef_q{a}': float(v) for a, v in zip('xyzw', np.asarray(obs['robot0_eef_quat'], dtype=float))},
                **{f'bowl_{a}': float(v) for a, v in zip('xyz', bowl_pos)},
                **{f'plate_{a}': float(v) for a, v in zip('xyz', plate_pos)},
                finger_width_m=float(np.abs(np.asarray(obs['robot0_gripper_qpos'], dtype=float)).sum()),
                left_pad_contact=pads['left_fingerpad'], right_pad_contact=pads['right_fingerpad'],
                both_pad_contact=both_pads, left_finger_contact=pads['left_finger'],
                right_finger_contact=pads['right_finger'], any_finger_contact=any_finger,
                finger_contact_without_both_pads=any_finger and not both_pads,
                bowl_in_eef_x_m=float(offset[0]), bowl_in_eef_y_m=float(offset[1]),
                bowl_in_eef_z_m=float(offset[2]), relative_slip_mm=float(slip),
                bowl_plate_contact=contact, xy_distance_m=xy, above_plate=above,
                xy_within_30mm=bool(xy < 0.03),
                success=bool(contact and xy < 0.03 and above))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model-name', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=6694)
    p.add_argument('--hold-close', nargs=2, type=int, metavar=('START', 'END'),
                   help='Inclusive action interval whose gripper is forced closed; '
                        'omit for a zero-intervention control run')
    p.add_argument('--reference-actions',
                   help='Archived actions.npy for the same state, enabling the divergence check')
    p.add_argument('--expect-init-sha256')
    p.add_argument('--expect-post-settle-sha256')
    p.add_argument('--save-videos', action='store_true')
    a = p.parse_args()

    from eval_protocol import MAX_STEPS, protocol_hash, validate_episode_results
    from finetune_utils import photometric_window, stable_seed

    window = parse_window(a.hold_close)
    protocol = json.loads(Path(a.protocol).read_text())
    fingerprint = protocol_hash(protocol)
    corruption = protocol['conditions'][a.condition]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() or (out / 'episodes.jsonl').exists():
        raise FileExistsError('Use a new evaluation output directory')
    reference = np.load(a.reference_actions) if a.reference_actions else None

    # Import official helpers only after parsing protocol and environment setup.
    from examples.LIBERO.eval_files.eval_libero import (
        LIBERO_DUMMY_ACTION, _binarize_gripper_open, _get_libero_env, benchmark)
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    np.random.seed(protocol['seed'])
    suite = benchmark.get_benchmark_dict()[protocol['suite']]()
    if max(protocol['task_ids']) >= suite.n_tasks:
        raise ValueError('Task ID outside this suite')
    states = {tid: suite.get_task_init_states(tid) for tid in protocol['task_ids']}
    if any(max(protocol['episode_ids']) >= len(v) for v in states.values()):
        raise ValueError('Episode ID exceeds available LIBERO initial states')

    client = ModelClient(policy_ckpt_path=a.ckpt, host='127.0.0.1', port=a.port, image_size=[224, 224])
    results = []
    for task_id in protocol['task_ids']:
        task = suite.get_task(task_id)
        for episode_id in protocol['episode_ids']:
            env_seed = stable_seed(protocol['seed'], protocol['suite'], task_id, episode_id) % (2**31)
            env, instruction = _get_libero_env(task, 256, env_seed)
            record = dict(task_id=task_id, episode_id=episode_id, env_seed=env_seed,
                          instruction=instruction, task_name=task.name,
                          condition=a.condition, success=False)
            images, actions, override_log = [], [], []
            state_rows, chunks, query_times, offsets = [], [], [], []
            start_episode = time.perf_counter()
            try:
                client.reset(task_description=instruction)
                env.reset()
                initial = states[task_id][episode_id]
                record['init_state_sha256'] = hashlib.sha256(np.asarray(initial).tobytes()).hexdigest()
                obs = env.set_init_state(initial)
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                record['post_settle_state_sha256'] = hashlib.sha256(
                    np.asarray(env.sim.get_state().flatten()).tobytes()).hexdigest()
                for key, expected in (('init_state_sha256', a.expect_init_sha256),
                                      ('post_settle_state_sha256', a.expect_post_settle_sha256)):
                    if expected and record[key] != expected:
                        raise ValueError(f'{key} differs from the archived run; pairing is broken')
                inner = env.env
                bowl = inner.objects_dict['akita_black_bowl_1']
                plate = inner.objects_dict['plate_1']
                bid = inner.obj_body_id['akita_black_bowl_1']
                pid = inner.obj_body_id['plate_1']
                grip = env.robots[0].gripper
                if not set(FINGER_GROUPS) <= set(grip.important_geoms):
                    raise ValueError('Gripper model lacks finger groups')
                from scipy.spatial.transform import Rotation
                for step in range(MAX_STEPS[protocol['suite']]):
                    eef_pos = np.asarray(obs['robot0_eef_pos'], dtype=float)
                    bowl_pos = env.sim.data.body_xpos[bid].copy()
                    offset = Rotation.from_quat(np.asarray(obs['robot0_eef_quat'], dtype=float)).as_matrix() \
                        .T @ (bowl_pos - eef_pos)
                    offsets.append(offset)
                    state_rows.append(state_record(step, obs, env, inner, bowl, plate, bid, pid, grip,
                                                  offset, slip_series(offsets)[-1]))
                    # Preserve the official 180-degree camera rotation and 224px resize.
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
                    seed = stable_seed(protocol['seed'], task_id, episode_id,
                                       step // client.action_chunk_size)
                    example = dict(image=viewed, lang=str(instruction), seed=seed)
                    horizon = client.goal_horizon_rows
                    goal_source = 'generate' if (client._cached_goal is None or
                                                 (horizon is not None and client._steps_since_refresh >= horizon)) \
                        else 'cached'
                    example['goal_horizon_rows'] = float(max(1.0, (horizon or 1) - client._steps_since_refresh))
                    is_query = step % client.action_chunk_size == 0
                    started = time.perf_counter()
                    response = client.step(example=example, step=step)
                    if is_query:
                        seconds = time.perf_counter() - started
                        query_times.append(dict(step=step, seconds=seconds, refresh=goal_source == 'generate'))
                        chunks.append(dict(step=step, goal_source=goal_source,
                                           goal_horizon_rows=example['goal_horizon_rows'], seed=seed,
                                           chunk_position=int(step % client.action_chunk_size),
                                           actions=np.asarray(client.raw_actions, dtype=float)))
                    raw = response['raw_action']
                    action = np.concatenate([
                        np.asarray(raw['world_vector'], dtype=np.float32).reshape(-1),
                        np.asarray(raw['rotation_delta'], dtype=np.float32).reshape(-1),
                        _binarize_gripper_open(raw['open_gripper'])])
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError('Invalid robot action')
                    model_gripper = float(action[6])
                    action, overridden = override_gripper(action, step, window)
                    actions.append(action)
                    override_log.append(dict(action_index=step, overridden=overridden,
                                             model_gripper_command=model_gripper,
                                             executed_gripper_command=float(action[6])))
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        record['success'] = True
                        break
                record.update(steps=len(actions), seconds=time.perf_counter() - start_episode,
                              query_times=query_times, video_frames=len(images),
                              video_fps=20, video_time_base='simulation',
                              video_layout='rows: primary/wrist; columns: original/observed')
            finally:
                env.close()
            stem = f'task_{task_id:02d}_episode_{episode_id:03d}'
            actions = np.stack(actions)
            np.save(out / (stem + '_actions.npy'), actions)
            write_csv(out / f'{stem}_override_log.csv', override_log)
            write_csv(out / f'{stem}_chunk_log.csv', chunk_records(chunks))
            write_csv(out / f'{stem}_state_log.csv', state_rows)
            (out / 'metadata.json').write_text(json.dumps(dict(
                coordinates='MuJoCo world frame, metres; object body origins',
                eef_quaternion='xyzw', state_index='0 is settled; k is after k actions',
                action_index='row k applies from state k to k+1', control_hz=20,
                gripper_sign='+1 close, -1 open, per the official client binarisation',
                success_predicate='bowl body z >= plate body z AND contact AND XY distance < 0.03 m',
                grasp_proxy='Both fingerpad groups contact the bowl; not a stability guarantee',
                contact_groups={name: grip.important_geoms[name] for name in FINGER_GROUPS},
                slip_note='per-step change of the bowl offset from the EEF origin, expressed in the EEF '
                          'frame, matching the column analyze_contact_transfer.py recomputes so the '
                          'closed-loop and offline metrics are directly comparable',
                hold_close_inclusive=list(window) if window else None,
                intervention_rule='only action[6] inside the window; the six arm channels are never modified',
                post_window_rule='no gripper override and no added release gate after the window',
                inference_cadence=f'action_chunk_size={client.action_chunk_size}, '
                                  f'goal_horizon_rows={client.goal_horizon_rows}',
                chunk_log='one row per query step, holding the full returned action block',
                scope='Closed loop with policy inference; intervention limited to the gripper channel'), indent=2))
            if a.save_videos and images:
                import imageio.v2 as imageio
                imageio.mimwrite(out / (stem + '.mp4'), images, fps=20)
            record['gripper_overridden_steps'] = sum(1 for r in override_log if r['overridden'])
            record['divergence_step_vs_reference'] = first_divergence_step(actions, reference)
            record['action_sha256'] = hashlib.sha256(actions.tobytes()).hexdigest()
            results.append(record)
            with (out / 'episodes.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
    validate_episode_results(results, protocol)
    successes = sum(r['success'] for r in results)
    summary = dict(schema_version=1, model=a.model_name, checkpoint=str(Path(a.ckpt).resolve()),
                   protocol_sha256=fingerprint, condition=a.condition, corruption=corruption,
                   successes=successes, episodes=len(results), success_rate=successes / len(results),
                   hold_close_inclusive=list(window) if window else None,
                   episode_results=results,
                   timing_scope='synchronous official client, cached goal cadence')
    (out / 'results.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'episode_results'}, indent=2))


if __name__ == '__main__':
    main()