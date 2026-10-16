"""LIBERO OSC_POSE axis diagnostics + deterministic action replay; no policy/GPU.

Run with LIBERO Python, MUJOCO_GL=osmesa on headless CPU Linux.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import gc
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from eval_protocol import CONDITIONS
from finetune_utils import photometric_window


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_sequence(axis, sign, amplitude, pulse_steps, hold_steps, gripper=-1.):
    """Initial neutral period, pulse on one axis, then neutral hold."""
    actions = np.zeros((hold_steps * 2 + pulse_steps, 7), dtype=np.float64)
    actions[:, 6] = gripper
    if axis == 6:
        # Test both transitions. Starting both cases open would make the
        # negative command a no-op, rather than testing the opening response.
        actions[:hold_steps, 6] = -sign
        actions[hold_steps:, 6] = sign * amplitude
    elif axis is not None:
        actions[hold_steps:hold_steps + pulse_steps, axis] = sign * amplitude
    return actions


def pose(obs):
    return np.r_[obs['robot0_eef_pos'], obs['robot0_eef_quat'],
                 obs['robot0_gripper_qpos']].astype(np.float64)


def motion(start, end):
    # MuJoCo/robosuite observable quaternion is xyzw. World-frame rotation delta.
    rot = Rotation.from_quat(end[3:7]) * Rotation.from_quat(start[3:7]).inv()
    return dict(translation_m=(end[:3] - start[:3]).tolist(),
                rotation_world_rotvec_rad=rot.as_rotvec().tolist(),
                gripper_qpos_delta=(end[7:] - start[7:]).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--suite', default='libero_goal')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--amplitude', type=float, default=.1)
    p.add_argument('--pulse-steps', type=int, default=10)
    p.add_argument('--hold-steps', type=int, default=10)
    p.add_argument('--replay-atol', type=float, default=1e-7)
    p.add_argument('--actions', help='Optional trusted Nx7 .npy fixed action sequence')
    p.add_argument('--asset-profile', help='Diagnostic texture profile JSON, if using a copy')
    a = p.parse_args()
    if not 0 < a.amplitude <= .25 or min(a.pulse_steps, a.hold_steps) < 1:
        p.error('Use amplitude in (0,.25] and positive step counts')
    if a.size < 64 or a.size % 2 or min(a.task, a.episode, a.seed) < 0 or a.replay_atol < 0:
        p.error('Invalid size/index/seed/tolerance')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    asset_profile = (json.loads(Path(a.asset_profile).read_text()) if a.asset_profile else
                     dict(mode='original'))
    # Set before importing simulator; no EGL / CUDA requirement on Linux CPU.
    os.environ.setdefault('MUJOCO_GL', 'osmesa' if os.name != 'nt' else 'glfw')
    if os.environ['MUJOCO_GL'] == 'osmesa':
        os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
    import imageio.v2 as imageio
    # Benchmark imports torch solely to deserialize state files. Keep its RSS
    # out of the rendering process on small CPU containers; child exits first.
    metadata_code = '''
import json,sys,numpy as np,torch
from pathlib import Path
from libero.libero import benchmark,get_libero_path
a=json.loads(sys.argv[1]);out=Path(sys.argv[2])
suite=benchmark.get_benchmark_dict()[a['suite']]();task=suite.get_task(a['task'])
path=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
states=torch.load(path,map_location='cpu',weights_only=False)
np.save(out/'initial_state.npy',np.asarray(states[a['episode']],dtype=np.float64))
metadata=dict(instruction=task.language,bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file))
(out/'task.json').write_text(json.dumps(metadata))
'''
    subprocess.run([sys.executable, '-c', metadata_code, json.dumps(vars(a)), str(out)], check=True)
    task = json.loads((out / 'task.json').read_text())
    initial = np.load(out / 'initial_state.npy', allow_pickle=False)
    bddl = task['bddl']
    from libero.libero.envs import OffScreenRenderEnv

    def create():
        # Fresh environment for every pulse and replay, identical RNG and state.
        np.random.seed(a.seed)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=a.size,
                                camera_widths=a.size, control_freq=20, hard_reset=False)
        try:
            env.seed(a.seed)
            env.reset()
            obs = env.set_init_state(initial.copy())
            for _ in range(10):
                obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
            return env, obs
        except BaseException:
            env.close()
            raise

    def frame(obs, text):
        # Match official GlanceWAM evaluation orientation: rotate both axes 180deg.
        cameras = [np.ascontiguousarray(obs[k][::-1, ::-1])
                   for k in ('agentview_image', 'robot0_eye_in_hand_image')]
        columns = []
        for name, factors in CONDITIONS.items():
            views = photometric_window(cameras, **factors)
            column = Image.fromarray(np.concatenate([np.asarray(v) for v in views], axis=0))
            tile = Image.new('RGB', (a.size, a.size * 2 + 36))
            tile.paste(column, (0, 36))
            tag = (f" CPU-tex{asset_profile['texture_max_size']}"
                   if asset_profile['mode'] == 'cpu_diagnostic_textures' else '')
            ImageDraw.Draw(tile).text((4, 3), name + tag + '\n' + text, fill='white')
            columns.append(np.asarray(tile))
        return np.concatenate(columns, axis=1)

    worker_peaks = []

    def rollout_in_process(name, actions, video=True):
        env, obs = create()
        poses, states, done_steps = [pose(obs)], [env.get_sim_state().copy()], []
        writer = None
        try:
            if video:
                writer = imageio.get_writer(out / (name + '.mp4'), fps=20,
                                           codec='libx264', macro_block_size=1,
                                           output_params=['-threads', '1'])
                writer.append_data(frame(obs, name + ' initial'))
            if name == 'zero':
                ctrl = env.robots[0].controller
                fields = ('input_min', 'input_max', 'output_min', 'output_max',
                          'control_ori', 'use_delta', 'kp', 'damping_ratio')
                config = {key: np.asarray(getattr(ctrl, key)).tolist()
                          for key in fields if hasattr(ctrl, key)}
                config.update(controller_class=type(ctrl).__name__, control_freq=20,
                              action_spec=[np.asarray(v).tolist() for v in env.env.action_spec])
                (out / 'controller.json').write_text(json.dumps(config, indent=2))
                del ctrl
            for step, action in enumerate(actions):
                obs, _, done, _ = env.step(action.tolist())
                poses.append(pose(obs))
                states.append(env.get_sim_state().copy())
                if done:
                    done_steps.append(step)
                # Diagnostic runs use a fixed length even if the task succeeds.
                if writer:
                    writer.append_data(frame(obs, f'{name} step {step + 1}'))
            if video:
                imageio.imwrite(out / (name + '_final.png'), frame(obs, name + ' final'))
        finally:
            if writer is not None:
                writer.close()
            env.close()
            gc.collect()
            if sys.platform.startswith('linux'):
                import ctypes
                ctypes.CDLL('libc.so.6').malloc_trim(0)
        poses, states = np.stack(poses), np.stack(states)
        if not np.isfinite(poses).all() or not np.isfinite(states).all():
            raise ValueError('Non-finite physical trajectory')
        peak_rss = 0.
        if sys.platform.startswith('linux'):
            import resource
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        np.savez_compressed(out / (name + '.npz'), actions=actions, poses=poses, states=states,
                            done_steps=np.asarray(done_steps, dtype=np.int64), peak_rss_mib=peak_rss)
        return poses, states, done_steps

    def rollout(name, actions, video=True):
        if not sys.platform.startswith('linux'):
            return rollout_in_process(name, actions, video)
        # OSMesa contexts retain substantial native allocations across envs.
        # Fork before any GL context is created; exit reclaims every allocation.
        import multiprocessing as mp
        worker = mp.get_context('fork').Process(target=rollout_in_process,
                                                args=(name, actions, video))
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f'Rollout {name} worker exit code {worker.exitcode}')
        with np.load(out / (name + '.npz'), allow_pickle=False) as data:
            worker_peaks.append(float(data['peak_rss_mib']))
            return data['poses'].copy(), data['states'].copy(), data['done_steps'].tolist()

    report = dict(description='Control interface diagnostics; no learned policy or robustness score',
                  args=vars(a), instruction=task['instruction'], conditions=CONDITIONS,
                  torch_loaded_in_sim_process='torch' in sys.modules, hard_reset=False,
                  numba_disable_jit=os.environ.get('NUMBA_DISABLE_JIT', '0'),
                  execution_mode='separate rollout processes' if sys.platform.startswith('linux') else 'single process',
                  visual_asset_profile=asset_profile,
                  initial_state_sha256=digest(initial), camera_transform='rotate180',
                  visual_pairing='three transforms of each identical physics frame',
                  packages={x: importlib.metadata.version(x) for x in
                            ['mujoco', 'robosuite', 'numpy', 'scipy', 'imageio']},
                  single_axis=[])
    csv_rows, traces = [], {}
    cases = [('zero', None, 0)] + [(f'{axis}_{label}', i, sign)
                for i, axis in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz'])
                for label, sign in [('pos', 1), ('neg', -1)]]
    cases += [('gripper_pos', 6, 1), ('gripper_neg', 6, -1)]
    for name, axis, sign in cases:
        actions = make_sequence(axis, sign, 1. if axis == 6 else a.amplitude,
                                a.pulse_steps, a.hold_steps)
        poses, states, done_steps = rollout(name, actions)
        start, end = a.hold_steps, a.hold_steps + a.pulse_steps
        result = dict(name=name, axis=axis, sign=sign, pulse=motion(poses[start], poses[end]),
                      total=motion(poses[0], poses[-1]), task_done_steps=done_steps)
        report['single_axis'].append(result)
        traces[name] = poses
        for step, values in enumerate(poses):
            csv_rows.append([name, step, step / 20, *values])
        print(json.dumps(result), flush=True)
    with (out / 'poses.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'step', 'time_s', 'x_m', 'y_m', 'z_m',
                         'qx', 'qy', 'qz', 'qw', 'finger0_qpos', 'finger1_qpos'])
        writer.writerows(csv_rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, (axis, ax) in enumerate(zip(['x', 'y', 'z', 'rx', 'ry', 'rz'], axes.flat)):
        for label in ['pos', 'neg']:
            trace = traces[f'{axis}_{label}']
            if i < 3:
                values = (trace[:, i] - trace[0, i]) * 1000
            else:
                rotations = Rotation.from_quat(trace[:, 3:7]) * Rotation.from_quat(trace[0, 3:7]).inv()
                values = np.rad2deg(rotations.as_rotvec()[:, i - 3])
            ax.plot(np.arange(len(trace)) / 20, values, label=label)
        ax.axvspan(a.hold_steps / 20, (a.hold_steps + a.pulse_steps) / 20,
                   color='gray', alpha=.15, label='pulse')
        ax.set(title=axis, xlabel='Time (s)', ylabel='Delta (mm)' if i < 3 else 'World rotation (deg)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(out / 'axis_response.png', dpi=150)
    plt.close(fig)

    if a.actions:
        fixed = np.load(a.actions, allow_pickle=False)
        if fixed.ndim != 2 or fixed.shape[1] != 7 or not len(fixed) or not np.isfinite(fixed).all():
            raise ValueError('Expected finite Nx7 actions')
        if np.max(np.abs(fixed)) > 1:
            raise ValueError('Actions outside normalized [-1,1] range')
    else:
        # Small scripted sequence, no grasping success claim.
        fixed = np.concatenate([make_sequence(i, s, a.amplitude, a.pulse_steps, a.hold_steps)
                                for i, s in [(2, 1), (0, 1), (1, 1), (1, -1), (0, -1)]])
    np.save(out / 'fixed_actions.npy', fixed)
    p1, s1, _ = rollout('fixed_replay', fixed)
    p2, s2, _ = rollout('fixed_replay_repeat', fixed, video=False)
    error = float(np.max(np.abs(s1 - s2)))
    report['replay'] = dict(actions_sha256=digest(fixed), steps=len(fixed),
                            max_abs_state_error=error, atol=a.replay_atol,
                            passed=bool(np.allclose(s1, s2, atol=a.replay_atol, rtol=0)),
                            states_sha256=[digest(s1), digest(s2)])
    if sys.platform.startswith('linux'):
        import resource
        report['peak_sim_rss_mib'] = max(worker_peaks, default=0.)
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['replay']), flush=True)
    if not report['replay']['passed']:
        raise RuntimeError('Replay mismatch: inspect states before using paired comparisons')


if __name__ == '__main__':
    main()
