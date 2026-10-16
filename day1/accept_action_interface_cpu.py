"""Inspect real OSC goals, physical axis responses, gripper mapping and demo replay.

Use the pinned low-texture LIBERO copy on a 2GiB CPU host. No Torch/policy loaded.
"""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS']:
    os.environ.setdefault(_name, '2')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def pose(obs):
    return np.r_[obs['robot0_eef_pos'], obs['robot0_eef_quat'], obs['robot0_gripper_qpos']]


def open_to_env(value):
    return 1. - 2. * (np.asarray(value) > .5)


def create(args, initial, render=False):
    from libero.libero.envs.env_wrapper import ControlEnv
    np.random.seed(42)
    env = ControlEnv(bddl_file_name=args.bddl, camera_heights=160, camera_widths=160,
        use_camera_obs=render, has_offscreen_renderer=render, hard_reset=False,
        ignore_done=True, control_freq=20)
    env.seed(42)
    env.reset()
    obs = env.set_init_state(initial.copy())
    for _ in range(getattr(args, 'settle_steps', 10)):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    return env, obs


def run_interface(args):
    initial = np.load(args.initial, allow_pickle=False)
    env, obs = create(args, initial)
    ctrl = env.robots[0].controller
    results = []
    for axis in range(6):
        for amplitude in [-2., -.2, .1, 2.]:
            action = np.zeros(6)
            action[axis] = amplitude
            ctrl.update(force=True)
            ctrl.goal_ori = ctrl.ee_ori_mat.copy()
            ctrl.set_goal(action)
            pos_error = ctrl.goal_pos - ctrl.ee_pos
            rot_error = Rotation.from_matrix(ctrl.goal_ori @ ctrl.ee_ori_mat.T).as_rotvec()
            actual = np.r_[pos_error, rot_error]
            expected = np.clip(action, -1, 1) * np.array([.05] * 3 + [.5] * 3)
            err = float(np.max(np.abs(actual - expected)))
            assert err < 2e-6, (axis, amplitude, actual, expected)
            results.append(dict(axis=axis, input=amplitude, expected_goal_delta=expected.tolist(),
                                measured_goal_delta=actual.tolist(), max_abs_error=err))
    mixed = np.array([0., 0., 0., .2, -.15, .1])
    ctrl.set_goal(mixed)
    current, goal = ctrl.ee_ori_mat.copy(), ctrl.goal_ori.copy()
    left = Rotation.from_rotvec(mixed[3:] * .5).as_matrix() @ current
    right = current @ Rotation.from_rotvec(mixed[3:] * .5).as_matrix()
    euler = Rotation.from_euler('xyz', mixed[3:] * .5).as_matrix() @ current
    rotation = dict(world_left_multiply_error=float(np.max(np.abs(left - goal))),
                    body_right_multiply_error=float(np.max(np.abs(right - goal))),
                    euler_xyz_error=float(np.max(np.abs(euler - goal))),
                    initial_eef_world_rotation=current.tolist())
    assert rotation['world_left_multiply_error'] < 2e-6
    assert rotation['body_right_multiply_error'] > .01 and rotation['euler_xyz_error'] > .001
    controller = dict(input_min=np.asarray(ctrl.input_min).tolist(), input_max=np.asarray(ctrl.input_max).tolist(),
        output_min=np.asarray(ctrl.output_min).tolist(), output_max=np.asarray(ctrl.output_max).tolist(),
        use_delta=bool(ctrl.use_delta), use_ori=bool(ctrl.use_ori), control_freq=20,
        kp=np.asarray(ctrl.kp).tolist(), class_name=type(ctrl).__name__)
    env.close()
    del env, ctrl
    # Every physical case gets a fresh controller, initial state and 10 settling steps.
    traces, rows, directions = {}, [], []
    cases = [('zero', None, 0, .1)]
    cases += [(f'{name}_{sign:+d}', axis, sign, .1) for axis, name in enumerate(['x','y','z','rx','ry','rz']) for sign in [1,-1]]
    cases += [('x_+0.2', 0, 1, .2), ('rz_+0.2', 5, 1, .2)]
    for name, axis, sign, amplitude in cases:
        env, obs = create(args, initial)
        actions = np.zeros((21, 7)); actions[:, 6] = -1
        if axis is not None:
            actions[5:13, axis] = sign * amplitude
        poses = [pose(obs)]
        for action in actions:
            obs, _, _, _ = env.step(action.tolist()); poses.append(pose(obs))
        env.close()
        traces[name] = np.asarray(poses)
        np.savez_compressed(Path(args.out) / (name + '.npz'), poses=poses, actions=actions)
        for step, values in enumerate(poses):rows.append([name, step, step / 20, *values])
        if axis is not None:
            p = traces[name]
            z = traces['zero']
            if axis < 3:
                vector = (p[13,:3] - p[5,:3]) - (z[13,:3] - z[5,:3])
                signed = float(vector[axis] * sign)
            else:
                rv = (Rotation.from_quat(p[13,3:7]) * Rotation.from_quat(p[5,3:7]).inv()).as_rotvec()
                rz = (Rotation.from_quat(z[13,3:7]) * Rotation.from_quat(z[5,3:7]).inv()).as_rotvec()
                vector = rv - rz; signed = float(vector[axis - 3] * sign)
            assert signed > 1e-5, (name, vector)
            directions.append(dict(case=name, signed_response=signed, vector=vector.tolist()))
        print('PHYSICAL_CASE', name, flush=True)
    env, obs = create(args, initial)
    grip = [pose(obs)]
    gripper_labels = np.r_[np.ones(5), np.zeros(25), np.ones(25)]
    for value in gripper_labels:
        obs, _, _, _ = env.step([0.] * 6 + [float(open_to_env(value))]); grip.append(pose(obs))
    grip = np.asarray(grip)
    widths = np.sum(np.abs(grip[:, 7:]), axis=1)
    assert widths[5] - widths[30] > .02 and widths[-1] - widths[30] > .02
    env.close()
    np.savez_compressed(Path(args.out) / 'gripper.npz', poses=grip, labels=gripper_labels,
                        commands=open_to_env(gripper_labels), widths=widths)
    with (Path(args.out) / 'interface_traces.csv').open('w', newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','step','time_s','x','y','z','qx','qy','qz','qw','finger1','finger2']);writer.writerows(rows)
    import resource
    dump(Path(args.out) / 'interface_report.json', dict(status='passed', controller=controller,
        goal_checks=results, rotation_convention=rotation, physical_direction_checks=directions,
        gripper=dict(open_label=1, closed_label=0, open_env_command=-1, closed_env_command=1,
            open_width_m=float(widths[5]), closed_width_m=float(widths[30]), reopened_width_m=float(widths[-1])),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        rendering=False, policy_loaded=False, initial_state_source=args.initial))


def frame(obs, text):
    from PIL import Image, ImageDraw
    cams = [obs[k][::-1, ::-1].copy() for k in ['agentview_image','robot0_eye_in_hand_image']]
    image = Image.fromarray(np.concatenate(cams, axis=1))
    tile=Image.new('RGB',(320,190),'white');tile.paste(image,(0,30))
    ImageDraw.Draw(tile).text((4,4),text+' | CPU texture1024',fill='black')
    return np.asarray(tile)


def run_video(args):
    import imageio.v2 as imageio
    initial=np.load(args.initial,allow_pickle=False)
    writer=imageio.get_writer(Path(args.out)/f'interface_axis{args.video_axis}.mp4',fps=20,codec='libx264',
        macro_block_size=1,quality=6,output_params=['-threads','1'])
    try:
        for axis,name in [(0,'world +x / -x'),(5,'world +rz / -rz'),(6,'gripper open / close')]:
            if axis != args.video_axis:continue
            env,obs=create(args,initial,render=True)
            try:
                for i in range(50):
                    action=np.array([0.] *6+[-1.])
                    if axis<6:
                        action[axis]=.15 if 5<=i<20 else (-.15 if 25<=i<40 else 0.)
                    else: action[6]=1 if 5<=i<30 else -1
                    writer.append_data(frame(obs,name+f' step {i}'))
                    obs,_,_,_=env.step(action.tolist())
                imageio.imwrite(Path(args.out)/f'interface_axis{axis}_last.jpg',frame(obs,name))
            finally:env.close()
    finally:writer.close()


def plot_interface(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for axis,(name,ax) in enumerate(zip(['x','y','z','rx','ry','rz'],axes.flat)):
        for sign in [1,-1]:
            p=np.load(out/(f'{name}_{sign:+d}.npz'))['poses']
            if axis<3:v=(p[:,axis]-p[0,axis])*1000
            else:v=(Rotation.from_quat(p[:,3:7])*Rotation.from_quat(p[0,3:7]).inv()).as_rotvec()[:,axis-3]*180/np.pi
            ax.plot(np.arange(len(p))/20,v,label=f'{sign:+d}')
        ax.axvspan(.25,.65,alpha=.15,color='gray');ax.set(title=name,xlabel='Time (s)',ylabel='mm' if axis<3 else 'World rotvec (deg)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'axis_response.png',dpi=110);plt.close(fig)
    report=json.loads((out/'interface_report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    tests=report['goal_checks']
    for axis,label in [(0,'translation x'),(5,'rotation z')]:
        part=[x for x in tests if x['axis']==axis]
        grid=np.linspace(-2,2,201)
        axes[0].plot(grid,np.clip(grid,-1,1)*(.05 if axis<3 else .5),label=label+' expected')
        axes[0].scatter([x['input'] for x in part],[x['measured_goal_delta'][axis] for x in part],marker='x',label=label+' measured')
    axes[0].set(xlabel='Controller input',ylabel='Goal delta (m / rad)',title='Controller scaling and clipping');axes[0].legend();axes[0].grid(alpha=.2)
    g=np.load(out/'gripper.npz');axes[1].plot(np.arange(len(g['widths']))/20,g['widths']*1000,label='finger width mm')
    axes[1].step(np.arange(len(g['labels']))/20,g['labels']*60,where='post',label='open label x60')
    axes[1].set(xlabel='Time (s)',title='Gripper 1=open, 0=closed');axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(out/'scale_gripper.png',dpi=110);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--bddl',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--mode',choices=['interface','video'],default='interface')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for axis in ([0] if args.mode=='interface' else [0,5,6]):
        args.video_axis=axis
        worker=mp.get_context('fork').Process(target=run_interface if args.mode=='interface' else run_video,args=(args,))
        worker.start();worker.join()
        if worker.exitcode:raise RuntimeError(f'{args.mode} worker failed: {worker.exitcode}')
    if args.mode=='video':
        import imageio_ffmpeg,subprocess
        listing=out/'concat.txt';listing.write_text(''.join(f"file '{(out/f'interface_axis{axis}.mp4').resolve()}'\n" for axis in [0,5,6]))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(out/'interface_control.mp4')],check=True)
    if args.mode=='interface':plot_interface(out)


if __name__=='__main__':main()
