"""Single-demo timing sensitivity: CPU recorded-action replay, not policy latency."""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS']:
    os.environ.setdefault(_key, '2')
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
from accept_action_interface_cpu import create, dump, frame, pose


CASES = [('baseline', 'baseline', 0)]
CASES += [(f'all_delay_{d}', 'all_delay', d) for d in [1, 2, 4]]
CASES += [(f'gripper_delay_{d}', 'gripper_delay', d) for d in [1, 2, 4]]
CASES += [(f'hold_{d}', 'hold', d) for d in [2, 4]]


def schedule(original, kind, amount, tail=24):
    """All cases share N+24 ticks; delays are flushed, never truncate the demo."""
    original = np.asarray(original, np.float64)
    if original.ndim != 2 or original.shape[1] != 7 or not np.isfinite(original).all():
        raise ValueError('Expected finite Nx7 environment commands')
    n = len(original)
    if tail < amount or kind not in ['baseline', 'all_delay', 'gripper_delay', 'hold']:
        raise ValueError('Invalid timing protocol')
    result = np.zeros((n + tail, 7))
    result[:, 6] = original[-1, 6]
    result[:n] = original
    if kind == 'all_delay':
        result[:] = [0.] * 6 + [-1.]
        result[amount:amount+n] = original
        result[amount+n:, 6] = original[-1, 6]
    elif kind == 'gripper_delay':
        result[:amount, 6] = -1.
        result[amount:amount+n, 6] = original[:, 6]
        result[amount+n:, 6] = original[-1, 6]
    elif kind == 'hold':
        if amount < 1:
            raise ValueError('Hold period must be positive')
        result[:n] = original[(np.arange(n) // amount) * amount]
    return result


def worker(args, case, video=False, suffix=''):
    name, kind, amount = case
    source = Path(args.source)
    meta = json.loads((source/'match.json').read_text())
    assert hashlib.sha256((source/'matched_demo.npz').read_bytes()).hexdigest() == meta['extracted_npz_sha256']
    with np.load(source/'matched_demo.npz', allow_pickle=False) as z:
        initial, original = z['initial_state'].copy(), z['env_actions'].copy()
    actions = schedule(original, kind, amount)
    env, obs = create(args, initial, render=video)
    writer = None
    if video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(Path(args.out)/(name+'.mp4'), fps=20, codec='libx264',
            macro_block_size=1, output_params=['-threads','1','-crf','30'])
    poses, states, objects, success = [], [], [], []
    try:
        for i, action in enumerate(actions):
            poses.append(pose(obs)); states.append(env.get_sim_state().copy())
            objects.append(np.r_[env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']],
                                 env.sim.data.body_xpos[env.env.obj_body_id['plate_1']]])
            if writer:
                writer.append_data(frame(obs, f'{name} t={i/20:.2f}s'))
            obs, _, _, _ = env.step(action.tolist())
            success.append(bool(env.check_success()))
        poses.append(pose(obs)); states.append(env.get_sim_state().copy())
        objects.append(np.r_[env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']],
                             env.sim.data.body_xpos[env.env.obj_body_id['plate_1']]])
        if writer:
            writer.append_data(frame(obs, f'{name} final success={success[-1]}'))
    finally:
        if writer: writer.close()
        env.close()
    poses, states, objects = np.asarray(poses), np.asarray(states), np.asarray(objects)
    assert np.isfinite(poses).all() and np.isfinite(states).all()
    dest = Path(args.out)/(name+suffix)
    np.savez_compressed(dest.with_suffix('.npz'), actions=actions, poses=poses, states=states,
                        objects=objects, success=np.asarray(success))
    edges = (np.flatnonzero(actions[1:,6] != actions[:-1,6]) + 1).tolist()
    import resource
    result = dict(case=name, kind=kind, amount=amount, control_hz=20,
        command_refresh_hz=20/amount if kind=='hold' else 20,
        delay_ms=amount*50 if 'delay' in kind else 0,
        original_actions=len(original), executed_actions=len(actions), common_horizon_s=len(actions)/20,
        final_success=success[-1], ever_success=any(success),
        first_success_step=next((i for i, value in enumerate(success) if value), None),
        bowl_lift_m=float(objects[:,2].max()-objects[0,2]),
        final_bowl_plate_xy_distance_m=float(np.linalg.norm(objects[-1,:2]-objects[-1,3:5])),
        final_bowl_plate_z_difference_m=float(objects[-1,2]-objects[-1,5]),
        gripper_transition_steps=edges,
        actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        initial_postsettle_state_sha256=hashlib.sha256(states[0].tobytes()).hexdigest(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        video=video, scope='One recorded demonstration, no policy or inference latency measurement')
    dump(dest.with_suffix('.json'), result)
    print(json.dumps(result), flush=True)


def spawn(args, case, video=False, suffix=''):
    process = mp.get_context('fork').Process(target=worker, args=(args,case,video,suffix))
    process.start(); process.join()
    if process.exitcode:
        raise RuntimeError(f'{case[0]} worker exit {process.exitcode}')


def summarize(args):
    root = Path(args.out)
    reports = [json.loads((root/(name+'.json')).read_text()) for name,_,_ in CASES]
    assert len({r['initial_postsettle_state_sha256'] for r in reports}) == 1
    assert len({r['executed_actions'] for r in reports}) == 1
    assert reports[0]['final_success'], 'Baseline failed: reject timing experiment'
    columns = ['case','kind','amount','delay_ms','command_refresh_hz','final_success','ever_success',
               'bowl_lift_m','final_bowl_plate_xy_distance_m','first_success_step','peak_rss_mib']
    with (root/'summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=columns, extrasaction='ignore');writer.writeheader();writer.writerows(reports)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    labels=[r['case'] for r in reports]
    for ax,field,title,mult in [(axes[0],'final_success','Final task predicate',1),
                                (axes[1],'bowl_lift_m','Maximum bowl lift (mm)',1000),
                                (axes[2],'final_bowl_plate_xy_distance_m','Final bowl/plate XY distance (mm)',1000)]:
        ax.bar(labels,[r[field]*mult for r in reports]);ax.tick_params(axis='x',rotation=65);ax.set_title(title)
    fig.savefig(root/'sensitivity_summary.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
    for name,_,_ in CASES:
        z=np.load(root/(name+'.npz'));t=np.arange(len(z['poses']))/20
        axes[0].plot(t,np.sum(np.abs(z['poses'][:,7:]),axis=1)*1000,label=name)
        axes[1].plot(t,(z['objects'][:,2]-z['objects'][0,2])*1000,label=name)
        axes[2].plot(t,np.linalg.norm(z['objects'][:,:2]-z['objects'][:,3:5],axis=1)*1000,label=name)
    for ax,title in zip(axes,['Actual finger width (mm)','Bowl lift (mm)','Bowl/plate XY distance (mm)']):
        ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Time from first command (s)')
    fig.savefig(root/'timing_traces.png',dpi=110);plt.close(fig)
    # Prespecified rule: smallest gripper delay causing final failure; otherwise
    # largest gripper delay. Hold=4 is always shown, irrespective of its outcome.
    failures=[r for r in reports if r['kind']=='gripper_delay' and not r['final_success']]
    selected=['baseline', failures[0]['case'] if failures else 'gripper_delay_4', 'hold_4']
    dump(root/'summary.json', dict(status='passed',cases=reports,selected_video_cases=selected,
        selection_rule='baseline + smallest failing gripper delay (fallback 4) + hold4',
        equal_initial_state=True,equal_horizon=True,
        interpretation=['Whole-action delay is a startup time shift, not closed-loop sensor-to-action latency',
                        'Hold repeats one command; it is not execution of a predicted multi-action chunk',
                        'One demonstration; no statistical success-rate or policy robustness claim']))
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--bddl',required=True);p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    for case in CASES:
        if not (args.resume and (root/(case[0]+'.json')).exists()):spawn(args,case)
    selected=summarize(args)
    for name in selected:
        spawn(args,next(c for c in CASES if c[0]==name),video=True,suffix='_video_check')
        a=np.load(root/(name+'.npz'));b=np.load(root/(name+'_video_check.npz'))
        assert np.max(np.abs(a['states']-b['states']))<1e-7, 'Rendering replay changed physics'
    dump(root/'video_checks.json',dict(status='passed',cases=selected,render_and_nonrender_state_atol=1e-7))
    files=[p for p in root.iterdir() if p.is_file() and p.name!='SHA256SUMS.json']
    dump(root/'SHA256SUMS.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__=='__main__':main()
