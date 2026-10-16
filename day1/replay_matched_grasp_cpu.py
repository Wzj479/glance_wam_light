"""Replay uniquely matched real grasp actions from their original full sim state."""
import os
for _name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS']:
    os.environ.setdefault(_name,'2')
os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
os.environ.setdefault('NUMBA_DISABLE_JIT','1')
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from accept_action_interface_cpu import create, dump, frame, pose


def worker(args, name, settle, video):
    source=Path(args.source)
    meta=json.loads((source/'match.json').read_text())
    assert hashlib.sha256((source/'matched_demo.npz').read_bytes()).hexdigest()==meta['extracted_npz_sha256']
    data=np.load(source/'matched_demo.npz',allow_pickle=False)
    initial=data['initial_state'];actions=data['env_actions'];reference=data['lerobot_states']
    # Validate full state restoration before any settling steps.
    args.settle_steps=0
    env,obs=create(args,initial,render=video)
    restore=float(np.max(np.abs(env.get_sim_state()-initial)))
    assert restore<1e-12
    actual_names=[env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
    assert actual_names==meta['joint_names'],(actual_names,meta['joint_names'])
    for _ in range(settle):obs,_,_,_=env.step([0.]*6+[-1.])
    writer=None
    if video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(Path(args.out)/(name+'.mp4'),fps=20,codec='libx264',quality=7,
            macro_block_size=1,output_params=['-threads','1'])
    poses,states,objects,success=[],[],[],[]
    snapshots={0:'start',48:'closing',70:'transport',99:'release',111:'finish'}
    try:
        for i,action in enumerate(actions):
            poses.append(pose(obs));states.append(env.get_sim_state().copy())
            objects.append(np.r_[env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']],
                                 env.sim.data.body_xpos[env.env.obj_body_id['plate_1']]])
            if writer:
                picture=frame(obs,f'{name} step {i} open={data["lerobot_actions"][i,6]:.0f}')
                writer.append_data(picture)
                if i in snapshots:imageio.imwrite(Path(args.out)/(name+'_'+snapshots[i]+'.jpg'),picture)
            obs,_,_,_=env.step(action.tolist());success.append(bool(env.check_success()))
        poses.append(pose(obs));states.append(env.get_sim_state().copy())
        objects.append(np.r_[env.sim.data.body_xpos[env.env.obj_body_id['akita_black_bowl_1']],
                             env.sim.data.body_xpos[env.env.obj_body_id['plate_1']]])
        if writer:writer.append_data(frame(obs,name+f' final success={success[-1]}'))
    finally:
        if writer:writer.close()
        env.close()
    poses,states,objects=np.asarray(poses),np.asarray(states),np.asarray(objects)
    assert np.isfinite(poses).all() and np.isfinite(states).all()
    pos_error=np.linalg.norm(poses[:-1,:3]-reference[:,:3],axis=1)
    rot_error=(Rotation.from_quat(poses[:-1,3:7])*Rotation.from_rotvec(reference[:,3:6]).inv()).magnitude()
    grip_error=poses[:-1,7:]-reference[:,6:8]
    np.savez_compressed(Path(args.out)/(name+'.npz'),poses=poses,states=states,objects=objects,
        actions=actions,reference=reference,success=success,position_error=pos_error,rotation_error=rot_error)
    import resource
    report=dict(protocol=name,settling_steps=settle,steps=len(actions),matched_source_demo=meta['matched_original_demo'],
        initial_restore_max_abs_error=restore,all_joint_names_and_order_match=True,
        final_success=success[-1],first_success_step=next((i for i,s in enumerate(success) if s),None),
        reference_position_rmse_m=float(np.sqrt(np.mean(pos_error**2))),reference_position_max_error_m=float(pos_error.max()),
        reference_rotation_rmse_rad=float(np.sqrt(np.mean(rot_error**2))),
        reference_finger_qpos_rmse_m=float(np.sqrt(np.mean(grip_error**2))),
        initial_reference_position_error_m=float(pos_error[0]),
        bowl_lift_m=float(objects[:,2].max()-objects[0,2]),
        final_bowl_minus_plate_xyz_m=(objects[-1,:3]-objects[-1,3:]).tolist(),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        camera_rotation='180deg to match training display',visual_assets='CPU texture1024 diagnostic copy',
        note='Recorded open-loop actions; no learned policy, not a benchmark success rate')
    dump(Path(args.out)/(name+'.json'),report)
    print(json.dumps(report),flush=True)


def summarize(args):
    out=Path(args.out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports={name:json.loads((out/(name+'.json')).read_text()) for name in ['matched_settle10','matched_settle0','matched_repeat']}
    a=np.load(out/'matched_settle10.npz');b=np.load(out/'matched_repeat.npz')
    repeat_error=float(np.max(np.abs(a['states']-b['states'])))
    assert repeat_error<1e-7
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,label in [('matched_settle10','10 settling steps'),('matched_settle0','0 settling steps')]:
        d=np.load(out/(name+'.npz'));t=np.arange(len(d['poses']))/20
        axes[0,0].plot(t[:-1],d['position_error']*1000,label=label)
        axes[0,1].plot(t[:-1],np.rad2deg(d['rotation_error']),label=label)
    axes[0,0].set(title='Pre-action EEF position error vs LeRobot',xlabel='Time (s)',ylabel='mm')
    axes[0,1].set(title='EEF rotation geodesic error vs LeRobot',xlabel='Time (s)',ylabel='deg')
    t=np.arange(len(a['poses']))/20
    axes[1,0].plot(t,np.sum(np.abs(a['poses'][:,7:]),axis=1)*1000,label='replayed finger width')
    axes[1,0].plot(t[:-1],np.sum(np.abs(a['reference'][:,6:]),axis=1)*1000,'--',label='dataset finger width')
    axes[1,0].set(title='Gripper response',xlabel='Time (s)',ylabel='mm')
    axes[1,1].plot(t,(a['objects'][:,2]-a['objects'][0,2])*1000,label='bowl lift')
    axes[1,1].plot(t,np.linalg.norm(a['objects'][:,:2]-a['objects'][:,3:5],axis=1)*1000,label='bowl/plate XY distance')
    axes[1,1].set(title='Object motion',xlabel='Time (s)',ylabel='mm')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'replay_errors.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for i,ax in enumerate(axes):
        ax.plot(t,a['poses'][:,i],label='replayed EEF');ax.plot(t[:-1],a['reference'][:,i],'--',label='LeRobot reference')
        ax.set_ylabel(['world x (m)','world y (m)','world z (m)'][i]);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (s)');fig.savefig(out/'eef_reference.png',dpi=110);plt.close(fig)
    with (out/'replay_trace.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step','time_s','x','y','z','qx','qy','qz','qw','finger1','finger2','bowl_x','bowl_y','bowl_z','plate_x','plate_y','plate_z'])
        for i,(p,o) in enumerate(zip(a['poses'],a['objects'])):writer.writerow([i,i/20,*p,*o])
    dump(out/'replay_report.json',dict(source_match=json.loads((Path(args.source)/'match.json').read_text()),
        protocols=reports,repeat_max_abs_state_error=repeat_error,repeat_passed=True,
        scope='One matched recorded demonstration replay, no policy or benchmark score'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--bddl',required=True)
    args=p.parse_args();Path(args.out).mkdir(parents=True,exist_ok=True)
    for name,settle,video in [('matched_settle10',10,True),('matched_settle0',0,True),('matched_repeat',10,False)]:
        proc=mp.get_context('fork').Process(target=worker,args=(args,name,settle,video));proc.start();proc.join()
        if proc.exitcode:raise RuntimeError(f'{name} worker failed {proc.exitcode}')
    summarize(args)


if __name__=='__main__':main()
