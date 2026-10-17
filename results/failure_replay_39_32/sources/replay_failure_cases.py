"""Replay six saved action streams with original LIBERO assets; no policy inference."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'4')
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0,str(Path.cwd()))
from examples.LIBERO.eval_files.eval_libero import LIBERO_DUMMY_ACTION, _get_libero_env, benchmark

def sha(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()

def original_pair(obs):
    return np.concatenate([cv2.resize(np.ascontiguousarray(obs[key][::-1,::-1]),(224,224),
        interpolation=cv2.INTER_AREA) for key in ['agentview_image','robot0_eye_in_hand_image']],axis=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='results/color_holdout')
    p.add_argument('--out',default='results/failure_replay_39_32')
    args=p.parse_args();source=Path(args.source);out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    protocol=json.loads((source/'protocol.json').read_text())
    suite=benchmark.get_benchmark_dict()[protocol['suite']]()
    task=suite.get_task(8);initials=suite.get_task_init_states(8)
    summaries=[]
    for condition,episode in [('clean',39),('color_strong',32)]:
        for model in ['base','clean','aug']:
            src=source/model/condition;stem=f'task_08_episode_{episode:03d}'
            record=next(r for r in json.loads((src/'results.json').read_text())['episode_results'] if r['episode_id']==episode)
            actions=np.load(src/(stem+'_actions.npy'));assert actions.shape==(record['steps'],7)
            np.random.seed(protocol['seed'])
            env,_=_get_libero_env(task,256,record['env_seed'])
            traces=[];states=[];goals=[];rotations=[]
            dest=out/f'{condition}_episode{episode}'/model;dest.mkdir(parents=True)
            try:
                env.reset();initial=initials[episode]
                assert sha(initial)==record['init_state_sha256']
                obs=env.set_init_state(initial)
                for _ in range(10):obs,_,_,_=env.step(LIBERO_DUMMY_ACTION)
                settled=sha(env.sim.get_state().flatten())
                assert settled==record['post_settle_state_sha256'], 'Initial physical state differs'
                inner=env.env;ctrl=env.robots[0].controller
                bowl=inner.objects_dict['akita_black_bowl_1'];plate=inner.objects_dict['plate_1']
                bid=inner.obj_body_id['akita_black_bowl_1'];pid=inner.obj_body_id['plate_1']
                metadata=dict(coordinates='MuJoCo world frame, metres; object body origins, not mesh centre of mass',
                    eef_quaternion='xyzw',object_quaternion='wxyz',state_index='0 is settled; k is after k actions',
                    action_index='row k applies from state k to k+1',control_hz=20,
                    success_predicate='bowl body z >= plate body z AND contact AND XY distance < 0.03 m',
                    grasp_proxy='Both left and right fingerpad groups contact bowl at sample time; not a grasp stability guarantee',
                    bowl_contact_geoms=bowl.contact_geoms,plate_contact_geoms=plate.contact_geoms)
                for step in range(len(actions)+1):
                    bp=env.sim.data.body_xpos[bid].copy();pp=env.sim.data.body_xpos[pid].copy()
                    br=env.sim.data.body_xmat[bid].reshape(3,3).copy()
                    grip=env.robots[0].gripper
                    left=bool(inner.check_contact(grip.important_geoms['left_fingerpad'],bowl.contact_geoms))
                    right=bool(inner.check_contact(grip.important_geoms['right_fingerpad'],bowl.contact_geoms))
                    contact=bool(inner.check_contact(bowl,plate));xy=float(np.linalg.norm(bp[:2]-pp[:2]));above=bool(bp[2]>=pp[2])
                    success=bool(env.check_success())
                    assert success==(contact and xy<.03 and above)
                    row=dict(state_step=step,time_s=step/20,
                        **{f'eef_{axis}':float(v) for axis,v in zip('xyz',obs['robot0_eef_pos'])},
                        **{f'eef_q{axis}':float(v) for axis,v in zip('xyzw',obs['robot0_eef_quat'])},
                        **{f'bowl_{axis}':float(v) for axis,v in zip('xyz',bp)},
                        **{f'plate_{axis}':float(v) for axis,v in zip('xyz',pp)},
                        **{f'bowl_q{axis}':float(v) for axis,v in zip('wxyz',env.sim.data.body_xquat[bid])},
                        **{f'plate_q{axis}':float(v) for axis,v in zip('wxyz',env.sim.data.body_xquat[pid])},
                        finger_width_m=float(np.abs(obs['robot0_gripper_qpos']).sum()),
                        left_pad_contact=left,right_pad_contact=right,both_pad_contact=left and right,
                        bowl_plate_contact=contact,xy_distance_m=xy,above_plate=above,xy_within_30mm=xy<.03,
                        bowl_tilt_deg=float(np.degrees(np.arccos(np.clip(br[2,2],-1,1)))),success=success)
                    traces.append(row);states.append(env.sim.get_state().flatten().copy())
                    rotations.append(np.stack([br,env.sim.data.body_xmat[pid].reshape(3,3).copy()]))
                    if step==len(actions):break
                    obs,_,done,_=env.step(actions[step].tolist())
                    goals.append(np.r_[ctrl.goal_pos,np.asarray(ctrl.goal_ori).reshape(-1)])
                success_steps=[r['state_step'] for r in traces if r['success']]
                first=success_steps[0] if success_steps else None
                assert bool(success_steps)==record['success']
                assert first==(len(actions) if record['success'] else None), 'Original stop time not reproduced'
                # Compare raw-camera terminal frame to the original lossy MP4.
                reader=imageio.get_reader(src/(stem+'.mp4'),format='FFMPEG')
                frame=reader.get_data(len(actions));reader.close()
                prior=frame[:,0:224];current=original_pair(obs)
                mae=float(np.abs(prior.astype(float)-current.astype(float)).mean())
                assert mae<6, ('Terminal camera mismatch',mae)
                imageio.imwrite(dest/'terminal_replay.png',current)
                imageio.imwrite(dest/'terminal_original_decoded.png',prior)
            finally:env.close()
            with (dest/'states.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(traces[0]));w.writeheader();w.writerows(traces)
            np.savez_compressed(dest/'replay.npz',actions=actions,full_states=np.asarray(states),
                controller_goals=np.asarray(goals),object_rotation_matrices=np.asarray(rotations))
            (dest/'metadata.json').write_text(json.dumps(metadata,indent=2))
            summary=dict(model=model,condition=condition,episode_id=episode,steps=len(actions),
                original_success=record['success'],replayed_success=bool(success_steps),first_success_state_step=first,
                settled_sha256=settled,actions_file_sha256=hashlib.sha256((src/(stem+'_actions.npy')).read_bytes()).hexdigest(),
                terminal_original_video_mae=mae,trace_directory=str(dest))
            (dest/'replay_check.json').write_text(json.dumps(summary,indent=2))
            summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'replay_checks.json').write_text(json.dumps(summaries,indent=2))
    (out/'REPLAY_COMPLETE').touch()

if __name__=='__main__':main()
