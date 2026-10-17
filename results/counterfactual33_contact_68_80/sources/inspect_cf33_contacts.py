"""Restore saved states68..80 and inspect collision geometry, no actions or policy."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'4')
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import mujoco
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

ROOT=Path('/root/autodl-tmp/wam_glance_project');MODEL=ROOT/'GlanceWAM'
OUT=MODEL/'results/counterfactual33_contact_68_80'
if OUT.exists():assert not any(OUT.iterdir()), 'Refuse to overwrite existing contact results'
else:OUT.mkdir(parents=True)
sources=dict(base=MODEL/'results/failure_replay_33/color_strong_episode33/base',
    C=MODEL/'results/failure_replay_33/color_strong_episode33/C',
    hold56_75=MODEL/'results/counterfactual_33_hold56_75/color_strong_episode33/P')
protocol=json.loads((MODEL/'results/stage_v1/development/protocol.json').read_text())
record=next(r for r in json.loads((MODEL/'results/stage_v1/development/P/color_strong/results.json').read_text())['episode_results'] if r['episode_id']==33)
assert protocol['task_ids']==[8] and 33 in protocol['episode_ids']
np.random.seed(42)
suite=benchmark.get_benchmark_dict()[protocol['suite']]();task=suite.get_task(8)
t=time.monotonic()
env=ControlEnv(bddl_file_name=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
    camera_heights=256,camera_widths=256,use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False)
env.seed(record['env_seed']);env.reset()
integrator=int(env.sim.model.opt.integrator);timestep=float(env.sim.model.opt.timestep)
print('GEOMETRY_INTEGRATOR',integrator,'TIMESTEP',timestep,flush=True)
events=[];summaries=[];checks=[];initial=None;alignment_checks=[]
try:
    inner=env.env;robot=env.robots[0];gripper=robot.gripper
    bowl=inner.objects_dict['akita_black_bowl_1'];plate=inner.objects_dict['plate_1']
    bid=inner.obj_body_id['akita_black_bowl_1'];pid=inner.obj_body_id['plate_1']
    bowl_geoms=set(bowl.contact_geoms);plate_geoms=set(plate.contact_geoms)
    grip_geoms=set(gripper.contact_geoms)
    left=set(gripper.important_geoms['left_fingerpad']);right=set(gripper.important_geoms['right_fingerpad'])
    for name,source in sources.items():
        with (source/'states.csv').open() as f:stored=list(csv.DictReader(f))
        with np.load(source/'replay.npz') as z:
            states=z['full_states'].copy();stored_rotations=z['object_rotation_matrices'].copy()
        if initial is None:initial=states[0].copy()
        else:np.testing.assert_array_equal(initial,states[0])
        checks.append(dict(variant=name,source=str(source),npz_sha256=hashlib.sha256((source/'replay.npz').read_bytes()).hexdigest()))
        for step in range(68,81):
            env.sim.set_state_from_flattened(states[step]);env.sim.forward()
            row=stored[step]
            def pose_errors():
                errors=[]
                for prefix,pos in [('bowl_',env.sim.data.body_xpos[bid]),('plate_',env.sim.data.body_xpos[pid])]:
                    errors.append(float(np.max(np.abs(pos-[float(row[prefix+a]) for a in 'xyz']))))
                return errors
            direct_errors=pose_errors();alignment='saved_qpos'
            if max(direct_errors)>1e-9:
                # mj_step integrates qpos after updating xpos/contact. Reconstruct
                # its cached pre-integration geometry for Euler/implicit schemes,
                # and require exact agreement with all logged poses and rotations.
                assert int(env.sim.model.opt.integrator) in [0,2,3], 'Cannot invert RK4 with one position integration'
                mujoco.mj_integratePos(env.sim.model._model,env.sim.data.qpos,env.sim.data.qvel,
                    -float(env.sim.model.opt.timestep))
                env.sim.forward();alignment='inverse_last_position_integration'
            aligned_errors=pose_errors()
            assert max(aligned_errors)<1e-9, dict(variant=name,step=step,direct=direct_errors,aligned=aligned_errors)
            rotations=np.stack([env.sim.data.body_xmat[bid].reshape(3,3),env.sim.data.body_xmat[pid].reshape(3,3)])
            np.testing.assert_allclose(rotations,stored_rotations[step],rtol=0,atol=1e-9)
            alignment_checks.append(dict(variant=name,state_step=step,alignment=alignment,
                direct_position_max_error_m=max(direct_errors),aligned_position_max_error_m=max(aligned_errors),
                logged_eef_vs_geometry_distance_mm=float(np.linalg.norm(env.sim.data.site_xpos[robot.eef_site_id]-
                    [float(row['eef_'+a]) for a in 'xyz'])*1000)))
            for prefix,pos in [('bowl_',env.sim.data.body_xpos[bid]),('plate_',env.sim.data.body_xpos[pid])]:
                np.testing.assert_allclose(pos,[float(row[prefix+a]) for a in 'xyz'],rtol=0,atol=1e-9)
            selected=[]
            for i in range(env.sim.data.ncon):
                contact=env.sim.data.contact[i]
                g1=env.sim.model.geom_id2name(contact.geom1);g2=env.sim.model.geom_id2name(contact.geom2)
                pair={g1,g2}
                if not pair.intersection(bowl_geoms|grip_geoms):continue
                bg=next((g for g in [g1,g2] if g in bowl_geoms),None)
                other=g2 if bg==g1 else (g1 if bg==g2 else None)
                b1=env.sim.model.body_id2name(env.sim.model.geom_bodyid[contact.geom1])
                b2=env.sim.model.body_id2name(env.sim.model.geom_bodyid[contact.geom2])
                event=dict(variant=name,state_step=step,geom1=g1,geom2=g2,body1=b1,body2=b2,
                    distance_mm=float(contact.dist*1000),efc_address=int(contact.efc_address),
                    x_mm=float(contact.pos[0]*1000),y_mm=float(contact.pos[1]*1000),z_mm=float(contact.pos[2]*1000),
                    bowl_other_geom=other,bowl_left_pad=bool(bg and other in left),
                    bowl_right_pad=bool(bg and other in right),bowl_any_gripper=bool(bg and other in grip_geoms),
                    bowl_nonpad_gripper=bool(bg and other in grip_geoms-left-right),
                    bowl_plate=bool(bg and other in plate_geoms),
                    gripper_plate=bool(pair.intersection(grip_geoms) and pair.intersection(plate_geoms)))
                selected.append(event);events.append(event)
            lp=any(e['bowl_left_pad'] for e in selected);rp=any(e['bowl_right_pad'] for e in selected)
            pc=any(e['bowl_plate'] for e in selected)
            original_flags=[row[k]=='True' for k in ['left_pad_contact','right_pad_contact','bowl_plate_contact']]
            geometry_eef=env.sim.data.site_xpos[robot.eef_site_id].copy()
            geometry_rotation=env.sim.data.site_xmat[robot.eef_site_id].reshape(3,3)
            relative=geometry_rotation.T@(env.sim.data.body_xpos[bid]-geometry_eef)*1000
            summaries.append(dict(variant=name,state_step=step,restored_left_pad=lp,restored_right_pad=rp,
                restored_plate_contact=pc,logged_left_pad=original_flags[0],logged_right_pad=original_flags[1],
                logged_plate_contact=original_flags[2],flags_match_logged=original_flags==[lp,rp,pc],
                any_gripper_contact=any(e['bowl_any_gripper'] for e in selected),
                nonpad_gripper_contact=any(e['bowl_nonpad_gripper'] for e in selected),
                gripper_plate_contact=any(e['gripper_plate'] for e in selected),
                **{f'geometry_eef_{a}_m':float(v) for a,v in zip('xyz',geometry_eef)},
                **{f'geometry_bowl_in_eef_{a}_mm':float(v) for a,v in zip('xyz',relative)},
                bowl_contact_geoms=sorted({e['bowl_other_geom'] for e in selected if e['bowl_other_geom']})))
finally:env.close()
for name,rows in [('contacts',events),('state_contact_summary',summaries)]:
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2))
    with (OUT/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
meta=dict(mode='Saved qpos/qvel state restoration and mj_forward only; no actions, policy or training',
    episode_id=33,condition='color_strong',state_indices=list(range(68,81)),sources=checks,
    important_geoms=gripper.important_geoms,gripper_contact_geoms=sorted(grip_geoms),
    all39_object_position_and_rotation_checks_passed=True,flags_match_count=sum(r['flags_match_logged'] for r in summaries),
    states_inspected=len(summaries),
    integrator=integrator,physics_timestep_s=timestep,alignment_checks=alignment_checks,
    force_caveat='No contact forces interpreted. Align cached pre-integration geometry by inverse last qpos integration; require exact logged object world positions and rotations. EEF observation sensor cache is not synchronized with collision geometry: its difference is recorded, and geometry-relative positions use site pose from aligned qpos. Original actuator/control/solver state is not restored.',
    reserved_initial_states=list(range(45,50)),seconds=time.monotonic()-t,
    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(OUT/'restore_checks.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
for row in summaries:
    if row['variant']=='hold56_75':print(json.dumps(row),flush=True)
