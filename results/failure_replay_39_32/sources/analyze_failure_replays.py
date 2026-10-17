"""Analyze logged motion and predicate components; no extra rollout or training."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw

ROOT=Path('/root/autodl-tmp/wam_glance_project')
MODEL=ROOT/'GlanceWAM';OUT=MODEL/'results/failure_replay_39_32'
assert (OUT/'REPLAY_COMPLETE').exists()
checks=json.loads((OUT/'replay_checks.json').read_text());assert len(checks)==6
COLORS=dict(base='tab:blue',clean='tab:orange',aug='tab:green')
all_metrics=[]
for condition,episode in [('clean',39),('color_strong',32)]:
    case=OUT/f'{condition}_episode{episode}'
    positions,paxes=plt.subplots(3,3,figsize=(12,8),sharex=True)
    diagnostics,axes=plt.subplots(2,3,figsize=(13,7))
    readers=[];lengths=[]
    for model in COLORS:
        path=case/model
        with (path/'states.csv').open() as f:rows=list(csv.DictReader(f))
        def array(names):return np.asarray([[float(r[n]) for n in names] for r in rows])
        def flags(name):return np.asarray([r[name]=='True' for r in rows])
        times=array(['time_s'])[:,0];eef=array(['eef_'+a for a in 'xyz'])
        bowl=array(['bowl_'+a for a in 'xyz']);plate=array(['plate_'+a for a in 'xyz'])
        quats=array(['eef_q'+a for a in 'xyzw']);relative=np.einsum('nji,nj->ni',Rotation.from_quat(quats).as_matrix(),bowl-eef)
        xy=np.linalg.norm(bowl[:,:2]-plate[:,:2],axis=1);lift=bowl[:,2]-bowl[0,2]
        width=array(['finger_width_m'])[:,0];tilt=array(['bowl_tilt_deg'])[:,0]
        contact=flags('bowl_plate_contact');grasp=flags('both_pad_contact');success=flags('success')
        above=flags('above_plate');aligned=xy<.03
        npz=np.load(path/'replay.npz');actions=npz['actions']
        lifted=np.flatnonzero(lift>.02);start=int(lifted[0]) if len(lifted) else None
        candidates=np.arange(start if start is not None else 0,len(rows));closest=int(candidates[np.argmin(xy[candidates])])
        first=lambda values:int(np.flatnonzero(values)[0]) if np.any(values) else None
        metric=dict(model=model,condition=condition,episode_id=episode,success=bool(success.any()),
            steps=len(actions),first_both_pad_contact_state=first(grasp),first_lift_20mm_state=start,
            max_lift_mm=float(lift.max()*1000),min_xy_after_lift_mm=float(xy[closest]*1000),closest_xy_state=closest,
            closest_z_difference_mm=float((bowl[closest,2]-plate[closest,2])*1000),
            contact_at_closest_xy=bool(contact[closest]),both_pad_contact_at_closest_xy=bool(grasp[closest]),
            final_xy_mm=float(xy[-1]*1000),final_z_difference_mm=float((bowl[-1,2]-plate[-1,2])*1000),
            final_bowl_plate_contact=bool(contact[-1]),final_xy_within_30mm=bool(aligned[-1]),final_above=bool(above[-1]),
            max_plate_displacement_mm=float(np.linalg.norm(plate-plate[0],axis=1).max()*1000),
            gripper_switch_action_indices=(np.flatnonzero(np.diff(actions[:,6])!=0)+1).tolist(),
            open_command_after_lift=bool(np.any(actions[start:,6]<0)) if start is not None else None,
            states_xy_aligned_after_lift=int(aligned[candidates].sum()),
            states_contact_after_lift=int(contact[candidates].sum()),
            states_aligned_and_above_without_contact=int((aligned[candidates]&above[candidates]&~contact[candidates]).sum()),
            max_bowl_tilt_deg=float(tilt.max()),
            max_bowl_in_eef_offset_drift_after_lift_mm=float(np.linalg.norm(relative[candidates]-relative[candidates[0]],axis=1).max()*1000))
        all_metrics.append(metric)
        for col,(name,values) in enumerate([('EEF',eef),('Bowl body origin',bowl),('Plate body origin',plate)]):
            for row,axis in enumerate('xyz'):
                paxes[row,col].plot(times,values[:,row],color=COLORS[model],label=model)
                paxes[row,col].set(title=f'{name} {axis}',ylabel='World position (m)')
                if row==2:paxes[row,col].set_xlabel('Simulation seconds')
        axes[0,0].plot(times,xy*1000,color=COLORS[model],label=model)
        axes[0,1].plot(times,lift*1000,color=COLORS[model])
        axes[0,2].plot(times,(bowl[:,2]-plate[:,2])*1000,color=COLORS[model])
        axes[1,0].plot(times,width*1000,color=COLORS[model])
        axes[1,1].step(times,contact.astype(int)+list(COLORS).index(model)*1.5,where='post',color=COLORS[model],label=model)
        axes[1,2].plot(times,tilt,color=COLORS[model])
        # Useful derived positions are exported with the original world-frame logs.
        with (path/'derived.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['state_step','time_s','xy_mm','lift_mm','bowl_minus_plate_z_mm','bowl_in_eef_x_m','bowl_in_eef_y_m','bowl_in_eef_z_m'])
            for i,t in enumerate(times):w.writerow([i,t,xy[i]*1000,lift[i]*1000,(bowl[i,2]-plate[i,2])*1000,*relative[i]])
        reader=imageio.get_reader(MODEL/'results/color_holdout'/model/condition/f'task_08_episode_{episode:03d}.mp4',format='FFMPEG')
        readers.append(reader);lengths.append(reader.count_frames())
    paxes[0,0].legend();positions.suptitle(f'{condition}, initial state {episode}: saved-action replay')
    positions.tight_layout();positions.savefig(case/'world_positions.png',dpi=150);plt.close(positions)
    axes[0,0].axhline(30,color='black',linestyle='--',label='30mm success threshold');axes[0,0].legend()
    axes[0,1].axhline(20,color='grey',linestyle=':',label='20mm analysis marker')
    axes[0,2].axhline(0,color='black',linestyle='--')
    titles=['Bowl/plate horizontal distance (mm)','Bowl lift relative to start (mm)',
            'Bowl minus plate origin height (mm)','Actual finger opening (mm)',
            'Bowl/plate contact (offset per model)','Bowl tilt from world up (degrees)']
    for ax,title in zip(axes.flat,titles):ax.set(title=title,xlabel='Simulation seconds');ax.grid(alpha=.2)
    axes[1,1].legend();diagnostics.suptitle(f'{condition}, state {episode}; end traces at original stop')
    diagnostics.tight_layout();diagnostics.savefig(case/'placement_diagnostics.png',dpi=150);plt.close(diagnostics)
    last=[None]*3
    with imageio.get_writer(case/'original_policy_comparison.mp4',format='FFMPEG',fps=20,
            codec='libx264',ffmpeg_params=['-threads','1']) as writer:
        for step in range(max(lengths)):
            canvas=Image.new('RGB',(672,480),'white');draw=ImageDraw.Draw(canvas)
            for i,(model,reader,n) in enumerate(zip(COLORS,readers,lengths)):
                if step<n:last[i]=reader.get_data(step)
                f=last[i];observed=np.concatenate([f[:224,224:448],f[224:448,224:448]],axis=0)
                canvas.paste(Image.fromarray(observed),(i*224,32))
                draw.text((i*224+3,2),f'{model} state {min(step,n-1)}'+(' held' if step>=n else ''),fill='black')
                draw.text((i*224+3,17),'Original policy recording',fill='black')
            writer.append_data(np.asarray(canvas))
    for reader in readers:reader.close()
    reader=imageio.get_reader(case/'original_policy_comparison.mp4',format='FFMPEG');assert reader.count_frames()==max(lengths);reader.close()

(OUT/'metrics.json').write_text(json.dumps(all_metrics,indent=2))
with (OUT/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(all_metrics[0]));w.writeheader();w.writerows(all_metrics)
sources=OUT/'sources';sources.mkdir(exist_ok=True)
for name in ['replay_failure_cases.py','analyze_failure_replays.py']:shutil.copy2(MODEL/'day1'/name,sources/name)
manifest={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ROOT/'wam_failure_replay_39_32.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT.parent))
print(json.dumps(all_metrics,indent=2))
