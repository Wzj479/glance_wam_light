"""Compare existing base/C/intervention traces; restore-contact output is input only."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/autodl-tmp/wam_glance_project');MODEL=ROOT/'GlanceWAM'
OUT=MODEL/'results/counterfactual33_contact_68_80'
meta=json.loads((OUT/'restore_checks.json').read_text())
assert meta['states_inspected']==meta['flags_match_count']==39
assert meta['all39_object_position_and_rotation_checks_passed']
contact_rows=json.loads((OUT/'state_contact_summary.json').read_text())
contacts={(r['variant'],r['state_step']):r for r in contact_rows}
models=['base','C','hold56_75'];data={};export=[]
sources=dict(base=MODEL/'results/failure_replay_33/color_strong_episode33/base',
    C=MODEL/'results/failure_replay_33/color_strong_episode33/C',
    hold56_75=MODEL/'results/counterfactual_33_hold56_75/hold56_75')
left=set(meta['important_geoms']['left_finger']);right=set(meta['important_geoms']['right_finger'])
for r in contact_rows:
    geoms=set(r['bowl_contact_geoms'])
    r['whole_left_finger_contact']=bool(geoms&left)
    r['whole_right_finger_contact']=bool(geoms&right)
    r['both_whole_finger_contact']=bool(geoms&left and geoms&right)
for model,source in sources.items():
    with (source/'states.csv').open() as f:states=list(csv.DictReader(f))
    with (source/'actions.csv').open() as f:action_rows=list(csv.DictReader(f))
    actions=np.array([[float(r[k]) for k in ['dx','dy','dz','rx','ry','rz','gripper']] for r in action_rows])
    array=lambda prefix:np.array([[float(r[prefix+a]) for a in 'xyz'] for r in states])
    eef=array('eef_')*1000;bowl=array('bowl_')*1000;plate=array('plate_')*1000
    lift=bowl[:,2]-bowl[0,2];xy=np.linalg.norm(bowl[:,:2]-plate[:,:2],axis=1)
    rel=np.array([[contacts[model,k]['geometry_bowl_in_eef_'+a+'_mm'] for a in 'xyz'] for k in range(68,81)])
    geometry_eef=np.array([[contacts[model,k]['geometry_eef_'+a+'_m']*1000 for a in 'xyz'] for k in range(68,81)])
    first_negative_z=56+int(np.flatnonzero(actions[56:76,2]<0)[0])
    data[model]=dict(states=states,actions=actions,eef=eef,bowl=bowl,plate=plate,lift=lift,xy=xy,
        rel=rel,geometry_eef=geometry_eef,first_negative_z_after56=first_negative_z)
    for k in range(64,81):
        r=states[k]
        entry=dict(variant=model,**r,lift_mm=float(lift[k]),bowl_plate_xy_mm=float(xy[k]))
        for j,a in enumerate(['dx','dy','dz','rx','ry','rz','gripper']):entry['action_'+a]=float(actions[k,j]) if k<len(actions) else None
        if k>=68:
            c=contacts[model,k]
            entry.update({a:c[a] for a in ['restored_left_pad','restored_right_pad','whole_left_finger_contact',
                'whole_right_finger_contact','restored_plate_contact']})
            entry.update({f'geometry_bowl_in_eef_{a}_mm':float(v) for a,v in zip('xyz',rel[k-68])})
        else:
            for a in ['restored_left_pad','restored_right_pad','whole_left_finger_contact','whole_right_finger_contact',
                    'restored_plate_contact','geometry_bowl_in_eef_x_mm','geometry_bowl_in_eef_y_mm','geometry_bowl_in_eef_z_mm']:entry[a]=None
        export.append(entry)
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv(OUT/'trajectory_actions_64_80.csv',export)
(OUT/'contact_groups.json').write_text(json.dumps(contact_rows,indent=2))

decomp={};cf=data['hold56_75']
for reference in ['base','C']:
    ref=data[reference];i=76-68
    ee=cf['geometry_eef'][i]-ref['geometry_eef'][i]
    offset=(cf['bowl'][76]-cf['geometry_eef'][i])-(ref['bowl'][76]-ref['geometry_eef'][i])
    delta=cf['bowl'][76]-ref['bowl'][76]
    np.testing.assert_allclose(delta,ee+offset,rtol=0,atol=1e-10)
    decomp[reference]=dict(state=76,bowl_world_difference_mm=delta.tolist(),
        eef_world_difference_mm=ee.tolist(),bowl_minus_eef_world_offset_difference_mm=offset.tolist())
shift=cf['rel'][73-68]-cf['rel'][71-68]
metrics={m:dict(first_negative_z_after56=d['first_negative_z_after56'],
    bowl_lift68_mm=float(d['lift'][68]),bowl_lift72_mm=float(d['lift'][72]),
    bowl_plate_xy72_mm=float(d['xy'][72]),bowl_plate_xy76_mm=float(d['xy'][76]),
    bowl_minus_plate_xy76_mm=(d['bowl'][76,:2]-d['plate'][76,:2]).tolist(),
    eef_xyz76_mm=d['geometry_eef'][8].tolist(),
    first_no_whole_gripper_contact_in68_80=next((k for k in range(68,81) if not contacts[m,k]['any_gripper_contact']),None))
    for m,d in data.items()}
overview=dict(metrics=metrics,state76_decomposition=decomp,
    cf_relative_delta71_73_mm=shift.tolist(),cf_relative_delta71_73_norm_mm=float(np.linalg.norm(shift)),
    restore_flags_verified=39,maximum_logged_eef_sensor_vs_geometry_mm=max(r['logged_eef_vs_geometry_distance_mm'] for r in meta['alignment_checks']),
    scope='Read-only inspection of existing state33 trajectories; no action replay, policy inference, training or states45-49')
(OUT/'overview.json').write_text(json.dumps(overview,indent=2))
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,constrained_layout=True)
for model,d in data.items():
    k=np.arange(64,81)
    for ax,values in zip(axes.flat,[d['eef'][k,2],d['lift'][k],d['eef'][k,0],d['xy'][k],d['eef'][k,1],None]):
        if values is not None:ax.plot(k,values,'-o',ms=3,label=model)
    end=min(81,len(d['actions']));axes[2,1].step(np.arange(64,end),d['actions'][64:end,6],where='post',label=model)
titles=['Logged EEF world z (mm)','Bowl lift from initial (mm)','Logged EEF world x (mm)',
    'Bowl/plate horizontal distance (mm)','Logged EEF world y (mm)','Gripper command (+1 close / -1 open)']
for ax,title in zip(axes.flat,titles):
    ax.set(title=title,xlim=(64,80),xlabel='State/action index; 20Hz');ax.axvline(72,color='gray',ls=':');ax.axvline(76,color='gray',ls='--');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[1,1].axhline(30,color='black',ls='--',lw=.8)
fig.savefig(OUT/'motion_and_alignment.png',dpi=130);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
for j,m in enumerate(models):
    rows=[contacts[m,k] for k in range(68,81)];k=np.arange(68,81);ax=axes[0,j]
    pads=[int(r['restored_left_pad'])+int(r['restored_right_pad']) for r in rows]
    fingers=[int(r['whole_left_finger_contact'])+int(r['whole_right_finger_contact']) for r in rows]
    ax.step(k,pads,where='post',label='Pad groups')
    ax.step(k,fingers,where='post',label='Whole finger groups',ls='--')
    ax.step(k,[int(r['restored_plate_contact']) for r in rows],where='post',label='Bowl/plate contact',ls=':')
    ax.set(title=m,yticks=[0,1,2],ylim=(-.1,2.2),xlabel='State index');ax.legend(fontsize=8)
for j,a in enumerate('xyz'):
    ax=axes[1,j]
    for m,d in data.items():ax.plot(np.arange(68,81),d['rel'][:,j],'-o',ms=3,label=m)
    ax.set(title='Bowl in aligned EEF frame '+a,ylabel='mm',xlabel='State index');ax.legend(fontsize=8)
for ax in axes.flat:ax.axvline(72,color='gray',ls=':');ax.axvline(76,color='gray',ls='--');ax.grid(alpha=.2)
fig.savefig(OUT/'contact_and_relative_position.png',dpi=130);plt.close(fig)
fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True,constrained_layout=True)
for m,d in data.items():
    k=np.arange(64,80)
    for j,channel in enumerate([0,1,2,6]):axes[j].step(k,d['actions'][64:80,channel],where='post',label=m)
for ax,title in zip(axes,['dx command','dy command','dz command','Gripper command']):
    ax.set(ylabel=title);ax.axvline(72,color='gray',ls=':');ax.axvline(76,color='gray',ls='--');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[-1].set_xlabel('Recorded action index; normalized OSC commands')
fig.savefig(OUT/'action_commands_64_79.png',dpi=125);plt.close(fig)

c=decomp['C'];lines=['# 33号保持闭爪后68–80：接触转移与未对齐','',
 '结论：72时是指垫接触转移到非指垫手指碰撞体，并伴随物体在末端系滑移；不是夹爪完全失去接触。76未对齐主要由原失败回合保留下来的六维搬运轨迹落后、抬升不足和y方向偏移造成，再叠加夹持相对偏移。不能把固定动作反事实的后半程直接当成新观测下P策略的闭环输出。','',
 '## 72附近发生了什么','',
 '- 保持闭爪的68–71：左指垫、左手指碰撞体、右手指碰撞体接触碗；右指垫没有接触。base/C此时两侧指垫都有接触。',
 '- 72–73：左指垫也失去接触，但左右手指非指垫碰撞体仍接触碗，闭爪命令保持+1，手指开度仍在变小。',
 f"- 71→73，同步碰撞几何下碗在末端系的位置变化为{shift.round(3).tolist()}mm，合计{np.linalg.norm(shift):.2f}mm。这是相对滑移和接触点迁移的证据；没有接触力历史，不能再断言由摩擦系数、夹持力或哪个单一因素导致。",
 '- 首次碗盘接触是74；68–80均未见夹爪与盘直接接触。因此不能把72的指垫接触消失直接归因于已发生的碗盘/夹爪盘碰撞。',
 '- 74–77仍有左右手指碰撞体接触；76恢复原始张爪命令后，到78才没有任何夹爪几何接触碗。','',
 '## 为什么76仍没有对齐','',
 '|模型|56后首次负z动作|68时抬升mm|72时碗盘XY距离mm|76时碗盘XY距离mm|','|---|---:|---:|---:|---:|']
for m,s in metrics.items():lines.append(f"|{m}|{s['first_negative_z_after56']}|{s['bowl_lift68_mm']:.2f}|{s['bowl_plate_xy72_mm']:.2f}|{s['bowl_plate_xy76_mm']:.2f}|")
lines+=['',
 '负z动作表示归一化控制输入，不等同于该瞬间末端立即下降。P的保留动作从64就转为负z，base/C则在73；68附近CF已达到低位搬运的峰值，之后带碗向下并继续向目标移动。相比base/C，其低轨迹更早与盘发生偏心接触。',
 f"76时CF碗盘XY误差为{metrics['hold56_75']['bowl_minus_plate_xy76_mm']}mm。相对C，碗世界位置差={np.round(c['bowl_world_difference_mm'],2).tolist()}mm；其中末端世界位置差={np.round(c['eef_world_difference_mm'],2).tolist()}mm，碗相对末端世界偏移差={np.round(c['bowl_minus_eef_world_offset_difference_mm'],2).tolist()}mm。三项满足逐轴精确相加，不把x方向分量当成总XY误差百分比。",
 '大部分x方向落后来自末端尚未搬到相同位置，另有物体相对末端的偏移。76与C用相同动作时刻松爪，不代表位于相同的物理阶段；继续按时间延迟释放不能保证先完成抬升和对齐。',
 '重要边界：原P在56张爪、58附近失去原指垫接触后，后续64等动作是根据原失败回合的观测生成的。现在只改夹爪却复用这些六维动作，后半程观察与当初模型输入不同；所以不能把本次低轨迹直接归因为“P在抓取成功后仍必然规划错误”。','',
 '## 最小改进方案（本次不执行训练）','',
 '1. 立即修正离线诊断指标：保留指垫接触，另报左右完整手指接触、物体相对末端滑移，以及碗盘位置。不要把pad=0直接判为完全脱持；这些仿真真值只用于评测，不作为模型部署输入或释放门控。',
 '2. 有GPU后只做33号一次带反馈的闭环诊断：冻结原P检查点，唯一干预仍为动作56–75保持闭爪，但后续六维动作随新观测重新推理，不再回放原失败六维动作。保持初态、种子、8步动作块、目标刷新周期一致，记录阈值前夹爪输出、生成目标/缓存来源、原始动作块和真实接触位姿；76起取消夹爪覆盖，执行重新推理出的动作块原输出，不加新的按坐标释放门控。',
 '3. 用该一次诊断分流：若搬运/对齐恢复，原来的后半段低轨迹主要是固定失败动作的局限，优先研究夹爪相位预测；若仍低位搬运，再在同一保存观测、同一采样噪声下比较base/C/P输出，结合保存的生成目标检查目标条件与动作解码，不先凭本次开环轨迹增加抬升损失。任何训练候选都先回到C均匀采样作对照，阶段重采样P暂不推进；颜色单变量适配仍只是候选，不能宣称一定解决夹持。',
 '不搜索新的固定松爪时刻，不拼接C的搬运动作，不追加训练，不使用45–49。干预闭环属于诊断，不能当成未改模型本身的性能提升。','',
 '## 可信度与文件','',
 f"恢复了3条已有轨迹×13个保存状态，共39个状态。MuJoCo Euler积分dt={meta['physics_timestep_s']}秒；积分后qpos需逆最后一次位置积分才能对齐日志缓存的碰撞几何。对齐后39个状态的碗盘位置/旋转均通过1e-9容差核对，全部39组指垫/碗盘接触标志与原日志相同。EEF观测缓存与碰撞几何的最大差为{overview['maximum_logged_eef_sensor_vs_geometry_mm']:.3f}mm；相对位置及76分解使用同步几何位姿，运动概览另明确标记logged EEF。",
 '仅恢复已有状态并重算碰撞几何，没有执行动作、推理或训练；未恢复原控制输入与求解器状态，因此不解释重算接触力。body原点高度差不是碗底到盘面的净间隙。',
 '`motion_and_alignment.png`、`action_commands_64_79.png`、`contact_and_relative_position.png`为曲线；`trajectory_actions_64_80.csv`为动作与位姿；`contacts.csv/json`为完整相关碰撞体名称、距离和接触点；`restore_checks.json`记录全部对齐和接触一致性检查；`overview.json`记录误差分解。']
(OUT/'阶段68_80分析与最小方案.md').write_text('\n'.join(lines),encoding='utf-8')
src=OUT/'sources';src.mkdir(exist_ok=True)
for p in [MODEL/'day1/inspect_cf33_contacts.py',MODEL/'day1/analyze_cf33_transport.py']:shutil.copy2(p,src/p.name)
shutil.copy2(ROOT/'cf33_contacts.log',OUT/'inspection.log')
manifest={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ROOT/'wam_cf33_transport.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT.parent))
print(json.dumps(overview,indent=2),flush=True)
