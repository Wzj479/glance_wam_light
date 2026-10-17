"""Analyze one fixed recorded-action intervention; no model or simulation execution."""
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

ROOT=Path('/root/autodl-tmp/wam_glance_project');MODEL=ROOT/'GlanceWAM'
OUT=MODEL/'results/counterfactual_33_hold56_75'
NEW=OUT/'color_strong_episode33/P';OLD=MODEL/'results/failure_replay_33/color_strong_episode33/P'
assert (OUT/'REPLAY_COMPLETE').exists()
check=json.loads((OUT/'replay_checks.json').read_text())[0]
audit=check['intervention']
assert audit['hold_close_inclusive']==[56,75]
assert audit['changed_elements']==[[k,6] for k in range(56,76)]
assert audit['prefix_states_exactly_equal_through']==56
data={};metrics={}
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def longest(mask):
    best=current=0
    for value in mask:
        current=current+1 if value else 0;best=max(best,current)
    return best
for name,path in [('original_P',OLD),('hold56_75',NEW)]:
    with (path/'states.csv').open() as f:rows=list(csv.DictReader(f))
    def vec(prefix,axes='xyz'):return np.asarray([[float(r[prefix+a]) for a in axes] for r in rows])
    def flag(key):return np.asarray([r[key]=='True' for r in rows])
    with np.load(path/'replay.npz') as z:
        actions=z['actions'].copy();states=z['full_states'].copy()
    assert actions.shape==(300,7) and len(rows)==301
    eef=vec('eef_');bowl=vec('bowl_');plate=vec('plate_')
    rel=np.einsum('nji,nj->ni',Rotation.from_quat(vec('eef_q','xyzw')).as_matrix(),bowl-eef)*1000
    lift=(bowl[:,2]-bowl[0,2])*1000
    pads=flag('left_pad_contact').astype(int)+flag('right_pad_contact').astype(int)
    both=flag('both_pad_contact');success=flag('success')
    first=lambda mask:int(np.flatnonzero(mask)[0]) if np.any(mask) else None
    fixed=rel[60:76];drift=np.linalg.norm(fixed-fixed[0],axis=1)
    metrics[name]=dict(max_lift_mm=float(lift.max()),max_lift_states56_76_mm=float(lift[56:77].max()),
        max_lift_state=int(np.argmax(lift)),max_lift_states56_76_state=int(np.argmax(lift[56:77]))+56,
        first_lift20mm_state=first(lift>20),first_both_pad_contact_state=first(both),
        any_pad_contact_states56_76=int((pads[56:77]>0).sum()),
        first_no_pad_contact_state56_76=(int(np.flatnonzero(pads[56:77]==0)[0])+56) if np.any(pads[56:77]==0) else None,
        both_pad_contact_states56_76=int(both[56:77].sum()),
        longest_both_pad_run_states56_76=longest(both[56:77]),
        relative_position_drift_states60_75_from60_max_mm=float(drift.max()),
        first_bowl_plate_contact_state=first(flag('bowl_plate_contact')),
        first_success_state=first(success),final_success=bool(success[-1]),
        final_xy_mm=float(np.linalg.norm(bowl[-1,:2]-plate[-1,:2])*1000),
        xy_state76_mm=float(np.linalg.norm(bowl[76,:2]-plate[76,:2])*1000),
        lift_state76_mm=float(lift[76]),lift_state80_mm=float(lift[80]))
    export=OUT/name;export.mkdir(exist_ok=True)
    shutil.copy2(path/'states.csv',export/'states.csv')
    write_csv(export/'actions.csv',[dict(action_index=i,**{k:float(v) for k,v in zip(['dx','dy','dz','rx','ry','rz','gripper'],a)}) for i,a in enumerate(actions)])
    write_csv(export/'derived.csv',[dict(state_step=i,time_s=i/20,lift_mm=float(lift[i]),
        left_pad=bool(flag('left_pad_contact')[i]),right_pad=bool(flag('right_pad_contact')[i]),both_pad=bool(both[i]),
        **{f'bowl_in_eef_{a}_mm':float(v) for a,v in zip('xyz',rel[i])},
        relative_drift_from_state56_mm=float(np.linalg.norm(rel[i]-rel[56])),
        xy_mm=float(np.linalg.norm(bowl[i,:2]-plate[i,:2])*1000)) for i in range(len(rows))])
    data[name]=dict(rows=rows,actions=actions,states=states,lift=lift,pads=pads,rel=rel,
        width=np.asarray([float(r['finger_width_m'])*1000 for r in rows]),
        xy=np.linalg.norm(bowl[:,:2]-plate[:,:2],axis=1)*1000,eef=eef,bowl=bowl,plate=plate)
original=data['original_P'];new=data['hold56_75']
np.testing.assert_array_equal(original['states'][:57],new['states'][:57])
np.testing.assert_array_equal(original['actions'][:,:6],new['actions'][:,:6])
assert np.argwhere(original['actions']!=new['actions']).tolist()==[[i,6] for i in range(56,76)]
snapshot=[]
for name,d in data.items():
    for k in [54,55,56,57,58,60,64,68,72,75,76,77,80,88,100,150,300]:
        snapshot.append(dict(variant=name,**d['rows'][k],lift_mm=float(d['lift'][k]),
            **{f'bowl_in_eef_{a}_mm':float(v) for a,v in zip('xyz',d['rel'][k])}))
write_csv(OUT/'key_states.csv',snapshot)
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,constrained_layout=True)
for name,d in data.items():
    k=np.arange(301);a=d['actions']
    axes[0,0].step(np.arange(300),a[:,6],where='post',label=name)
    axes[0,1].plot(k,d['lift'],label=name)
    axes[1,0].step(k,d['pads'],where='post',label=name)
    axes[1,1].plot(k,d['width'],label=name)
    axes[2,0].plot(k,np.linalg.norm(d['rel']-d['rel'][56],axis=1),label=name)
    axes[2,1].plot(k,d['xy'],label=name)
titles=['Gripper command (+1 close / -1 open)','Bowl lift from initial (mm)',
    'Contacting fingerpad groups (0,1,2)','Actual finger opening (mm)',
    'Bowl-in-EEF displacement from state56 (mm)','Bowl/plate horizontal distance (mm)']
for ax,title in zip(axes.flat,titles):
    ax.set(title=title,xlim=(40,100),xlabel='State/action index; 20Hz')
    ax.axvspan(56,76,alpha=.09,color='gray');ax.axvline(76,color='gray',ls=':',lw=.8)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[1,0].set_yticks([0,1,2]);axes[2,1].axhline(30,color='black',ls='--',lw=.8)
fig.suptitle('State33: only P gripper actions56–75 changed; same six arm commands')
fig.savefig(OUT/'grasp_comparison.png',dpi=135);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
for j,axis in enumerate('xyz'):
    for name,d in data.items():axes[j].plot(np.arange(301),d['rel'][:,j],label=name)
    axes[j].set(ylabel='Bowl in EEF '+axis+' (mm)',xlim=(40,100))
    axes[j].axvspan(56,76,alpha=.09,color='gray');axes[j].grid(alpha=.2);axes[j].legend()
axes[-1].set_xlabel('State index; position R_eef.T @ (bowl_origin - eef_position)')
fig.savefig(OUT/'bowl_relative_to_eef.png',dpi=135);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
for name,d in data.items():
    k=np.arange(301)
    for ax,values in zip(axes.flat,[d['lift'],d['xy'],d['width'],d['pads']]):ax.plot(k,values,label=name)
for ax,title in zip(axes.flat,['Full replay bowl lift (mm)','Full replay bowl/plate XY (mm)',
    'Full replay finger opening (mm)','Full replay fingerpad contact count']):
    ax.set(title=title,xlabel='State index (300 actions total)');ax.axvspan(56,76,alpha=.09,color='gray');ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/'full_replay.png',dpi=120);plt.close(fig)
resource=json.loads((ROOT/'cf33_resource.json').read_text());assert resource['exit_code']==0
result=dict(intervention=audit,metrics=metrics,resource=resource,
    reserved_initial_states=list(range(45,50)),interpretation_scope='One fixed recorded-action sequence; not a re-inferred closed-loop policy or success-rate improvement')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2))
lines=['# 33号P：固定夹爪反事实回放','',
 '唯一干预：原P的动作56–75（两端包含，20条）夹爪设为+1闭合。其余六维及其他时间的夹爪命令逐项相同；完整执行300条动作，不提前结束、不调用模型。',
 '状态0–56与原P物理回放逐位一致，差异从干预之后才可能产生。原始物理资产、环境种子和稳定后状态哈希相同。无训练，未执行45–49。','',
 '|指标|原P|保持闭爪56–75|','|---|---:|---:|']
labels=[('max_lift_mm','全程最大抬升mm'),('max_lift_states56_76_mm','状态56–76最大抬升mm'),
    ('first_lift20mm_state','首次抬升超过20mm状态'),('first_both_pad_contact_state','首次双侧指垫接触状态'),
    ('any_pad_contact_states56_76','状态56–76至少单侧指垫接触观测数'),
    ('both_pad_contact_states56_76','状态56–76双侧接触观测数'),
    ('longest_both_pad_run_states56_76','该窗口最长连续双侧接触观测数'),
    ('relative_position_drift_states60_75_from60_max_mm','状态60–75相对60的碗在末端系最大位置漂移mm'),
    ('first_success_state','首次满足官方成功判据状态'),('final_success','状态300成功判据'),('final_xy_mm','最终碗盘水平距离mm')]
for key,label in labels:
    values=[metrics[n][key] for n in data]
    fmt=lambda v:f'{v:.3f}' if isinstance(v,float) else str(v)
    lines.append('|'+label+'|'+'|'.join(map(fmt,values))+'|')
old=metrics['original_P'];cf=metrics['hold56_75']
lines+=['','## 结论与限制','',
 f"固定延长闭爪后，状态56–76最大抬升由{old['max_lift_states56_76_mm']:.2f}mm增至{cf['max_lift_states56_76_mm']:.2f}mm，状态60–75的末端系位置漂移从{old['relative_position_drift_states60_75_from60_max_mm']:.2f}mm降至{cf['relative_position_drift_states60_75_from60_max_mm']:.2f}mm。在相同前缀物理状态和相同六维动作输入下，这支持提前张爪是原记录早期抬升失败的重要贡献因素。",
 f"干预窗口最高点在状态{cf['max_lift_states56_76_state']}；全程最高点在状态{cf['max_lift_state']}。后者不处于闭爪干预窗口，不能拿全程最大抬升替代干预阶段的抬升效果。",
 f"干预后仍没有同时接触两侧指垫的记录；状态56–76有{cf['any_pad_contact_states56_76']}个状态至少单侧接触，首次没有指垫接触是状态{cf['first_no_pad_contact_state56_76']}。但碗被带动、随后也发生碗盘接触，不能把pad代理为0直接当成所有夹爪几何都失去接触，或把相对漂移较小直接当成稳定受力夹持。",
 f"碗盘首次接触为状态{cf['first_bowl_plate_contact_state']}；动作76恢复原始松爪时，碗盘水平距离仍{cf['xy_state76_mm']:.2f}mm，超过官方30mm成功阈值。整段300动作未出现成功判据。因此，该固定延迟改善了早期抬升，却没有使这条动作序列完成任务，也不支持把固定延迟直接加入控制器并宣称模型提升。",
 '本次不搜索新的闭爪时长、不改六维动作、不追加训练。是否改变接近动作、生成目标或后续动作能进一步修复，均未在本实验中验证。']
lines+=['',
 '相对位置使用末端旋转矩阵的逆变换：R_eef.T @ (bowl_body_origin - eef_position)。固定状态60–75的漂移指标用于同一窗口比较，不是在结果出来后挑选稳定片段。双侧指垫接触只是代理指标，应与抬升和相对位置曲线一起判断。',
 '碗的抬升相对初始body原点；碗盘原点z差不等于碗底到盘面的间隙。动作k推进状态k到k+1，76恢复原始张爪命令。六维输入相同不保证实际末端轨迹完全相同，负载和接触会改变控制响应。',
 '完整执行了300条固定动作，分别报告首次成功判据与最终状态。即使出现成功判据，也不能作为策略成功率提升：干预后没有根据新观测重新推理动作，后续动作仍来自原失败回合。',
 f"CPU物理回放耗时{resource['seconds']:.2f}秒，峰值RSS约{resource['max_rss_kib']/1024**2:.2f}GiB。没有新渲染视频，曲线来自本次真实仿真日志。",'',
 '## 文件','',
 '`grasp_comparison.png`：动作、抬升、接触、实际开度与相对漂移；`bowl_relative_to_eef.png`：末端系XYZ位置；`full_replay.png`：完整300动作回放。',
 '`original_P/`与`hold56_75/`提供完整动作、世界位姿及派生CSV；`key_states.csv`提供关键时刻；原始反事实完整仿真状态和控制目标在`color_strong_episode33/P/replay.npz`。','',
 '## 复现','', '```bash','source /root/autodl-tmp/wam_glance_project/deploy_env.sh','cd "$WAM_MODEL_ROOT"',
 'export PYTHONPATH="$WAM_MODEL_ROOT:/root/autodl-tmp/wam_glance_project/LIBERO"',
 'export LIBERO_CONFIG_PATH=/root/autodl-tmp/wam_glance_project/LIBERO/libero',
 'export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl',
 '/root/wam_sim_env/bin/python day1/replay_failure_cases.py --physics-only \\',
 ' --source results/stage_v1/development --out results/counterfactual_33_hold56_75_new \\',
 ' --case color_strong:33 --models P --hold-close 56 75 \\',
 ' --reference-replay results/failure_replay_33/color_strong_episode33/P/replay.npz','```']
(OUT/'反事实回放记录.md').write_text('\n'.join(lines),encoding='utf-8')
sources=OUT/'sources';sources.mkdir(exist_ok=True)
for p in [MODEL/'day1/replay_failure_cases.py',MODEL/'day1/analyze_counterfactual33.py',ROOT/'run_cf33_cpu.py']:
    shutil.copy2(p,sources/p.name)
shutil.copy2(ROOT/'cf33_replay.log',OUT/'replay.log')
shutil.copy2(ROOT/'cf33_resource.json',OUT/'resource.json')
manifest={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(ROOT/'wam_counterfactual33.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT.parent))
print(json.dumps(result,indent=2),flush=True)
