"""Report state33 from existing saved-action replays; no environment/model execution."""
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

ROOT=Path('/root/autodl-tmp/wam_glance_project');MODEL=ROOT/'GlanceWAM'
OUT=MODEL/'results/failure_replay_33';CASE=OUT/'color_strong_episode33'
SOURCE=MODEL/'results/stage_v1/development'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

assert (OUT/'REPLAY_COMPLETE').exists()
plan=read(OUT/'replay_plan.json')
assert plan['cases']==[['color_strong',33]] and plan['models']==['base','C','P'] and plan['physics_only']
checks=read(OUT/'replay_checks.json');assert len(checks)==3
assert len({r['settled_sha256'] for r in checks})==1
assert all(r['original_success']==r['replayed_success'] for r in checks)
metrics={r['model']:r for r in read(OUT/'metrics.json')}
assert not metrics['P']['success'] and metrics['P']['first_lift_20mm_state'] is None
assert metrics['P']['min_xy_after_lift_mm'] is None
rows={};actions={};summary={};keyframes=[]
for m in ['base','C','P']:
    with (CASE/m/'states.csv').open() as f:rows[m]=list(csv.DictReader(f))
    source=SOURCE/m/'color_strong/task_08_episode_033_actions.npy'
    actions[m]=np.load(source)
    write_csv(CASE/m/'actions.csv',[dict(action_index=k,time_s=k/20,
        **{name:float(v) for name,v in zip(['dx','dy','dz','rx','ry','rz','gripper'],a)})
        for k,a in enumerate(actions[m])])
    with np.load(CASE/m/'replay.npz') as z:np.testing.assert_array_equal(actions[m],z['actions'])
    check=next(r for r in checks if r['model']==m);assert sha(source)==check['actions_file_sha256']
    close=int(np.flatnonzero(actions[m][:,6]>0)[0])
    release=int(np.flatnonzero((np.arange(len(actions[m]))>close)&(actions[m][:,6]<0))[0])
    summary[m]=dict(first_close_action=close,first_release_action=release,
        first_continuous_closed_command_seconds=(release-close)/20,
        first_both_pad_contact_state=metrics[m]['first_both_pad_contact_state'],
        max_lift_mm=metrics[m]['max_lift_mm'],first_lift_20mm_state=metrics[m]['first_lift_20mm_state'],
        first_bowl_plate_contact_state=metrics[m]['first_bowl_plate_contact_state'],
        success=metrics[m]['success'],steps=len(actions[m]),final_xy_mm=metrics[m]['final_xy_mm'])
    for k in [0,40,44,48,52,54,55,56,57,58,60,64,72,76,78,80,100,150,300]:
        if k>=len(rows[m]):continue
        r=rows[m][k];entry=dict(model=m,**r)
        for j,key in enumerate(['dx','dy','dz','rx','ry','rz','gripper']):
            entry['next_action_'+key]=float(actions[m][k,j]) if k<len(actions[m]) else None
        keyframes.append(entry)
write_csv(OUT/'key_states_and_actions.csv',keyframes)

def vector(row,prefix,axes):return np.array([float(row[prefix+a]) for a in axes])
deltas=[]
for ref in ['base','C']:
    for k in range(min(len(rows[ref]),len(rows['P']))):
        p=rows['P'][k];r=rows[ref][k]
        qp=Rotation.from_quat(vector(p,'eef_q','xyzw'));qr=Rotation.from_quat(vector(r,'eef_q','xyzw'))
        deltas.append(dict(reference=ref,state_step=k,
            eef_position_distance_mm=float(np.linalg.norm(vector(p,'eef_','xyz')-vector(r,'eef_','xyz'))*1000),
            eef_rotation_distance_deg=float(np.degrees((qr.inv()*qp).magnitude())),
            bowl_position_distance_mm=float(np.linalg.norm(vector(p,'bowl_','xyz')-vector(r,'bowl_','xyz'))*1000)))
write_csv(OUT/'P_pose_differences.csv',deltas)

fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,constrained_layout=True)
for m,color in zip(rows,['tab:blue','tab:orange','tab:green']):
    r=rows[m];k=np.arange(len(r));a=actions[m];b0=float(r[0]['bowl_z'])
    axes[0,0].step(np.arange(len(a)),a[:,6],where='post',label=m,color=color)
    axes[0,1].plot(k,[float(x['finger_width_m'])*1000 for x in r],label=m,color=color)
    axes[1,0].plot(k,[(float(x['bowl_z'])-b0)*1000 for x in r],label=m,color=color)
    axes[1,1].plot(k,[float(x['eef_z'])*1000 for x in r],label=m,color=color)
    pads=[int(x['left_pad_contact']=='True')+int(x['right_pad_contact']=='True') for x in r]
    axes[2,0].step(k,pads,where='post',label=m,color=color)
    axes[2,1].plot(k,[float(x['bowl_x'])*1000 for x in r],label=m,color=color)
for ax,title in zip(axes.flat,['Gripper command: +1 close / -1 open','Actual finger opening (mm)',
        'Bowl lift relative to initial (mm)','EEF world height (mm)','Contacting fingerpad groups (0,1,2)','Bowl body world x (mm)']):
    ax.set(title=title,xlim=(40,81),xlabel='State/action index (20Hz)')
    ax.axvline(56,color='black',ls='--',lw=.8,label='P opens: action56')
    ax.axvline(58,color='gray',ls=':',lw=.8,label='P no pad contact: state58')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7,loc='lower left');axes[2,0].set_yticks([0,1,2])
fig.suptitle('State33 under color shift: grasp establishment and premature opening')
fig.savefig(CASE/'grasp_failure_focus.png',dpi=130);plt.close(fig)

snapshots=[48,54,56,58,64]
sheet=Image.new('RGB',(672,480*len(snapshots)),'white');draw=ImageDraw.Draw(sheet)
for col,m in enumerate(rows):
    reader=imageio.get_reader(SOURCE/m/'color_strong/task_08_episode_033.mp4',format='FFMPEG')
    for row,k in enumerate(snapshots):
        frame=reader.get_data(k)
        observed=np.concatenate([frame[:224,224:448],frame[224:448,224:448]],axis=0)
        sheet.paste(Image.fromarray(observed),(224*col,480*row+32))
        r=rows[m][k]
        draw.text((224*col+3,480*row+2),f'{m} state{k}, original recording',fill='black')
        draw.text((224*col+3,480*row+17),f"pads={int(r['left_pad_contact']=='True')+int(r['right_pad_contact']=='True')} next_grip={actions[m][k,6]:+.0f}",fill='black')
    reader.close()
sheet.save(CASE/'grasp_keyframes.png')
sheet.save(CASE/'grasp_keyframes.jpg',quality=88)

resource=read(ROOT/'replay_33_resource.json');assert resource['exit_code']==0
evidence=dict(summary=summary,resource=resource,replay_checks=checks,
    comparison_at_close44=next(r for r in deltas if r['reference']=='C' and r['state_step']==44),
    comparison_before_open56=next(r for r in deltas if r['reference']=='C' and r['state_step']==56),
    P_gripper_remains_open_after56=bool(np.all(actions['P'][56:,6]<0)),
    reserved_initial_states=list(range(45,50)),scope='Saved actions only; original physics; camera rendering disabled; no policy/training')
(OUT/'evidence_summary.json').write_text(json.dumps(evidence,indent=2))
lines=['# P偏色状态33：抓取建立失败与提前张爪','',
 '结论：P没有完成有效搬运。抓取接触尚未稳定时，它在动作56提前且持续张爪，碗短暂抬起后落回桌面；机械臂随后空手离开，直到300步都未重新闭爪。不能把此轨迹归为已经夹稳后的搬运脱持或盘上放置失败。','',
 '|模型|首次闭爪动作|首次重新张爪动作|连续闭爪命令时长|首次双侧指垫接触状态|最大抬升mm|结果|','|---|---:|---:|---:|---:|---:|---|']
for m,s in summary.items():lines.append(f"|{m}|{s['first_close_action']}|{s['first_release_action']}|{s['first_continuous_closed_command_seconds']:.2f}s|{s['first_both_pad_contact_state']}|{s['max_lift_mm']:.2f}|{'成功' if s['success'] else '失败'}，{s['steps']}步|")
lines+=['',
 '1. 接近与闭爪：三者均从动作44开始闭爪。此时P相对C的末端位置差1.08mm、旋转差0.128°；到状态56位置差0.51mm、旋转差0.333°。没有看到大的坐标方向或旋转翻转，但毫米级接近差异仍可能改变接触，不能据此排除抓取几何敏感性。',
 '2. 张爪前已存在夹持差异：状态56（尚未执行动作56），base/C已经双侧指垫接触；P只有单侧接触。P实际开度6.67mm，base/C约4.93/4.95mm；P碗的世界x为-92.06mm，base/C为-83.67/-82.73mm，物体响应已经分化。不能将之前所有差异都归因于尚未执行的张爪动作。',
 '3. 关键相位切换：动作56，P夹爪命令从+1变为-1，base/C保持+1；同时P仍输出向上移动命令dz=+0.200，base/C约+0.315/+0.319（均为归一化控制输入）。此时P碗只抬高2.20mm，尚未达到用于分析的20mm抬升标记。',
 '4. 接触丢失：P状态57抬升达到全程最高4.29mm；到58降至0.28mm，而末端从958.76mm继续升到961.63mm。开度从6.94增至9.06mm，两个指垫都不再接触碗。碗随后停在桌面附近，未随末端进入搬运。',
 '5. 没有恢复抓取：P动作56–299的夹爪命令全部为-1。末端继续移动，碗留在桌面；全程没有碗盘接触，最终碗盘水平距离137.74mm，超过官方30mm阈值。盘的最大位移约0.00025mm，目标盘移动不是本次失败的解释。','',
 '可支持的近端原因是“夹持建立不充分 + 过早张爪 + 后续不再抓取”。小幅位姿变化是否触发接触分叉、提前张爪是否单独足以导致失败，尚未做改动作反事实实验，不能给出单一因果归因。',
 '本轮C和P唯一训练变量是采样分布；此样本表明阶段采样并未保证闭环抓取相位正确。正常条件阶段FM均值下降不能替代偏色闭环中的夹持/相位检查。没有该回合世界模型目标视频及阈值前夹爪预测记录，不能继续归因到世界模型目标错误、某个注意力层或夹爪阈值裕量。','',
 '复现可信度：三条已有动作在原始LIBERO物理资产下重放，稳定后完整状态哈希一致，首次成功状态分别81、80、无，与原始策略结果逐一吻合。现服务器为2GB CPU模式，关闭相机与渲染；未执行重渲染图像一致性检查。视频和关键帧来自之前偏色闭环录制，不是本次生成的渲染。',
 f"物理回放耗时{resource['seconds']:.2f}秒，峰值RSS约{resource['max_rss_kib']/1024**2:.2f}GiB；无模型推理、无训练、无45–49闭环或调参。",'',
 '索引与单位：状态k是执行k条动作后的状态，动作k将状态k推进到k+1，20Hz为仿真控制频率。位置是MuJoCo世界坐标，CSV单位米；物体位置为body origin，碗盘原点z差不是碗底到盘面的物理间隙。双侧指垫接触只是代理指标，本结论同时依据开度、抬升和相对运动。','',
 '## 产物','',
 '- `color_strong_episode33/grasp_failure_focus.png`：关键抓取窗口的动作、开度、抬升和接触。',
 '- `color_strong_episode33/action_commands.png`：三者完整7维命令及前120步对照。',
 '- `color_strong_episode33/world_positions*.png`：末端、碗、盘的世界位置。',
 '- `color_strong_episode33/relative_pose_first120.png`：碗在末端坐标系的位置和末端旋转变化。',
 '- `color_strong_episode33/original_policy_comparison.mp4`、`grasp_keyframes.jpg`：原策略视频对照；成功模型结束后保留末帧并标记held。',
 '- 各模型`states.csv`、`derived.csv`、`replay.npz`，以及`key_states_and_actions.csv`、`P_pose_differences.csv`：可核对原始位姿、控制目标和动作。',
 '- `replay_checks.json`、`evidence_summary.json`、`SHA256SUMS.json`：回放校验与证据清单。','',
 '## 服务器复现命令','',
 '```bash','source /root/autodl-tmp/wam_glance_project/deploy_env.sh','cd "$WAM_MODEL_ROOT"',
 'export PYTHONPATH="$WAM_MODEL_ROOT:/root/autodl-tmp/wam_glance_project/LIBERO"',
 'export LIBERO_CONFIG_PATH=/root/autodl-tmp/wam_glance_project/LIBERO/libero',
 'export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl',
 '# 必须使用新的输出目录；--physics-only 不创建渲染上下文。',
 '/root/wam_sim_env/bin/python day1/replay_failure_cases.py --physics-only \\',
 '  --source results/stage_v1/development --out results/failure_replay_33_new \\',
 '  --case color_strong:33 --models base C P',
 '/root/wam_sim_env/bin/python day1/analyze_failure_replays.py \\',
 '  --source results/stage_v1/development --out results/failure_replay_33_new','```']
(OUT/'失败原因分析.md').write_text('\n'.join(lines),encoding='utf-8')
for source in [ROOT/'run_replay_33_cpu.py',MODEL/'day1/report_failure_33.py']:
    shutil.copy2(source,OUT/'sources'/source.name)
shutil.copy2(ROOT/'replay_33_resource.json',OUT/'resource.json')
shutil.copy2(ROOT/'replay_33_physics.log',OUT/'replay.log')
review_omit={'.npz','world_positions.png','placement_diagnostics.png','placement_diagnostics_first120.png',
    'grasp_keyframes.png','terminal_original_decoded.png'}
(OUT/'review_scope.json').write_text(json.dumps(dict(review_omits=sorted(review_omit),
    retained='All numeric CSVs, original-action comparison video, focus/action/first120 world/relative plots, JPEG keyframes, sources and checks',
    full_archive=str(ROOT/'wam_failure_replay_33_full.zip')),indent=2))
manifest={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
for kind in ['full','review']:
    with zipfile.ZipFile(ROOT/f'wam_failure_replay_33_{kind}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if not p.is_file() or (kind=='review' and (p.suffix in review_omit or p.name in review_omit)):continue
            z.write(p,p.relative_to(OUT.parent))
print(json.dumps(evidence,indent=2),flush=True)
