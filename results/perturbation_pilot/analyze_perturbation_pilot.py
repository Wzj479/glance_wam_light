"""Summarize completed baseline rollouts without inferring unobserved failure causes."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/root/autodl-tmp/wam_glance_project')
MODEL = ROOT / 'GlanceWAM'
OUT = MODEL / 'results/perturbation_pilot'
sys.path.insert(0, str(MODEL / 'day1'))
from eval_protocol import protocol_hash, validate_episode_results

protocol = json.loads((OUT / 'protocol.json').read_text())
assert (OUT / 'summary.json').is_file(), 'Evaluation must finish first'
rows, summaries, pairs = [], {}, {}
for condition in protocol['conditions']:
    result = json.loads((OUT / 'base' / condition / 'results.json').read_text())
    assert result['protocol_sha256'] == protocol_hash(protocol)
    validate_episode_results(result['episode_results'], protocol)
    for record in result['episode_results']:
        episode = record['episode_id']
        pair = tuple(record[k] for k in ('env_seed', 'init_state_sha256', 'post_settle_state_sha256'))
        if episode in pairs:
            assert pairs[episode] == pair
        pairs[episode] = pair
        actions = np.load(OUT / 'base' / condition / f'task_08_episode_{episode:03d}_actions.npy')
        assert actions.shape == (record['steps'], 7) and np.isfinite(actions).all()
        assert record['video_frames'] == record['steps'] + 1
        changes = np.flatnonzero(np.diff(actions[:, 6]) != 0) + 1
        rows.append(dict(condition=condition, episode_id=episode, success=record['success'],
                         steps=record['steps'], gripper_switch_steps=changes.tolist(),
                         wall_seconds=record['seconds']))
    summaries[condition] = dict(successes=result['successes'], episodes=result['episodes'],
        median_success_steps=float(np.median([r['steps'] for r in result['episode_results'] if r['success']]))
        if result['successes'] else None)
clean = {r['episode_id']:r['success'] for r in rows if r['condition']=='clean'}
selected = []
for condition in protocol['conditions']:
    if condition == 'clean': continue
    current = {r['episode_id']:r['success'] for r in rows if r['condition']==condition}
    losses = sum(clean[e] and not current[e] for e in clean)
    gains = sum(not clean[e] and current[e] for e in clean)
    summaries[condition].update(paired_losses_vs_clean=losses, paired_gains_vs_clean=gains)
    if losses > gains: selected.append(condition)
decision = dict(protocol_sha256=protocol_hash(protocol), summaries=summaries,
    selected_conditions=selected, selection_rule='paired baseline losses exceed gains vs clean; pilot is exploratory',
    next_step='held-out states 25..44, base/clean200/aug200, clean plus selected conditions'
    if selected else 'Stop expansion: this pilot does not justify the planned three-model robustness comparison.')
(OUT/'pilot_analysis.json').write_text(json.dumps(decision,indent=2))
with (OUT/'episode_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
fig, axes=plt.subplots(1,2,figsize=(11,4))
names=list(summaries)
axes[0].bar(names,[summaries[c]['successes'] for c in names])
axes[0].set(ylim=(0,6),ylabel='Successful episodes / 5',title='Baseline perturbation pilot')
for i,c in enumerate(names): axes[0].text(i,summaries[c]['successes']+.1,f"{summaries[c]['successes']}/5",ha='center')
for c in names:
    r=[x for x in rows if x['condition']==c]
    axes[1].plot([x['episode_id'] for x in r],[x['steps'] for x in r],marker='o',label=c)
axes[1].set(xlabel='Initial-state ID',ylabel='Executed control steps',title='300-step cap; failures retained')
axes[1].legend();fig.tight_layout();fig.savefig(OUT/'pilot_metrics.png',dpi=160);plt.close(fig)
fig, axes=plt.subplots(4,1,figsize=(10,8),sharex=True)
for ax,condition in zip(axes,names):
    action=np.load(OUT/'base'/condition/'task_08_episode_020_actions.npy')
    ax.step(np.arange(len(action))/20,action[:,6],where='post')
    ax.set(ylabel=condition,ylim=(-1.2,1.2),yticks=[-1,1])
axes[0].set_title('Initial state 20: executed gripper command (-1 open, +1 close)')
axes[-1].set_xlabel('Simulation seconds (not wall-clock inference time)')
fig.tight_layout();fig.savefig(OUT/'gripper_episode20.png',dpi=160);plt.close(fig)
lines=['# 基线视觉扰动试验','', '固定 task8：put_the_bowl_on_the_plate；初始状态20–24，seed42。仅评测原始基线，不训练。', '',
       '| 条件 | 成功数 | 成功轨迹步数中位数 |','|---|---:|---:|']
for c,v in summaries.items():lines.append(f"| {c} | {v['successes']}/5 | {v['median_success_steps']} |")
lines += ['', '各条件的环境种子、初始状态及稳定后状态哈希均配对一致。视频保留两相机的原始/扰动画面，末帧包含最后动作执行结果。', '',
          '夹爪切换时刻与实际执行步数见 episode_metrics.csv；失败原因必须检查视频后填写，不能从超时直接推断为抓取偏移。', '',
          '筛选规则：相对 clean，配对失败增加数大于成功恢复数。仅用于选择下一阶段条件，不作为提升结论。', '',
          '下一步：'+('在初始状态25–44上比较 base / clean200 / aug200，条件为 clean 和 '+', '.join(selected) if selected else '停止扩展本轮视觉扰动对照；当前试验未找到有效退化条件。'), '',
          '限制：只有5个物理初始状态；同步等待模型推理，不代表20Hz实时控制。成功采用官方LIBERO判据。']
(OUT/'试验记录.md').write_text('\n'.join(lines),encoding='utf-8')
for name in ['run_perturbation_pilot.sh','analyze_perturbation_pilot.py']:
    (OUT/name).write_bytes((ROOT/name).read_bytes())
files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json']
(OUT/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
with zipfile.ZipFile(ROOT/'wam_perturbation_pilot.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT.parent))
print(json.dumps(decision,indent=2))
