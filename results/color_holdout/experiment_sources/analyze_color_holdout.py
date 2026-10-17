"""Audit paired held-out evaluation and export reproducible results; no policy changes."""
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
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ROOT=Path('/root/autodl-tmp/wam_glance_project')
MODEL=ROOT/'GlanceWAM'
OUT=MODEL/'results/color_holdout'
sys.path.insert(0,str(MODEL/'day1'))
from eval_protocol import protocol_hash, validate_episode_results, paired_delta

protocol=json.loads((OUT/'protocol.json').read_text())
assert protocol['episode_ids']==list(range(25,45))
assert list(protocol['conditions'])==['clean','color_strong']
assert (OUT/'summary.json').is_file(), 'Evaluation not complete'
report=OUT/'comparison';report.mkdir(exist_ok=True)
data={};pairs={};rows=[];resources={}
for model in ['base','clean','aug']:
    data[model]={}
    for condition in protocol['conditions']:
        r=json.loads((OUT/model/condition/'results.json').read_text())
        assert r['protocol_sha256']==protocol_hash(protocol)
        validate_episode_results(r['episode_results'],protocol)
        assert r['episodes']==20
        assert r['successes']==sum(x['success'] for x in r['episode_results'])
        data[model][condition]=r
        for e in r['episode_results']:
            key=(e['task_id'],e['episode_id'])
            pair=tuple(e[k] for k in ['env_seed','init_state_sha256','post_settle_state_sha256'])
            if key in pairs:assert pairs[key]==pair
            pairs[key]=pair
            assert e['task_name']=='put_the_bowl_on_the_plate'
            a=np.load(OUT/model/condition/f"task_08_episode_{e['episode_id']:03d}_actions.npy")
            assert a.shape==(e['steps'],7) and np.isfinite(a).all()
            assert e['video_frames']==e['steps']+1
            rows.append(dict(model=model,condition=condition,episode_id=e['episode_id'],
                success=e['success'],steps=e['steps'],seconds=e['seconds'],
                gripper_switch_steps=(np.flatnonzero(np.diff(a[:,6])!=0)+1).tolist()))
    metrics=[json.loads(x) for x in (OUT/model/'server_metrics.jsonl').read_text().splitlines()]
    resources[model]=dict(queries=len(metrics),peak_allocated_gib=max(x['peak_allocated_gib'] for x in metrics),
        peak_reserved_gib=max(x['peak_reserved_gib'] for x in metrics),
        median_request_seconds=float(np.median([x['seconds'] for x in metrics])),
        goal_refresh_median_seconds=float(np.median([x['seconds'] for x in metrics if x['goal_refresh']])))
assert len(rows)==120 and len(pairs)==20
summary={m:{c:dict(successes=data[m][c]['successes'],episodes=20) for c in protocol['conditions']} for m in data}
paired={};paired_cases={}
for c in protocol['conditions']:
    paired[c]={};paired_cases[c]={}
    for candidate in ['clean','aug']:
        paired[c][candidate+'_vs_base']=paired_delta(data['base'][c]['episode_results'],data[candidate][c]['episode_results'])
    paired[c]['aug_vs_clean']=paired_delta(data['clean'][c]['episode_results'],data['aug'][c]['episode_results'])
    for source,target in [('base','clean'),('base','aug'),('clean','aug')]:
        b={e['episode_id']:e['success'] for e in data[source][c]['episode_results']}
        t={e['episode_id']:e['success'] for e in data[target][c]['episode_results']}
        paired_cases[c][target+'_vs_'+source]=dict(recovered=[e for e in b if not b[e] and t[e]],regressed=[e for e in b if b[e] and not t[e]])
overview=dict(protocol_sha256=protocol_hash(protocol),unique_initial_states=20,episodes=120,
    results=summary,paired=paired,paired_cases=paired_cases,resources=resources,
    scope='One pretrained LIBERO goal task; visual OOD test. Synchronous inference, not hardware real-time.',
    example_selection='Initial state 25 was selected before held-out results; failures run to 300-step cap.')
(report/'overview.json').write_text(json.dumps(overview,indent=2))
with (report/'episode_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(8,4));x=np.arange(2)
for i,m in enumerate(data):
    values=[summary[m][c]['successes'] for c in protocol['conditions']]
    bars=ax.bar(x+(i-1)*.25,values,width=.23,label=m)
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.2,f'{v}/20',ha='center')
ax.set(xticks=x,xticklabels=list(protocol['conditions']),ylim=(0,23),ylabel='Successful episodes / 20',title='Held-out initial states 25–44')
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.30),ncol=3)
fig.subplots_adjust(bottom=.27);fig.savefig(report/'success_rates.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
video_checks={}
for ax,c in zip(axes,protocol['conditions']):
    paths=[];readers=[];lengths=[]
    for m in data:
        a=np.load(OUT/m/c/'task_08_episode_025_actions.npy')
        ax.step(np.arange(len(a))/20,a[:,6],where='post',label=m)
        p=OUT/m/c/'task_08_episode_025.mp4';paths.append(p)
        reader=imageio.get_reader(p,format='FFMPEG');readers.append(reader);lengths.append(reader.count_frames())
    ax.set(ylabel='Gripper command',title=c,yticks=[-1,1]);ax.legend()
    destination=report/f'{c}_episode25_comparison.mp4'
    last=[None]*3
    with imageio.get_writer(destination,format='FFMPEG',fps=20,codec='libx264',ffmpeg_params=['-threads','1']) as writer:
        for k in range(max(lengths)):
            canvas=Image.new('RGB',(672,480),'white');draw=ImageDraw.Draw(canvas)
            for i,(m,reader,n) in enumerate(zip(data,readers,lengths)):
                if k<n:last[i]=reader.get_data(k)
                frame=last[i]
                # Raw video: primary/wrist rows, original/observed columns.
                observed=np.concatenate([frame[:224,224:448],frame[224:448,224:448]],axis=0)
                canvas.paste(Image.fromarray(observed),(i*224,32))
                e=next(x for x in data[m][c]['episode_results'] if x['episode_id']==25)
                draw.text((i*224+4,3),f"{m} {'success' if e['success'] else 'timeout'}",fill='black')
                draw.text((i*224+4,17),f"step {min(k,n-1)}"+(' held' if k>=n else ''),fill='black')
            writer.append_data(np.asarray(canvas))
    for reader in readers:reader.close()
    reader=imageio.get_reader(destination,format='FFMPEG');count=reader.count_frames();reader.close()
    assert count==max(lengths)
    video_checks[c]=dict(frames=count,source_frames=lengths,fps=20)
axes[-1].set_xlabel('Simulation seconds; -1 open / +1 close')
fig.tight_layout();fig.savefig(report/'gripper_episode25.png',dpi=160);plt.close(fig)
(report/'video_checks.json').write_text(json.dumps(video_checks,indent=2))
diagnostics=[]
for c in protocol['conditions']:
    cases=paired_cases[c]['aug_vs_base']
    # Diagnostic examples are selected after results and labelled as such.
    for outcome in ['recovered','regressed']:
        if not cases[outcome]:continue
        episode=min(cases[outcome]);sheet=Image.new('RGB',(672,1440),'white');draw=ImageDraw.Draw(sheet)
        for column,m in enumerate(data):
            e=next(x for x in data[m][c]['episode_results'] if x['episode_id']==episode)
            reader=imageio.get_reader(OUT/m/c/f'task_08_episode_{episode:03d}.mp4',format='FFMPEG')
            for row,step in enumerate([32,80,300]):
                index=min(step,e['steps']);frame=reader.get_data(index)
                observed=np.concatenate([frame[:224,224:448],frame[224:448,224:448]],axis=0)
                sheet.paste(Image.fromarray(observed),(column*224,row*480+32))
                draw.text((column*224+3,row*480+2),f'{m} step {index}'+(' held' if step>e['steps'] else ''),fill='black')
                draw.text((column*224+3,row*480+17),f"success={e['success']}",fill='black')
            reader.close()
        filename=f'diagnostic_{c}_{outcome}_episode{episode}.png'
        sheet.save(report/filename)
        diagnostics.append(dict(condition=c,outcome=outcome,episode_id=episode,image=filename,
            selection='Lowest initial-state ID with this aug-vs-base outcome; post-hoc diagnostic, not representative sample.'))
(report/'diagnostic_cases.json').write_text(json.dumps(diagnostics,indent=2))
lines=['# 偏色条件：独立初始状态闭环对照','',
 '筛选阶段仅用基线、初始状态20–24：正常5/5、亮度0.5为5/5、亮度0.35为5/5、RGB[0.7,1,1.3]为2/5。',
 '据此固定正常与偏色两条件；正式阶段使用25–44，三个模型、同一随机种子42，共120次。使用原始、clean200末步、aug200末步检查点；没有新增训练。','',
 '| 模型 | 正常 | 偏色 |','|---|---:|---:|']
for m in data:lines.append(f"| {m} | {summary[m]['clean']['successes']}/20 | {summary[m]['color_strong']['successes']}/20 |")
lines+=['','| 条件及比较 | 恢复成功 | 退化失败 | 成功率差（百分点） |','|---|---:|---:|---:|']
for c,v in paired.items():
    for name,p in v.items():lines.append(f"| {c}: {name} | {p['recovered']} | {p['regressed']} | {p['success_delta_pp']:.1f} |")
delta=paired['color_strong']['aug_vs_clean'];clean_delta=paired['clean']['aug_vs_clean']
base_delta=paired['color_strong']['aug_vs_base']
lines+=['',f"aug 相对 clean 的偏色净差为 {delta['success_delta_pp']:.1f} 个百分点；正常画面净差为 {clean_delta['success_delta_pp']:.1f} 个百分点。应同时报告两个结果。",'',
 f"aug 相对原始 base 的偏色净差为 {base_delta['success_delta_pp']:.1f} 个百分点。"+('本轮观察到成功数增加，但样本量小且只有一个任务。' if base_delta['success_delta_pp']>0 else '本轮未观察到超过原始模型的偏色成功率，不能声称后训练改善了原始模型。'),'',
 '所有配对的环境种子、原始初始状态和稳定后完整状态哈希一致；动作有限且长度与轨迹一致。夹爪切换记录见CSV。',
 '恢复与退化的具体初始状态编号见 overview.json 的 paired_cases；服务器各模型目录保留全部原始视频与动作。',
 '视频固定展示初始状态25，成功后停留末帧并标记held；不是挑选成功样例。',
 '只含一个预训练任务及20个物理初始状态，各条件复用状态。偏色强于训练增益0.9–1.1，属于更强扰动泛化测试。不能据此宣称跨任务、真机泛化或20Hz实时控制。',
 '成功为官方LIBERO判据；没有额外验证释放后长时间稳定性。恢复/退化使用相同状态配对计数。',
 '导出的检查点不含优化器状态；本阶段没有改变任何模型参数。']
(report/'实验记录.md').write_text('\n'.join(lines),encoding='utf-8')
sources=OUT/'experiment_sources';sources.mkdir(exist_ok=True)
for name in ['run_color_holdout.sh','analyze_color_holdout.py']:(sources/name).write_bytes((ROOT/name).read_bytes())
for name in ['eval_protocol.py','run_robust_eval.py','eval_robust_libero.py','paired_policy_server.py','finetune_utils.py']:
    (sources/name).write_bytes((MODEL/'day1'/name).read_bytes())
all_files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json']
(OUT/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in all_files},indent=2))
for kind in ['full','review']:
    with zipfile.ZipFile(ROOT/f'wam_color_holdout_{kind}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(OUT)
            if kind=='review' and p.suffix in ['.mp4','.npy'] and rel.parts[0]!='comparison':continue
            z.write(p,Path('color_holdout')/rel)
print(json.dumps(overview,indent=2))
