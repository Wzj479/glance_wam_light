"""Audit completed C/P experiment, export figures and a weight-free review bundle."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT=Path('/root/autodl-tmp/wam_glance_project');MODEL=ROOT/'GlanceWAM';OUT=MODEL/'results/stage_v1'
deadline=time.monotonic()+14400
while not (OUT/'EVALUATION_COMPLETE').exists():
    for name in ['stage_v1_training.log','stage_v1_evaluation.log']:
        if (ROOT/name).exists() and 'Traceback' in (ROOT/name).read_text():raise RuntimeError(name+' failed')
    if time.monotonic()>deadline:raise TimeoutError('Evaluation deadline')
    time.sleep(10)
sys.path.insert(0,str(MODEL/'day1'))
from eval_protocol import paired_delta,protocol_hash,validate_episode_results
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

report=OUT/'comparison';report.mkdir(exist_ok=True)
times=read(OUT/'run_wall_times.json');selection=read(OUT/'evaluation_selection.json')
training={};traces={};args={};checkpoints={}
for name in ['C','P']:
    run=OUT/(name+'_seed42');rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='train'];val=[r for r in rows if r['event']=='validation']
    selections=[r for r in rows if r['event']=='selection'];final=read(run/'summary.json');args[name]=read(run/'train_args.json')
    assert final['completed_steps']==100 and final['frozen_unchanged']
    assert [r['step'] for r in updates]==list(range(1,101))
    assert [r['step'] for r in val]==[0,25,50,75,100]
    assert all(math.isfinite(r['action_fm_loss']) and math.isfinite(r['grad_norm']) and r['grad_norm']>0 for r in updates)
    expected={'uniform':8} if name=='C' else {'uniform':4,'grasp':2,'place':2}
    assert all(r['sampling_source_counts']==expected for r in updates)
    assert any(r['event']=='official_feature_parity' and r['passed'] for r in rows)
    assert any(r['event']=='integrity' and r['action_updated'] and r['frozen_unchanged'] for r in rows)
    assert final['trainable_parameters']==29392903
    training[name]=dict(steps=100,best_step=final['best_step'],best_score=final['best_selection_score'],
        trainable_parameters=final['trainable_parameters'],frozen_sha256=final['frozen_sha256'],
        split_sha256=final['split_sha256'],phase_manifest_sha256=sha(run/'phase_manifest.json'),
        process_seconds=times[name]['seconds'],update_seconds=sum(r['seconds'] for r in updates),
        mean_update_seconds=float(np.mean([r['seconds'] for r in updates])),
        peak_allocated_gib=max(r.get('peak_allocated_gib',0) for r in rows),
        peak_reserved_gib=max(r.get('peak_reserved_gib',0) for r in rows),
        sampling_counts={k:v*100 for k,v in expected.items()},initial_validation=val[0],final_validation=val[-1],selection=selections)
    traces[name]=(updates,val,selections)
    for tag in ['best','last']:
        p=run/'checkpoints'/f'{tag}.action.pt';metadata=read(p.with_suffix('.json'))
        digest=sha(p);assert metadata['checkpoint_sha256']==digest
        checkpoints[name+'/'+tag]=dict(server_path=str(p),bytes=p.stat().st_size,sha256=digest,metadata=metadata)
assert training['C']['frozen_sha256']==training['P']['frozen_sha256']
for file in ['split_manifest.json','phase_manifest.json']:
    assert (OUT/'C_seed42'/file).read_bytes()==(OUT/'P_seed42'/file).read_bytes()
common=lambda a:{k:v for k,v in a.items() if k not in ['out','sampling_mode']}
assert common(args['C'])==common(args['P']) and args['C']['seed']==42
assert args['C']['aug_prob']==args['P']['aug_prob']==0

loop=OUT/'development';protocol=read(loop/'protocol.json');fingerprint=protocol_hash(protocol)
assert protocol['episode_ids']==list(range(25,45)) and protocol['seed']==42
data={};pairs={};episode_rows=[]
for name in ['base','C','P']:
    data[name]={}
    for condition in protocol['conditions']:
        r=read(loop/name/condition/'results.json');assert r['protocol_sha256']==fingerprint
        validate_episode_results(r['episode_results'],protocol)
        assert r['episodes']==20 and r['successes']==sum(e['success'] for e in r['episode_results'])
        data[name][condition]=r
        for e in r['episode_results']:
            key=(e['task_id'],e['episode_id']);pair=tuple(e[k] for k in ['env_seed','init_state_sha256','post_settle_state_sha256'])
            if key in pairs:assert pairs[key]==pair
            pairs[key]=pair
            a=np.load(loop/name/condition/f"task_08_episode_{e['episode_id']:03d}_actions.npy")
            assert a.shape==(e['steps'],7) and np.isfinite(a).all() and e['video_frames']==e['steps']+1
            episode_rows.append(dict(model=name,condition=condition,episode_id=e['episode_id'],success=e['success'],steps=e['steps'],seconds=e['seconds']))
assert len(episode_rows)==120 and len(pairs)==20
counts={m:{c:dict(successes=r['successes'],episodes=20) for c,r in d.items()} for m,d in data.items()}
primary={m:counts['base' if selection['models'][m]['primary_source']=='base_reused' else m] for m in ['C','P']}
paired={};cases={}
for c in protocol['conditions']:
    paired[c]={};cases[c]={}
    for a,b in [('base','C'),('base','P'),('C','P')]:
        label=b+'_vs_'+a
        paired[c][label]=paired_delta(data[a][c]['episode_results'],data[b][c]['episode_results'])
        old={e['episode_id']:e['success'] for e in data[a][c]['episode_results']}
        new={e['episode_id']:e['success'] for e in data[b][c]['episode_results']}
        cases[c][label]=dict(recovered=[i for i in old if not old[i] and new[i]],regressed=[i for i in old if old[i] and not new[i]])
overview=dict(training=training,checkpoint_records=checkpoints,evaluation_selection=selection,
    rollout_results=counts,selected_best_results=primary,paired_rollout_comparisons=paired,paired_cases=cases,
    protocol_sha256=fingerprint,physical_state_pairing_verified=True,new_rollouts=80,reused_baseline_rollouts=40,
    reserved_initial_states=list(range(45,50)),evaluation_wall_time=read(OUT/'evaluation_wall_time.json'))
(report/'overview.json').write_text(json.dumps(overview,indent=2))
validation_rows=[]
for name,(_,val,sels) in traces.items():
    lookup={s['step']:s for s in sels}
    for v in val:
        validation_rows.append(dict(model=name,**v,
            selection_score=lookup.get(v['step'],{}).get('selection_score',1.),
            selection_eligible=lookup.get(v['step'],{}).get('eligible',True)))
with (report/'phase_validation.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(validation_rows[0]));w.writeheader();w.writerows(validation_rows)
with (report/'episode_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(episode_rows[0]));w.writeheader();w.writerows(episode_rows)
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
keys=['clean_fm_loss','clean_grasp_fm_loss','clean_place_fm_loss','corrupt_fm_loss','corrupt_grasp_fm_loss','corrupt_place_fm_loss']
for ax,key in zip(axes.flat,keys):
    for name,(_,val,_) in traces.items():ax.plot([v['step'] for v in val],[v[key] for v in val],'-o',label=name)
    ax.set(title=key,xlabel='Optimizer step',ylabel='Flow matching MSE');ax.grid(alpha=.2);ax.legend()
fig.savefig(report/'phase_validation.png',dpi=140);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for name,(updates,_,sels) in traces.items():
    axes[0].plot([u['step'] for u in updates],[u['action_fm_loss'] for u in updates],alpha=.6,label=name)
    axes[1].plot([s['step'] for s in sels],[s['selection_score'] for s in sels],'-o',label=name)
    for s in sels:
        if not s['eligible']:axes[1].plot(s['step'],s['selection_score'],'rx')
    axes[2].plot([u['step'] for u in updates],[u['peak_allocated_gib'] for u in updates],label=name+' allocated')
    axes[2].plot([u['step'] for u in updates],[u['peak_reserved_gib'] for u in updates],'--',label=name+' reserved')
axes[0].set(title='Train loss; sampling distributions differ',ylabel='FM MSE')
axes[1].axhline(1,color='gray',ls=':');axes[1].set(title='Selection score; x = normal gate rejected')
axes[2].set(title='Cumulative PyTorch memory peak',ylabel='GiB')
for ax in axes:ax.set_xlabel('Optimizer step');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(report/'training_resources.png',dpi=140);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(2)
for j,name in enumerate(data):
    label=name if name=='base' else name+' '+selection['models'][name]['rollout_role']
    bars=ax.bar(x+(j-1)*.25,[counts[name][c]['successes'] for c in protocol['conditions']],width=.23,label=label)
    ax.bar_label(bars,labels=[f"{counts[name][c]['successes']}/20" for c in protocol['conditions']],padding=3)
ax.set(xticks=x,xticklabels=list(protocol['conditions']),ylim=(0,23),ylabel='Successes / 20',title='Development states 25–44; baseline reused')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=8);fig.subplots_adjust(bottom=.25)
fig.savefig(report/'closed_loop_success.png',dpi=150);plt.close(fig)

video_checks={}
for condition,episode in [('clean',25),('color_strong',25),('clean',39),('color_strong',32)]:
    readers=[];lengths=[];entries=[]
    for name in data:
        reader=imageio.get_reader(loop/name/condition/f'task_08_episode_{episode:03d}.mp4',format='FFMPEG')
        readers.append(reader);lengths.append(reader.count_frames())
        entries.append(next(e for e in data[name][condition]['episode_results'] if e['episode_id']==episode))
    output=report/f'{condition}_episode{episode}_comparison.mp4';last=[None]*3
    with imageio.get_writer(output,format='FFMPEG',fps=20,codec='libx264',ffmpeg_params=['-threads','1']) as writer:
        for k in range(max(lengths)):
            canvas=Image.new('RGB',(672,496),'white');draw=ImageDraw.Draw(canvas)
            for j,(name,reader,n,e) in enumerate(zip(data,readers,lengths,entries)):
                if k<n:last[j]=reader.get_data(k)
                frame=last[j];observed=np.concatenate([frame[:224,224:448],frame[224:448,224:448]],axis=0)
                canvas.paste(Image.fromarray(observed),(j*224,48))
                label='base cached' if name=='base' else name+' '+('best' if selection['models'][name]['rollout_role']=='selected_best' else 'last diagnostic')
                draw.text((j*224+3,2),label,fill='black')
                draw.text((j*224+3,17),f"success={e['success']} step {min(k,n-1)}"+(' held' if k>=n else ''),fill='black')
                draw.text((j*224+3,32),f'{condition} init={episode}',fill='black')
            writer.append_data(np.asarray(canvas))
    for r in readers:r.close()
    reader=imageio.get_reader(output,format='FFMPEG');n=reader.count_frames();reader.close();assert n==max(lengths)
    video_checks[output.name]=dict(frames=n,source_frames=lengths,fps=20,selection='Fixed state25; prior failure diagnosis states39/32')
(report/'video_checks.json').write_text(json.dumps(video_checks,indent=2))
lines=['# C/P 各100步：阶段采样对照','',
 'C为均匀窗口采样；P每8个样本包含4个均匀、2个夹持、2个放置窗口。两组均不加视觉增强，唯一训练变量为采样分布。',
 '单张A100 40GB依次执行；相同种子42、8条训练/2条验证示范、lr=3e-6、梯度累积8。只更新动作交叉注意力与解码器29,392,903个参数，世界模型和其余动作参数冻结。',
 '数据为LIBERO仿真示范。训练轨迹ID与闭环初始状态ID属于不同命名空间。闭环25–44已作为开发集，45–49本轮未执行闭环或用于调参。','',
 '|组别|best步|训练进程秒数|更新耗时秒数|峰值allocated/reserved GiB|','|---|---:|---:|---:|---:|']
for m,t in training.items():lines.append(f"|{m}|{t['best_step']}|{t['process_seconds']:.1f}|{t['update_seconds']:.1f}|{t['peak_allocated_gib']:.2f}/{t['peak_reserved_gib']:.2f}|")
lines+=['','阶段验证固定26个窗口，夹持/放置各8个（允许标签重叠）；正常与偏色使用相同噪声和采样种子。损失为官方动作流匹配联合MSE，未额外添加夹爪分类或位置损失。',
 'best按正常夹持/放置相对初始损失的均值选择，并要求全局正常损失不超过初始的1.05倍；若无合格改善则best停留step0。last始终保存100步。',
 '固定窗口验证使用示范未来帧作为hindsight目标，闭环使用世界模型预测的目标。FM损失是速度场预测误差，不能当作实际末端位置误差或夹持稳定性指标。','',
 '|闭环实际执行模型|角色|正常|偏色RGB[0.7,1,1.3]|','|---|---|---:|---:|']
for m in data:
    role='缓存基线' if m=='base' else selection['models'][m]['rollout_role']
    lines.append(f"|{m}|{role}|{counts[m]['clean']['successes']}/20|{counts[m]['color_strong']['successes']}/20|")
lines+=['','|验证集选择的模型|best步|正常|偏色|','|---|---:|---:|---:|']
for m in ['C','P']:lines.append(f"|{m}|{training[m]['best_step']}|{primary[m]['clean']['successes']}/20|{primary[m]['color_strong']['successes']}/20|")
lines+=['',
 'best为0时，已严格验证其动作权重在部署BF16精度下与原始模型一致，复用基线为主要结果；额外运行last仅用于诊断，不能替代best结果。',
 '本轮新增80次模型闭环，复用40次同协议原始模型结果；所有配对环境种子、原始初始状态与稳定后完整状态哈希相同。原始动作、视频和全部分阶段损失留存服务器。',
 '恢复/退化初始状态见overview.json的paired_cases。固定展示25，以及此前已分析的39正常/32偏色；未根据本轮结果选择视频或继续调参。',
 '这些是一个预训练抓取放置任务的开发集结果，不能声称跨任务或真机泛化。LIBERO成功判据未额外要求松爪后长时间稳定；同步推理不代表20Hz实时部署。',
 'best/last动作检查点保留服务器，不包含世界模型或优化器状态；SHA、大小、基座指纹与保存步数见overview.json。冻结哈希一致、采样配额、有限梯度、动作参数实际更新和官方特征一致性均已核验。']
(report/'实验记录.md').write_text('\n'.join(lines),encoding='utf-8')
sources=OUT/'experiment_sources';sources.mkdir(exist_ok=True)
for name in ['run_stage_v1_training.py','run_stage_v1_evaluation.py','analyze_stage_v1.py','status_stage_v1.py']:
    shutil.copy2(ROOT/name,sources/name)
for p in (MODEL/'results/stage_v1_audit/sources').iterdir():
    if p.suffix in ['.py','.sh']:shutil.copy2(p,sources/p.name)
for name in ['stage_v1_training.log','stage_v1_evaluation.log']:shutil.copy2(ROOT/name,OUT/name)
files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.pt' and p.name!='SHA256SUMS.json']
(OUT/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in files},indent=2))
for kind in ['full','review']:
    with zipfile.ZipFile(ROOT/f'wam_stage_v1_{kind}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if not p.is_file() or p.suffix=='.pt':continue
            rel=p.relative_to(OUT)
            if kind=='review' and p.suffix in ['.mp4','.npy'] and rel.parts[0]!='comparison':continue
            z.write(p,Path('stage_v1')/rel)
(OUT/'ANALYSIS_COMPLETE').touch()
print(json.dumps(overview,indent=2),flush=True)
