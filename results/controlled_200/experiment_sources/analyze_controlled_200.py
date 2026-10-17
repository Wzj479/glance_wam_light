"""Package completed paired training and 90 closed-loop episodes, without weights."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT=Path('/root/autodl-tmp/wam_glance_project')
MODEL=ROOT/'GlanceWAM'
OUT=MODEL/'results/controlled_200'
for _ in range(1080):
    if (OUT/'ALL_COMPLETE').exists():break
    time.sleep(10)
assert (OUT/'ALL_COMPLETE').exists()
sys.path.insert(0,str(MODEL/'day1'))
from eval_protocol import paired_delta, protocol_hash, validate_episode_results
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

report_dir=OUT/'comparison'
report_dir.mkdir(exist_ok=True)
traces={}
summary={}
args={}
checkpoint_records={}
times=read(OUT/'run_wall_times.json')['completed']
for name in ['clean','aug']:
    run=OUT/(name+'_seed42')
    rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    training=[r for r in rows if r['event']=='train']
    validation=[r for r in rows if r['event']=='validation']
    final=read(run/'summary.json')
    assert final['completed_steps']==200 and final['frozen_unchanged']
    assert [r['step'] for r in training]==list(range(1,201))
    assert all(math.isfinite(r['action_fm_loss']) and math.isfinite(r['grad_norm']) and r['grad_norm']>0 for r in training)
    assert any(r['event']=='official_feature_parity' and r['passed'] for r in rows)
    assert any(r['event']=='integrity' and r['frozen_unchanged'] and r['action_updated'] for r in rows)
    traces[name]=(training,validation)
    args[name]=read(run/'train_args.json')
    summary[name]=dict(completed_steps=200,trainable_parameters=final['trainable_parameters'],
        frozen_sha256=final['frozen_sha256'],split_sha256=final['split_sha256'],
        final_validation=validation[-1],initial_validation=validation[0],
        step_seconds_mean=float(np.mean([r['seconds'] for r in training])),
        step_seconds_median=float(np.median([r['seconds'] for r in training])),
        summed_train_step_seconds=sum(r['seconds'] for r in training),
        process_start_to_summary_seconds=times[name]['process_start_to_summary_seconds'],
        peak_allocated_gib=max(r.get('peak_allocated_gib',0) for r in rows),
        peak_reserved_gib=max(r.get('peak_reserved_gib',0) for r in rows))
    for p in (run/'checkpoints').glob('*.pt'):
        checkpoint_records[name+'/'+p.name]=dict(server_path=str(p),bytes=p.stat().st_size,sha256=sha(p),
            export_metadata=read(p.with_suffix('.json')))
assert summary['clean']['frozen_sha256']==summary['aug']['frozen_sha256']
assert summary['clean']['split_sha256']==summary['aug']['split_sha256']
common=lambda d:{k:v for k,v in d.items() if k not in {'out','aug_prob'}}
assert common(args['clean'])==common(args['aug'])
assert args['clean']['seed']==args['aug']['seed']==42
assert args['clean']['aug_prob']==0 and args['aug']['aug_prob']==0.5
assert read(OUT/'clean_seed42/split_manifest.json')==read(OUT/'aug_seed42/split_manifest.json')

loop=OUT/'closed_loop'
protocol=read(loop/'protocol.json')
fingerprint=protocol_hash(protocol)
results={}
state_pairs={}
total=0
for model in ['base','clean','aug']:
    results[model]={}
    for condition in protocol['conditions']:
        result=read(loop/model/condition/'results.json')
        assert result['protocol_sha256']==fingerprint
        validate_episode_results(result['episode_results'],protocol)
        assert result['episodes']==10
        assert result['successes']==sum(r['success'] for r in result['episode_results'])
        results[model][condition]=result
        for r in result['episode_results']:
            assert r['task_name']=='put_the_bowl_on_the_plate'
            key=(r['task_id'],r['episode_id'])
            pair=(r['env_seed'],r['init_state_sha256'],r['post_settle_state_sha256'])
            if key in state_pairs:assert state_pairs[key]==pair
            else:state_pairs[key]=pair
            assert r['video_frames']==r['steps']+1
            total+=1
assert total==90 and len(state_pairs)==10
paired={c:paired_delta(results['clean'][c]['episode_results'],results['aug'][c]['episode_results'])
        for c in protocol['conditions']}
overview=dict(training=summary,closed_loop={m:{c:dict(successes=r['successes'],episodes=r['episodes'])
        for c,r in conditions.items()} for m,conditions in results.items()},
    aug_vs_clean=paired,protocol_sha256=fingerprint,total_evaluation_episodes=90,
    unique_initial_states=10,same_data_split_and_seed_verified=True,
    physical_state_pairing_verified=True,checkpoint_records=checkpoint_records)
(report_dir/'overview.json').write_text(json.dumps(overview,indent=2))

fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for name,(training,validation) in traces.items():
    steps=[r['step'] for r in training];loss=np.array([r['action_fm_loss'] for r in training])
    line,=axes[0].plot(steps,loss,alpha=.2,label=name+' raw')
    smooth=[]
    for value in loss:smooth.append(value if not smooth else .9*smooth[-1]+.1*value)
    axes[0].plot(steps,smooth,color=line.get_color(),label=name+' EMA(0.9)')
    for ax,key in [(axes[1],'clean_fm_loss'),(axes[2],'corrupt_fm_loss')]:
        ax.plot([r['step'] for r in validation],[r[key] for r in validation],marker='o',label=name)
for ax,title in zip(axes,['Training loss (different input distributions)','Fixed clean validation','Fixed photometric validation']):
    ax.set(title=title,xlabel='Optimizer step',ylabel='Flow-matching MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(report_dir/'loss_curves.png',dpi=130);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for name,(training,_) in traces.items():
    steps=[r['step'] for r in training]
    axes[0].plot(steps,[r['seconds'] for r in training],label=name)
    axes[1].plot(steps,[r['peak_allocated_gib'] for r in training],label=name+' allocated')
    axes[1].plot(steps,[r['peak_reserved_gib'] for r in training],ls='--',label=name+' reserved')
axes[0].set(title='Step time (excludes validation/export/hash)',ylabel='Seconds',xlabel='Optimizer step')
axes[1].set(title='Cumulative PyTorch peak during run',ylabel='GiB',xlabel='Optimizer step')
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(report_dir/'resources.png',dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
x=np.arange(3)
for j,model in enumerate(['base','clean','aug']):
    counts=[results[model][c]['successes'] for c in protocol['conditions']]
    bars=ax.bar(x+(j-1)*.24,counts,width=.23,label=model)
    ax.bar_label(bars,labels=[f'{n}/10' for n in counts],padding=3,fontsize=9)
ax.set(xticks=x,xticklabels=list(protocol['conditions']),ylim=(0,11.5),ylabel='Successful episodes',title='Paired grasp evaluation: task 8, initial states 10–19')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False);ax.grid(axis='y',alpha=.2)
fig.savefig(report_dir/'closed_loop_success.png',dpi=130);plt.close(fig)

font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),13)
video_info=[]
for condition in protocol['conditions']:
    frames={}
    episode_rows={}
    for model in ['base','clean','aug']:
        path=loop/model/condition/'task_08_episode_010.mp4'
        reader=imageio.get_reader(path)
        frames[model]=[frame.copy() for frame in reader];reader.close()
        episode_rows[model]=next(r for r in results[model][condition]['episode_results'] if r['episode_id']==10)
        assert len(frames[model])==episode_rows[model]['video_frames']
    count=max(map(len,frames.values()))
    dest=report_dir/(condition+'_episode10_comparison.mp4')
    with imageio.get_writer(dest,fps=20,codec='libx264',quality=7,macro_block_size=16,
                           ffmpeg_params=['-threads','1']) as writer:
        for i in range(count):
            canvas=np.zeros((480,672,3),np.uint8)
            pil=Image.fromarray(canvas);draw=ImageDraw.Draw(pil)
            for col,model in enumerate(['base','clean','aug']):
                outcome='SUCCESS' if episode_rows[model]['success'] else 'FAIL'
                held=' held' if i>=len(frames[model])-1 else ''
                draw.text((col*224+4,3),f'{model}: {outcome}{held}',font=font,fill='white')
                draw.text((col*224+4,17),f'{condition}  t={i/20:.2f}s',font=font,fill='white')
                source_frame=frames[model][min(i,len(frames[model])-1)]
                assert source_frame.shape[:2]==(448,448)
                pil.paste(Image.fromarray(source_frame[:,224:448]),(col*224,32))
            writer.append_data(np.asarray(pil))
    reader=imageio.get_reader(dest);actual=sum(1 for _ in reader);reader.close()
    assert actual==count
    video_info.append(dict(condition=condition,episode_id=10,frames=count,fps=20,
        comparison_order=['base','clean','aug'],alignment='simulation step; hold last frame after termination',
        source='policy-observed primary/wrist cameras (right half of raw rollout video)',sha256=sha(dest)))
(report_dir/'video_checks.json').write_text(json.dumps(video_info,indent=2))

lines=['# Clean / Aug 各200步与闭环抓取对照','',
    '2026-10-16，单张 A100 40GB。完整官方纹理与 EGL 渲染；真实策略根据当前相机观测输出动作，未使用示范动作回放。','',
    '两组均从同一原始 GlanceWAM 检查点开始，seed=42、同一8条训练/2条验证轨迹、766个训练帧和26个固定验证窗口。batch=1、梯度累积8、AdamW、学习率1e-5、预热20步，共200次更新；唯一训练差异为增强概率0或0.5。每组消费1600个训练窗口，按相同顺序跨越数据集。',
    '', '只更新动作模块，世界模型、VAE、目标时间投影、tap embedding 与未使用的状态编码器冻结；两组官方特征一致性、非零有限梯度、冻结参数哈希和动作参数更新检查通过。','',
    '## 训练与资源','',
    '| 模型 | 最终干净验证损失 | 最终扰动验证损失 | 峰值分配/保留 GiB | 单步均值 s | 进程开始至summary min |',
    '|---|---:|---:|---:|---:|---:|']
for name,s in summary.items():
    v=s['final_validation']
    lines.append(f"| {name} | {v['clean_fm_loss']:.6f} | {v['corrupt_fm_loss']:.6f} | {s['peak_allocated_gib']:.3f}/{s['peak_reserved_gib']:.3f} | {s['step_seconds_mean']:.3f} | {s['process_start_to_summary_seconds']/60:.2f} |")
lines+=['','显存为PyTorch计数；单步耗时不含验证、哈希和保存，完整耗时由/proc进程开始时刻与summary写入时刻计算（约1秒精度）。clean前段与仿真环境安装并行，耗时仅作本次观测，不据此比较速度优劣。训练损失对应不同输入分布；模型比较应看固定验证与闭环结果。','',
    '## 预先固定的闭环评测','',
    'LIBERO goal task_id=8：put the bowl on the plate。初始状态10–19，正常/亮度0.7/RGB增益[0.9,1,1.1]三种条件；原始模型、clean last、aug last共90回合。每回合上限300步、20Hz仿真控制、10步稳定等待、8步动作块和官方目标刷新节奏。init_state哈希、稳定等待后完整状态哈希及环境seed在全部配对条件中一致。','',
    '| 视觉条件 | 原始模型 | clean | aug | aug相对clean恢复/退化 |',
    '|---|---:|---:|---:|---:|']
for c in protocol['conditions']:
    counts=[results[m][c]['successes'] for m in ['base','clean','aug']]
    lines.append(f"| {c} | {counts[0]}/10 | {counts[1]}/10 | {counts[2]}/10 | {paired[c]['recovered']}/{paired[c]['regressed']} |")
lines+=['','这里每个条件只有10个初始状态，三种条件复用同一批状态；不是90个独立场景。另有初始状态9的一回合链路验收，不计入这90回合。训练只做了一个随机种子，原始预训练模型可能见过这些任务与示范，不声称未见任务泛化或统计显著提升。固定扰动验证使用亮度0.7和RGB增益[0.9,1,1.1]的组合，与三种闭环条件分别报告。','',
    '策略服务使用同步官方客户端和缓存目标节奏，仿真会等待推理，不据此声称实机20Hz实时运行。视频按仿真时间20fps播放；评测已包括终止帧。并排视频固定选episode 10，左/中/右为base/clean/aug；较早结束者保持最后一帧并标注held，没有挑选最有利回合。','',
    '## 检查点和复现入口','',
    '正式闭环比较固定使用两组last.pt；best.pt另行保留，但未据闭环结果选择模型。检查点在服务器，哈希与路径见overview.json。模型导出不含优化器状态，不支持精确续训。','',
    '```bash','source /root/autodl-tmp/wam_glance_project/deploy_env.sh',
    'cd /root/autodl-tmp/wam_glance_project/GlanceWAM',
    'cat results/controlled_200/closed_loop/summary.txt',
    '```','',
    '训练入口train_clean_aug_200.sh；闭环入口run_closed_loop_200.sh；已有输出目录拒绝覆盖，复跑前应使用新的实验输出与协议文件名。完整脚本、环境版本、全部回合记录、数值曲线和视频保留服务器；本地审阅包不包含大模型权重。']
(report_dir/'实验记录.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

code=OUT/'experiment_sources';code.mkdir(exist_ok=True)
for name in ['train_clean_aug_200.sh','setup_full_libero_eval.sh','run_closed_loop_200.sh',
             'monitor_200_runs.py','check_full_sim.py','analyze_controlled_200.py','deploy_env.sh']:
    shutil.copy2(ROOT/name,code/name)
for name in ['finetune_action_head.py','finetune_utils.py','eval_robust_libero.py','run_robust_eval.py',
             'paired_policy_server.py','eval_protocol.py','run_finetune_a100_40g.sh']:
    shutil.copy2(MODEL/'day1'/name,code/name)
shutil.copy2(ROOT/'libero_full_source_manifest.json',code/'libero_full_source_manifest.json')
shutil.copy2(ROOT/'logs/libero_eval_requirements.txt',code/'libero_eval_requirements.txt')
shutil.copy2(MODEL/'protocols/grasp200_fixed.json',code/'grasp200_fixed.json')
manifest={}
for p in OUT.rglob('*'):
    if p.is_file() and p.suffix!='.pt' and p.name!='SHA256SUMS.json':
        manifest[p.relative_to(OUT).as_posix()]=sha(p)
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
full=ROOT/'wam_clean_aug_200_full.zip'
review=ROOT/'wam_clean_aug_200_review.zip'
with zipfile.ZipFile(full,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix!='.pt':z.write(p,'controlled_200/'+p.relative_to(OUT).as_posix())
with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if not p.is_file() or p.suffix=='.pt':continue
        rel=p.relative_to(OUT)
        if p.suffix=='.mp4' and rel.parts[0]!='comparison':continue
        if p.suffix=='.npy' or p.name=='SHA256SUMS.json':continue
        z.write(p,'controlled_200/'+rel.as_posix())
    chosen={n:sha(OUT/n[len('controlled_200/'):]) for n in z.namelist()}
    z.writestr('controlled_200/REVIEW_SHA256SUMS.json',json.dumps(chosen,indent=2,ensure_ascii=False))
print(json.dumps({'training':summary,'closed_loop':overview['closed_loop'],
    'review_bytes':review.stat().st_size,'review_sha256':sha(review),'full_bytes':full.stat().st_size},indent=2),flush=True)
