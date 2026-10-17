import json
from pathlib import Path
import subprocess

ROOT=Path('/root/autodl-tmp/wam_glance_project');OUT=ROOT/'GlanceWAM/results/stage_v1'
status={'training':{},'evaluation':{},'training_complete':(OUT/'TRAINING_COMPLETE').exists()}
for name in ['C','P']:
    run=OUT/(name+'_seed42');p=run/'metrics.jsonl'
    if not p.exists():continue
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    train=[r for r in rows if r['event']=='train'];val=[r for r in rows if r['event']=='validation']
    info=dict(step=train[-1]['step'] if train else 0)
    if train:info.update(last_train_loss=train[-1]['action_fm_loss'],peak_gib=max(r.get('peak_allocated_gib',0) for r in rows))
    if val:info['validation']={k:val[-1][k] for k in ['step','clean_fm_loss','clean_grasp_fm_loss','clean_place_fm_loss']}
    best=run/'checkpoints/best.action.json'
    if best.exists():info['best_step']=json.loads(best.read_text())['step']
    info['summary_exists']=(run/'summary.json').exists();status['training'][name]=info
for p in (OUT/'development').glob('*/*/episodes.jsonl'):
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    status['evaluation'][p.parent.parent.name+'/'+p.parent.name]=dict(completed=len(rows),successes=sum(r['success'] for r in rows))
status['evaluation_complete']=(OUT/'development/summary.json').exists()
status['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
for log in ['stage_v1_training.log','stage_v1_evaluation.log','stage_v1_analysis.log']:
    p=ROOT/log
    if p.exists() and 'Traceback' in p.read_text():status[log+'_error']=p.read_text()[-1500:]
print(json.dumps(status))
