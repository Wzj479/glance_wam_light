"""Sequential authorized C/P training. Development evaluation reserves states 45..49."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/root/autodl-tmp/wam_glance_project')
MODEL=ROOT/'GlanceWAM';OUT=MODEL/'results/stage_v1'
if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
OUT.mkdir(exist_ok=True)
sources=MODEL/'results/stage_v1_audit/sources'
for p in sources.iterdir():
    if p.suffix in ('.py','.sh'):
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((MODEL/'day1'/p.name).read_bytes()).digest(),p.name
protocol=json.loads((MODEL/'protocols/grasp_color_holdout.json').read_text())
assert protocol['episode_ids']==list(range(25,45)) and protocol['task_ids']==[8] and protocol['seed']==42
assert list(protocol['conditions'])==['clean','color_strong']
plan=dict(training=['C','P'],steps_each=100,seed=42,development_protocol=protocol,
    reserved_initial_states=list(range(45,50)),baseline_source='results/color_holdout/base',
    evaluation_rule='Use best as primary. If best is step0, primary stays base; evaluate last only as diagnostic. Never silently replace rejected best with last.',
    baseline_reuse='Reuse prior baseline with identical protocol; check hashes, seeds and post-settle states in aggregation.')
(OUT/'execution_plan.json').write_text(json.dumps(plan,indent=2))
env=os.environ.copy();env.pop('AUDIT_ONLY',None);env.pop('OUT_DIR',None)
env['CUDA_VISIBLE_DEVICES']='0'
times={}
for variant in ['C','P']:
    t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(dict(event='start',variant=variant,utc=start)),flush=True)
    proc=subprocess.run(['bash','day1/run_phase_finetune_a100_40g.sh',variant],cwd=MODEL,env=env)
    times[variant]=dict(start_utc=start,seconds=time.perf_counter()-t,exit_code=proc.returncode)
    (OUT/'run_wall_times.json').write_text(json.dumps(times,indent=2))
    if proc.returncode:raise RuntimeError(f'{variant} failed; do not start next stage')
    run=OUT/(variant+'_seed42');summary=json.loads((run/'summary.json').read_text())
    assert summary['completed_steps']==100 and summary['frozen_unchanged']
    records=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in records if r['event']=='train']
    assert [r['step'] for r in updates]==list(range(1,101))
    assert all(math.isfinite(r['action_fm_loss']) and r['grad_norm']>0 for r in updates)
    expected=dict(uniform=8) if variant=='C' else dict(uniform=4,grasp=2,place=2)
    assert all(r['sampling_source_counts']==expected for r in updates)
    print(json.dumps(dict(event='complete',variant=variant,wall_seconds=times[variant]['seconds'],best_step=summary['best_step'])),flush=True)
for filename in ['split_manifest.json','phase_manifest.json']:
    assert (OUT/'C_seed42'/filename).read_bytes()==(OUT/'P_seed42'/filename).read_bytes()
c=json.loads((OUT/'C_seed42/summary.json').read_text());p=json.loads((OUT/'P_seed42/summary.json').read_text())
assert c['frozen_sha256']==p['frozen_sha256']
(OUT/'TRAINING_COMPLETE').touch()
