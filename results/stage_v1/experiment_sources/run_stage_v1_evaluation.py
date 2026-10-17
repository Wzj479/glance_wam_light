"""Use fixed development states only; honor best-step0 fallback and label last diagnostics."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT=Path('/root/autodl-tmp/wam_glance_project');MODEL=ROOT/'GlanceWAM';OUT=MODEL/'results/stage_v1'
sys.path.insert(0,str(MODEL/'day1'))
from eval_protocol import protocol_hash
from run_robust_eval import aggregate
from action_checkpoint import read_action_checkpoint

deadline=time.monotonic()+7200
while not (OUT/'TRAINING_COMPLETE').exists():
    log=(ROOT/'stage_v1_training.log').read_text() if (ROOT/'stage_v1_training.log').exists() else ''
    if 'Traceback' in log:raise RuntimeError('Training failed; evaluation stopped')
    if time.monotonic()>deadline:raise TimeoutError('Training deadline exceeded')
    time.sleep(10)
protocol=json.loads((OUT/'execution_plan.json').read_text())['development_protocol']
assert protocol['episode_ids']==list(range(25,45)) and not set(protocol['episode_ids'])&set(range(45,50))
fingerprint=protocol_hash(protocol)
base_source=MODEL/'results/color_holdout/base'
for condition in protocol['conditions']:
    result=json.loads((base_source/condition/'results.json').read_text())
    assert result['protocol_sha256']==fingerprint
evaluation={};arguments=[]
for variant in ['C','P']:
    run=OUT/(variant+'_seed42')
    best=json.loads((run/'checkpoints/best.action.json').read_text())
    step=best['step'];checkpoint=run/'checkpoints/best.action.pt'
    role='selected_best'
    if step==0:
        # The published base is cast to BF16 for deployment. Check exact action
        # equivalence in that dtype before reusing its rollouts as the primary.
        base_path,payload=read_action_checkpoint(checkpoint)
        base=torch.load(base_path,map_location='cpu',weights_only=True)
        source={k[len('action_model.'):]:v for k,v in base.items() if k.startswith('action_model.')}
        assert set(source)==set(payload['action_state_dict'])
        for key,value in payload['action_state_dict'].items():
            torch.testing.assert_close(value.to(torch.bfloat16),source[key].to(torch.bfloat16),rtol=0,atol=0)
        del base,source,payload
        checkpoint=run/'checkpoints/last.action.pt';role='last_diagnostic_best_rejected'
    read_action_checkpoint(checkpoint) # verify file/base/config/statistics before any rollout
    evaluation[variant]=dict(best_step=step,primary_source='base_reused' if step==0 else variant,
        rollout_role=role,rollout_checkpoint=str(checkpoint),
        rollout_checkpoint_step=json.loads(checkpoint.with_suffix('.json').read_text())['step'])
    arguments += ['--model',variant+'='+str(checkpoint)]
(OUT/'evaluation_selection.json').write_text(json.dumps(dict(models=evaluation,
    baseline_reused=str(base_source),protocol_sha256=fingerprint,reserved_initial_states=list(range(45,50))),indent=2))
destination=OUT/'development'
command=[str(MODEL/'.venv/bin/python'),'day1/run_robust_eval.py',
    '--protocol',str(MODEL/'protocols/grasp_color_holdout.json'),*arguments,
    '--out',str(destination),'--model-python',str(MODEL/'.venv/bin/python'),
    '--libero-python','/root/wam_sim_env/bin/python','--libero-home',str(ROOT/'LIBERO'),'--save-videos']
started=time.perf_counter()
subprocess.run(command,cwd=MODEL,check=True)
assert not (destination/'base').exists()
shutil.copytree(base_source,destination/'base')
aggregate(destination,protocol,['base','C','P'])
models=json.loads((destination/'models.json').read_text())
models['base']=str(MODEL/'results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt')
(destination/'models.json').write_text(json.dumps(models,indent=2))
(OUT/'evaluation_wall_time.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
    new_policy_rollouts=80,reused_baseline_rollouts=40),indent=2))
(OUT/'EVALUATION_COMPLETE').touch()
