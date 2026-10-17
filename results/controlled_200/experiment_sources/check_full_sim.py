import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['LIBERO_CONFIG_PATH']='/root/autodl-tmp/wam_glance_project/LIBERO/libero'
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,'/root/autodl-tmp/wam_glance_project/LIBERO')
sys.path.insert(0,'/root/autodl-tmp/wam_glance_project/GlanceWAM')
import imageio.v2 as imageio
import numpy as np
import mujoco
from examples.LIBERO.eval_files.eval_libero import _get_libero_env, LIBERO_DUMMY_ACTION, benchmark
from day1.finetune_utils import stable_seed

root=Path('/root/autodl-tmp/wam_glance_project')
manifest=json.loads((root/'libero_full_source_manifest.json').read_text())
for name,sha in manifest['files'].items():
    assert hashlib.sha256((root/'LIBERO'/name).read_bytes()).hexdigest()==sha,name
suite=benchmark.get_benchmark_dict()['libero_goal']()
task=suite.get_task(8)
assert task.name=='put_the_bowl_on_the_plate', task.name
states=suite.get_task_init_states(8)
assert len(states)>19
seed=stable_seed(42,'libero_goal',8,10)%(2**31)
env,instruction=_get_libero_env(task,256,seed)
try:
    env.reset();obs=env.set_init_state(states[10])
    for _ in range(10): obs,_,_,_=env.step(LIBERO_DUMMY_ACTION)
    pair=[np.ascontiguousarray(obs[k][::-1,::-1]) for k in ['agentview_image','robot0_eye_in_hand_image']]
    assert all(x.shape==(256,256,3) for x in pair)
    assert not env.env._check_success()
    out=root/'GlanceWAM/results/controlled_200'
    imageio.imwrite(out/'initial_scene.png',np.concatenate(pair,axis=1))
    report=dict(task_id=8,task=task.name,instruction=instruction,mujoco=mujoco.__version__,
        renderer='egl',official_textures=True,source_commit=manifest['commit'],
        source_files_sha256_verified=len(manifest['files']),initial_state_count=len(states),
        episode_id=10,env_seed=seed,initial_success=False,passed=True)
    (out/'sim_preflight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:env.close()
