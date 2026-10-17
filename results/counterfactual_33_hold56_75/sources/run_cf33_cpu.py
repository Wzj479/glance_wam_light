"""One pre-specified intervention, saved P actions56..75 gripper +1 only."""
import json
from pathlib import Path
import resource
import subprocess
import time

ROOT=Path('/root/autodl-tmp/wam_glance_project')
command=f'''source {ROOT}/deploy_env.sh
cd "$WAM_MODEL_ROOT"
export PYTHONPATH="$WAM_MODEL_ROOT:{ROOT}/LIBERO"
export LIBERO_CONFIG_PATH={ROOT}/LIBERO/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
exec /root/wam_sim_env/bin/python day1/replay_failure_cases.py --physics-only --source results/stage_v1/development --out results/counterfactual_33_hold56_75 --case color_strong:33 --models P --hold-close 56 75 --reference-replay results/failure_replay_33/color_strong_episode33/P/replay.npz
'''
t=time.monotonic();p=subprocess.run(['bash','-c',command])
result=dict(exit_code=p.returncode,seconds=time.monotonic()-t,
    max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
(ROOT/'cf33_resource.json').write_text(json.dumps(result,indent=2))
print(result,flush=True)
raise SystemExit(p.returncode)
