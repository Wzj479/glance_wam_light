"""Replay recorded state33 actions without rendering or policy execution."""
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
exec /root/wam_sim_env/bin/python day1/replay_failure_cases.py --physics-only --source results/stage_v1/development --out results/failure_replay_33 --case color_strong:33 --models base C P
'''
t=time.monotonic()
p=subprocess.run(['bash','-c',command])
r=dict(exit_code=p.returncode,seconds=time.monotonic()-t,max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
(ROOT/'replay_33_resource.json').write_text(json.dumps(r,indent=2))
print(r,flush=True)
raise SystemExit(p.returncode)
