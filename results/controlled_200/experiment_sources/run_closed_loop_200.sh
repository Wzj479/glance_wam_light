#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
source "$ROOT/deploy_env.sh"
cd "$WAM_MODEL_ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GLANCEWAM_EGL_DEVICE_ID=0
export NUMBA_NUM_THREADS=4
for ((i=0; i<1800; i++)); do
  if [[ -f results/controlled_200/TRAINING_COMPLETE && -f results/controlled_200/sim_preflight.json ]]; then break; fi
  sleep 5
done
test -f results/controlled_200/TRAINING_COMPLETE
python - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('results/controlled_200/sim_preflight.json').read_text())['passed']
p=json.loads(Path('protocols/grasp200_fixed.json').read_text())
p['episode_ids']=[9];p['conditions']={'clean':p['conditions']['clean']}
Path('protocols/grasp200_integration.json').write_text(json.dumps(p,indent=2))
PY
python -u day1/run_robust_eval.py --protocol protocols/grasp200_integration.json \
  --model base=results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt \
  --out results/controlled_200/integration_base --port 6694 \
  --libero-python /root/wam_sim_env/bin/python --libero-home ../LIBERO --save-videos
python -u day1/run_robust_eval.py --protocol protocols/grasp200_fixed.json \
  --model base=results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt \
  --model clean=results/controlled_200/clean_seed42/checkpoints/last.pt \
  --model aug=results/controlled_200/aug_seed42/checkpoints/last.pt \
  --out results/controlled_200/closed_loop --port 6694 \
  --libero-python /root/wam_sim_env/bin/python --libero-home ../LIBERO --save-videos
touch results/controlled_200/ALL_COMPLETE
echo TRAINING_AND_90_CLOSED_LOOP_EPISODES_COMPLETE
