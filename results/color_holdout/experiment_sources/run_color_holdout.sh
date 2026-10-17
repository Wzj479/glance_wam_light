#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
source "$ROOT/deploy_env.sh"
cd "$WAM_MODEL_ROOT"
python day1/run_robust_eval.py \
  --protocol protocols/grasp_color_holdout.json \
  --model base=results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt \
  --model clean=results/controlled_200/clean_seed42/checkpoints/last.pt \
  --model aug=results/controlled_200/aug_seed42/checkpoints/last.pt \
  --out results/color_holdout \
  --model-python "$WAM_MODEL_ROOT/.venv/bin/python" \
  --libero-python /root/wam_sim_env/bin/python \
  --libero-home "$ROOT/LIBERO" --save-videos
