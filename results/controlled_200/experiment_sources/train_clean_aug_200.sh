#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
source "$ROOT/deploy_env.sh"
cd "$WAM_MODEL_ROOT"
mkdir -p results/controlled_200 protocols
python day1/make_eval_protocol.py --suite libero_goal --task-ids 8 \
  --episode-start 10 --episodes 10 --seed 42 --out protocols/grasp200_fixed.json
for variant in clean aug; do
  probability=0
  if [[ "$variant" == aug ]]; then probability=0.5; fi
  OUT_DIR="results/controlled_200/${variant}_seed42" MAX_STEPS=200 WARMUP_STEPS=20 \
    GRAD_ACCUM=8 EVAL_EVERY=50 SAVE_EVERY=1000 VAL_SAMPLES=1000 \
    SEED=42 AUG_PROB="$probability" \
    EPISODE_SPLIT="$ROOT/glance_wam_light/results/grasp_episode_split.json" \
    bash day1/run_finetune_a100_40g.sh
done
touch results/controlled_200/TRAINING_COMPLETE
echo BOTH_200_STEP_RUNS_COMPLETE
