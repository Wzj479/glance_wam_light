#!/usr/bin/env bash
# Explicitly invoked only. C/P differ solely in sampling; AUDIT_ONLY=1 does not train.
set -euo pipefail
variant="${1:?Usage: bash day1/run_phase_finetune_a100_40g.sh C|P}"
case "$variant" in
  C) export SAMPLING_MODE=uniform ;;
  P) export SAMPLING_MODE=phase ;;
  *) echo 'Expected C or P' >&2; exit 2 ;;
esac
export SEED=42 TRAIN_SCOPE=crossattn_decoder AUG_PROB=0
export MAX_STEPS=100 WARMUP_STEPS=10 LR=3e-6 GRAD_ACCUM=8
export EVAL_EVERY=25 SAVE_EVERY=1000 VAL_SAMPLES=1000
export SELECTION_MODE=phases NORMAL_LOSS_TOLERANCE=0.05 CHECKPOINT_FORMAT=action
export VAL_BRIGHTNESS=1 VAL_SATURATION=1 VAL_COLOR_GAIN='0.7 1.0 1.3'
export EPISODE_SPLIT="${EPISODE_SPLIT:-../glance_wam_light/results/grasp_episode_split.json}"
export OUT_DIR="${OUT_DIR:-results/stage_v1/${variant}_seed42}"
bash "$(dirname "$0")/run_finetune_a100_40g.sh"
