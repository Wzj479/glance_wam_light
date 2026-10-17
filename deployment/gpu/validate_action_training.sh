#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
cd "$ROOT/GlanceWAM"
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/ffmpeg6_shim:$PWD/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="${VALIDATION_RUN:-results/gpu_validation/action_finetune_amp_3steps}"
OUT_DIR="$RUN" MAX_STEPS=3 WARMUP_STEPS=1 \
  GRAD_ACCUM=1 EVAL_EVERY=3 SAVE_EVERY=100 VAL_SAMPLES=2 AUG_PROB=0.5 \
  EPISODE_SPLIT="$ROOT/glance_wam_light/results/grasp_episode_split.json" \
  bash day1/run_finetune_a100_40g.sh
python -u day1/verify_export.py --ckpt "$RUN/checkpoints/last.pt" \
  2>&1 | tee "$ROOT/logs/export_reload.log"
python -u day1/smoke_action_head.py --mode infer --ckpt "$RUN/checkpoints/last.pt" \
  --out results/gpu_validation/reloaded_inference 2>&1 | tee "$ROOT/logs/reloaded_inference.log"
touch results/gpu_validation/ALL_PASSED
echo ALL_GPU_VALIDATION_PASSED
