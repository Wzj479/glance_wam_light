#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
cd "$ROOT/GlanceWAM"
for ((i=0; i<360; i++)); do
  if grep -q '^ENVIRONMENT_READY' "$ROOT/logs/finish_env.log" && [[ -f results/gpu_validation/download_receipt.json ]]; then break; fi
  sleep 5
done
grep -q '^ENVIRONMENT_READY' "$ROOT/logs/finish_env.log"
test -f results/gpu_validation/download_receipt.json
export PATH="$PWD/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/ffmpeg6_shim:$PWD/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python day1/preflight.py --workspace . --require-gpu --min-free-gib 25 --out results/gpu_validation/preflight.json
python day1/prepare_checkpoint.py --data-mix libero_goal
python -u day1/smoke_action_head.py --mode infer \
  --ckpt results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt \
  --out results/gpu_validation/base_inference 2>&1 | tee "$ROOT/logs/base_inference.log"
OUT_DIR=results/gpu_validation/action_finetune_3steps MAX_STEPS=3 WARMUP_STEPS=1 \
  GRAD_ACCUM=1 EVAL_EVERY=3 SAVE_EVERY=100 VAL_SAMPLES=2 AUG_PROB=0.5 \
  EPISODE_SPLIT="$ROOT/glance_wam_light/results/grasp_episode_split.json" \
  bash day1/run_finetune_a100_40g.sh
python -u day1/verify_export.py --ckpt results/gpu_validation/action_finetune_3steps/checkpoints/last.pt \
  2>&1 | tee "$ROOT/logs/export_reload.log"
python -u day1/smoke_action_head.py --mode infer \
  --ckpt results/gpu_validation/action_finetune_3steps/checkpoints/last.pt \
  --out results/gpu_validation/reloaded_inference 2>&1 | tee "$ROOT/logs/reloaded_inference.log"
touch results/gpu_validation/ALL_PASSED
echo ALL_GPU_VALIDATION_PASSED
