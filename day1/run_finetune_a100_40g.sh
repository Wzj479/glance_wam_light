#!/usr/bin/env bash
# From GlanceWAM root: OUT_DIR=results/finetune/goal_aug_seed42 bash day1/run_finetune_a100_40g.sh
# Clean-data control: set AUG_PROB=0 and a different OUT_DIR; keep SEED fixed.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python_bin="${WAM_PYTHON:-${repo_root}/.venv/bin/python}"
out_dir="${OUT_DIR:-results/finetune/goal_aug_seed42}"
# Logs live beside, not inside, the new run directory (the trainer rejects overwrites).
mkdir -p "$(dirname "${out_dir}")"
split_args=()
audit_args=()
if [[ "${AUDIT_ONLY:-0}" == 1 ]]; then audit_args+=(--audit-only); fi
read -r -a val_color <<< "${VAL_COLOR_GAIN:-0.9 1.0 1.1}"
if [[ -n "${EPISODE_SPLIT:-}" ]]; then
    split_args+=(--episode-split "${EPISODE_SPLIT}")
fi
"${python_bin}" day1/finetune_action_head.py \
    --ckpt "${CKPT:-results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt}" \
    --out "${out_dir}" \
    --data-root "${DATA_ROOT:-glancewam_bundle/datasets}" \
    --base-model "${BASE_MODEL:-models/skyreels-base}" \
    --data-mix "${DATA_MIX:-libero_goal}" \
    "${split_args[@]}" \
    "${audit_args[@]}" \
    --max-steps "${MAX_STEPS:-2000}" \
    --batch-size 1 --grad-accum "${GRAD_ACCUM:-8}" \
    --lr "${LR:-1e-5}" --warmup-steps "${WARMUP_STEPS:-100}" \
    --eval-every "${EVAL_EVERY:-200}" --save-every "${SAVE_EVERY:-500}" \
    --val-fraction 0.1 --val-samples "${VAL_SAMPLES:-32}" --val-stride 8 \
    --seed "${SEED:-42}" --workers 0 \
    --train-scope "${TRAIN_SCOPE:-all}" \
    --sampling-mode "${SAMPLING_MODE:-legacy}" \
    --selection-mode "${SELECTION_MODE:-legacy}" \
    --checkpoint-format "${CHECKPOINT_FORMAT:-full}" \
    --normal-loss-tolerance "${NORMAL_LOSS_TOLERANCE:-0.05}" \
    --aug-prob "${AUG_PROB:-0.5}" \
    --brightness 0.6 1.4 --saturation 0.6 1.4 --color-gain 0.9 1.1 \
    --val-brightness "${VAL_BRIGHTNESS:-0.7}" --val-saturation "${VAL_SATURATION:-0.6}" \
    --val-color-gain "${val_color[@]}" \
    2>&1 | tee "${out_dir}.log"
