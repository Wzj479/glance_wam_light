# Source before running model inference/training on this deployed server.
export WAM_PROJECT_ROOT=/root/autodl-tmp/wam_glance_project
export WAM_MODEL_ROOT="$WAM_PROJECT_ROOT/GlanceWAM"
export PATH="$WAM_MODEL_ROOT/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$WAM_MODEL_ROOT/.venv/lib/ffmpeg6_shim:$WAM_MODEL_ROOT/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
