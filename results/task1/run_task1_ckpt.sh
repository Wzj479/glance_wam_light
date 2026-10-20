#!/usr/bin/env bash
# task-1 run with an arbitrary action checkpoint (for cross-checkpoint verification).
# Usage: run_task1_ckpt.sh <ckpt> <model_name> <proto> <tag> <port> [extra...]
set -uo pipefail
REPO=/root/autodl-tmp/wam_glance_project/glance_wam_light
WORK=/root/autodl-tmp/wam_glance_project/GlanceWAM
LIBERO=/root/autodl-tmp/wam_glance_project/LIBERO
CKPT="${1:?ckpt}"; MODEL="${2:?model}"; PROTO="${3:?proto}"; TAG="${4:?tag}"; PORT="${5:?port}"; shift 5
cd "$WORK"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$WORK:$LIBERO" LIBERO_CONFIG_PATH="$LIBERO/libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$WORK/.venv/lib/ffmpeg6_shim:$WORK/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
out="results/task1/${TAG}"; rm -rf "$out"
echo "=== TASK1 $TAG model=$MODEL port=$PORT extra=[$*] ==="
"$WORK/.venv/bin/python" "$REPO/day1/run_gripper_feedback.py" \
  --protocol "$REPO/results/task1/protocols/$PROTO.json" \
  --condition color_strong --ckpt "$CKPT" --model-name "$MODEL" \
  --libero-python /root/wam_sim_env/bin/python --libero-home "$LIBERO" --port "$PORT" \
  --out "$out" --driver-args --target-body flat_stove_1 "$@"
echo "--- $TAG exit=$? ---"
