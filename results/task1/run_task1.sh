#!/usr/bin/env bash
# Second grasp-and-place task: does the premature-release mechanism + online rule transfer?
# Usage: run_task1.sh <proto-a|proto-b> <tag> <port> <extra-driver-args...>
set -uo pipefail
REPO=/root/autodl-tmp/wam_glance_project/glance_wam_light
WORK=/root/autodl-tmp/wam_glance_project/GlanceWAM
LIBERO=/root/autodl-tmp/wam_glance_project/LIBERO
HANDOVER=/root/autodl-tmp/glance_handover
PROTO="${1:?proto}"; TAG="${2:?tag}"; PORT="${3:?port}"; shift 3
cd "$WORK"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$WORK:$LIBERO" LIBERO_CONFIG_PATH="$LIBERO/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$WORK/.venv/lib/ffmpeg6_shim:$WORK/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
out="results/task1/${TAG}"; rm -rf "$out"
echo "=== TASK1 $TAG ($PROTO) port $PORT extra=[$*] ==="
"$WORK/.venv/bin/python" "$REPO/day1/run_gripper_feedback.py" \
  --protocol "$HANDOVER/$PROTO.json" \
  --condition color_strong --ckpt results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt --model-name base \
  --libero-python /root/wam_sim_env/bin/python --libero-home "$LIBERO" --port "$PORT" \
  --out "$out" --driver-args --target-body flat_stove_1 "$@"
echo "--- $TAG exit=$? ---"
