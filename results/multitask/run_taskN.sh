#!/usr/bin/env bash
# task-aware run: run_taskN.sh <proto> <tag> <port> <grasp_body> <target_body> [extra...]
set -uo pipefail
REPO=/root/autodl-tmp/wam_glance_project/glance_wam_light
WORK=/root/autodl-tmp/wam_glance_project/GlanceWAM
LIBERO=/root/autodl-tmp/wam_glance_project/LIBERO
PROTO="${1:?proto}"; TAG="${2:?tag}"; PORT="${3:?port}"; GRASP="${4:?grasp}"; TARGET="${5:?target}"; shift 5
cd "$WORK"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$WORK:$LIBERO" LIBERO_CONFIG_PATH="$LIBERO/libero" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$WORK/.venv/lib/ffmpeg6_shim:$WORK/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
out="results/${TAG_DIR:-multitask}/${TAG}"; rm -rf "$out"
echo "=== $TAG ($PROTO) grasp=$GRASP target=$TARGET port=$PORT extra=[$*] ==="
"$WORK/.venv/bin/python" "$REPO/day1/run_gripper_feedback.py" \
  --protocol "$REPO/results/multitask/protocols/$PROTO.json" \
  --condition color_strong --ckpt results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt --model-name base \
  --libero-python /root/wam_sim_env/bin/python --libero-home "$LIBERO" --port "$PORT" \
  --out "$out" --driver-args --grasp-body "$GRASP" --target-body "$TARGET" "$@"
echo "--- $TAG exit=$? ---"
