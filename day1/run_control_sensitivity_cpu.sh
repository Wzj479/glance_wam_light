#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/LIBERO_CPU_DIAG"
export LIBERO_CONFIG_PATH="$TASK_ROOT/LIBERO_CPU_DIAG/cpu_config"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
"$TASK_ROOT/cpu_sim_env/bin/python" -u day1/control_sensitivity_cpu.py \
  --source data/matched_grasp_source \
  --bddl "$TASK_ROOT/LIBERO_CPU_DIAG/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl" \
  --out results/control_sensitivity --resume
"$TASK_ROOT/cpu_sim_env/bin/python" -u day1/package_control_sensitivity.py --root "$TASK_ROOT"
