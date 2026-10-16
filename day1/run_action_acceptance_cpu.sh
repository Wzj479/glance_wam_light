#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/LIBERO_CPU_DIAG"
export LIBERO_CONFIG_PATH="$TASK_ROOT/LIBERO_CPU_DIAG/cpu_config"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
PYTHON="$TASK_ROOT/cpu_sim_env/bin/python"
BDDL="$TASK_ROOT/LIBERO_CPU_DIAG/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"
"$PYTHON" day1/accept_action_interface_cpu.py --mode interface --out results/action_interface \
  --bddl "$BDDL" --initial results/cpu_control_diag/initial_state.npy
"$PYTHON" day1/check_action_roundtrip_cpu.py --source data/matched_grasp_source \
  --checkpoint-stats data/matched_grasp_source/checkpoint_dataset_statistics.json \
  --suite-stats data/libero_goal/meta/stats_gr00t.json \
  --official-interface data/interface_source/model2libero_interface.py \
  --official-eval data/interface_source/eval_libero.py --out results/action_interface
"$PYTHON" day1/replay_matched_grasp_cpu.py --source data/matched_grasp_source \
  --out results/matched_grasp_replay --bddl "$BDDL"
"$PYTHON" day1/accept_action_interface_cpu.py --mode video --out results/action_interface \
  --bddl "$BDDL" --initial results/cpu_control_diag/initial_state.npy
"$PYTHON" day1/package_action_acceptance.py --root "$TASK_ROOT"
