#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON="$TASK_ROOT/cpu_sim_env/bin/python"
"$PYTHON" day1/audit_demo_data.py \
  --root data/libero_goal --out results/grasp_data_audit \
  --plan day1/grasp_audit_plan.json
"$PYTHON" day1/demo_audit_plan.py combine \
  --splits results/grasp_data_audit/split.json \
  --out results/grasp_episode_split.json
"$PYTHON" day1/demo_audit_plan.py combine \
  --splits results/demo_data_audit/split.json results/grasp_data_audit/split.json \
  --out results/drawer_grasp_episode_split.json
