#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$TASK_ROOT/cpu_sim_env/bin/python" -u day1/audit_demo_data.py \
  --root "$TASK_ROOT/data/libero_goal" \
  --out "$TASK_ROOT/results/demo_data_audit" \
  --episodes 10 --val-fraction 0.2 --seed 42
