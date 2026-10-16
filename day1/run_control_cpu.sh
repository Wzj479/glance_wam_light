#!/usr/bin/env bash
set -euo pipefail
scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SIM_PYTHON:?Set SIM_PYTHON to your LIBERO Python interpreter}"
: "${LIBERO_CONFIG_PATH:?Set LIBERO_CONFIG_PATH to the directory containing config.yaml}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
"${SIM_PYTHON}" "${scripts}/control_replay_cpu.py" "$@"
