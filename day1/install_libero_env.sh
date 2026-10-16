#!/usr/bin/env bash
# Compatibility installation for official LIBERO, which has setup.py but no uv project.
# Run from the pinned GlanceWAM repository root, after installing uv.
set -euo pipefail
repo_root="$(pwd)"
sim_root="${repo_root}/../LIBERO"
sim_commit=8f1084e3132a39270c3a13ebe37270a43ece2a01
if [ ! -d "${sim_root}" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${sim_root}"
    git -C "${sim_root}" checkout "${sim_commit}"
fi
actual_commit="$(git -C "${sim_root}" rev-parse HEAD)"
if [ "${actual_commit}" != "${sim_commit}" ]; then
    echo "Expected LIBERO ${sim_commit}, found ${actual_commit}. Use a separate checkout or inspect before changing it."
    exit 1
fi
uv venv --python 3.10 "${sim_root}/.venv"
sim_python="${sim_root}/.venv/bin/python"
# Benchmark reads init-state .pt files and config helpers import torch. CPU torch suffices.
uv pip install --python "${sim_python}" torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "${sim_python}" \
    numpy==1.26.4 mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 \
    opencv-python-headless==4.10.0.84 matplotlib==3.8.4 \
    h5py scipy numba easydict pyyaml omegaconf rich tyro \
    'websockets>=12,<16' msgpack imageio imageio-ffmpeg tqdm \
    cloudpickle pillow typing_extensions transforms3d future
uv pip install --python "${sim_python}" --no-deps -e "${sim_root}"
LIBERO_DAY1_ROOT="${sim_root}" "${sim_python}" - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ['LIBERO_DAY1_ROOT']).resolve()
base = root / 'libero/libero'
config = root / 'libero/config.yaml'
if not config.exists():
    config.write_text(yaml.safe_dump({
        'benchmark_root': str(base), 'bddl_files': str(base / 'bddl_files'),
        'init_states': str(base / 'init_files'), 'assets': str(base / 'assets'),
        'datasets': str(root / 'datasets'),
    }))
print('LIBERO_CONFIG_PATH=' + str(root / 'libero'))
PY
export LIBERO_CONFIG_PATH="${sim_root}/libero"
"${sim_python}" -c 'from libero.libero import benchmark; import mujoco; print("LIBERO imports OK", mujoco.__version__)'
uv pip freeze --python "${sim_python}" > "${sim_root}/day1-requirements-resolved.txt"
echo "Simulation interpreter: ${sim_python}"
