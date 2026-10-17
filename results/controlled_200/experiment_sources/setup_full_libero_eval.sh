#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
cd "$ROOT"
if [[ ! -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y libegl1 libgles2
fi
curl -fL --retry 3 --connect-timeout 15 --max-time 600 \
  https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/zip/8f1084e3132a39270c3a13ebe37270a43ece2a01 \
  -o libero_official_full.zip
/root/miniconda3/bin/python - <<'PY'
import zipfile
from pathlib import Path
root=Path('/root/autodl-tmp/wam_glance_project')
with zipfile.ZipFile(root/'libero_official_full.zip') as z:
    names=z.namelist();prefix=names[0]
    for info in z.infolist():
        if info.is_dir(): continue
        dest=root/'LIBERO'/info.filename[len(prefix):]
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(z.read(info))
PY
/root/miniconda3/bin/conda create -p /root/wam_sim_python310 python=3.10 \
  --override-channels -c https://repo.anaconda.com/pkgs/main -y
UV=/root/miniconda3/bin/uv
"$UV" venv --python /root/wam_sim_python310/bin/python /root/wam_sim_env
PY=/root/wam_sim_env/bin/python
"$UV" pip install --python "$PY" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple \
  numpy==1.26.4 mujoco==3.2.3 robosuite==1.4.0 bddl==1.0.1 gym==0.25.2 \
  opencv-python-headless==4.10.0.84 matplotlib==3.8.4 h5py scipy==1.15.3 \
  numba==0.61.2 easydict pyyaml omegaconf rich tyro websockets==16.0 \
  msgpack imageio imageio-ffmpeg tqdm cloudpickle pillow typing_extensions transforms3d future termcolor
"$UV" pip install --python "$PY" --no-deps -e "$ROOT/LIBERO"
"$PY" - <<'PY'
from pathlib import Path
import yaml
root=Path('/root/autodl-tmp/wam_glance_project/LIBERO/libero/libero')
config=root.parent/'config.yaml'
config.write_text(yaml.safe_dump(dict(benchmark_root=str(root),bddl_files=str(root/'bddl_files'),
    init_states=str(root/'init_files'),assets=str(root/'assets'),datasets=str(root/'datasets'))))
(root/'datasets').mkdir(exist_ok=True)
PY
"$UV" pip freeze --python "$PY" >logs/libero_eval_requirements.txt
echo FULL_LIBERO_ENV_READY
