#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/wam_glance_project
cd "$ROOT"
for ((i=0; i<240; i++)); do
  if ! kill -0 2499 2>/dev/null; then break; fi
  sleep 5
done
PY="$ROOT/GlanceWAM/.venv/bin/python"
UV=/root/miniconda3/bin/uv
"$UV" pip install --python "$PY" --no-deps "$ROOT/diffusers-pinned"
"$UV" pip install --python "$PY" --no-deps --no-build-isolation -e "$ROOT/GlanceWAM"
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple pytest
cd GlanceWAM
"$PY" tools/setup_ffmpeg6_shim.py
export LD_LIBRARY_PATH="$PWD/.venv/lib/ffmpeg6_shim:$PWD/.venv/lib/python3.11/site-packages/av.libs:${LD_LIBRARY_PATH:-}"
"$PY" - <<'PY'
import json, sys
import torch, diffusers, torchcodec, av
from glancewam.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from glancewam.model.framework.wam.GlanceWAM import GlanceWAM
report = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(), bf16=torch.cuda.is_bf16_supported(),
    diffusers=diffusers.__version__, av=av.__version__, imports_passed=True)
assert torch.cuda.is_available() and report['bf16']
x = torch.randn(128,128,device='cuda',dtype=torch.bfloat16)
assert torch.isfinite(x@x).all()
from pathlib import Path
out=Path('results/gpu_validation');out.mkdir(parents=True,exist_ok=True)
(out/'environment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
PY
"$PY" -m pytest -q day1/test_control_sensitivity.py day1/test_finetune_utils.py day1/test_eval_protocol.py day1/test_robust_pipeline.py
"$UV" pip freeze --python "$PY" >results/gpu_validation/requirements-resolved.txt
echo ENVIRONMENT_READY
