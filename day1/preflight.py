"""Record GPU, disk and interpreter readiness without downloading or modifying packages."""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', default='.')
    p.add_argument('--out', default='preflight_report.json')
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--min-free-gib', type=float, default=60.)
    a = p.parse_args()
    workspace = Path(a.workspace).resolve()
    usage = shutil.disk_usage(workspace)
    report = dict(python=sys.version, executable=sys.executable, workspace=str(workspace),
        free_gib=usage.free / 2**30, min_free_gib=a.min_free_gib,
        nvidia_smi=shutil.which('nvidia-smi'), gpu_available=False, gpu_count=0)
    if report['nvidia_smi']:
        report['nvidia_smi_bytes'] = Path(report['nvidia_smi']).stat().st_size
    if importlib.util.find_spec('torch'):
        try:
            import torch
            report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                          gpu_available=torch.cuda.is_available(), gpu_count=torch.cuda.device_count())
            if report['gpu_available']:
                report.update(gpu=torch.cuda.get_device_name(0),
                    gpu_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                    bf16=torch.cuda.is_bf16_supported())
        except Exception as e:
            report['torch_error'] = str(e)
    report['disk_ready'] = report['free_gib'] >= a.min_free_gib
    report['ready'] = bool(report['gpu_available'] and report.get('bf16') and
                           report.get('gpu_memory_gib', 0) >= 38 and report['disk_ready'])
    Path(a.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if a.require_gpu and not report['ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
