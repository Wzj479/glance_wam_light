"""Load an exported checkpoint through the unmodified official entry point.

Run after training exits, not concurrently (the CPU model needs host RAM).
This checks model reconstruction/loading, not closed-loop control performance.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    a = p.parse_args()
    from accelerate import Accelerator
    from glancewam.model.framework.base_framework import baseframework
    from smoke_action_head import digest
    Accelerator()
    model = baseframework.from_pretrained(a.ckpt)
    report = dict(official_loader_passed=True, checkpoint=str(Path(a.ckpt).resolve()),
        framework=type(model).__name__, action_horizon=model.action_horizon,
        action_sha256=digest(model.action_model.named_parameters()),
        backbone_parameters=sum(p.numel() for p in model.backbone.transformer.parameters()))
    destination = Path(a.ckpt).with_suffix('.load_check.json')
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
