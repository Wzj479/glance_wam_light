"""CPU round-trip of the real published action weights. No forward/backward/optimizer."""
import argparse
import gc
import json
from pathlib import Path
import tempfile

import torch
from omegaconf import OmegaConf
from action_checkpoint import base_fingerprint, save_action_checkpoint, read_action_checkpoint, apply_action_checkpoint
from glancewam.model.framework.share_tools import read_mode_config
from glancewam.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(4)
    cfg,stats=read_mode_config(Path(a.base));cfg=OmegaConf.create(cfg)
    first=FlowmatchingActionHead(cfg)
    weights=torch.load(a.base,map_location='cpu',weights_only=True)
    action={k[len('action_model.'):]:v for k,v in weights.items() if k.startswith('action_model.')}
    first.load_state_dict(action,strict=True)
    del weights,action;gc.collect()
    identity=base_fingerprint(a.base)
    with tempfile.TemporaryDirectory(prefix='temporary_cpu_export_',dir=out) as directory:
        checkpoint=Path(directory)/'step0.action.pt'
        save_action_checkpoint(first,checkpoint,identity,dict(step=0,audit_only=True,checkpoint_format='action'))
        size=checkpoint.stat().st_size
        base,payload=read_action_checkpoint(checkpoint)
        second=FlowmatchingActionHead(cfg)
        apply_action_checkpoint(second,payload)
        for name,value in first.state_dict().items():
            torch.testing.assert_close(value,second.state_dict()[name],rtol=0,atol=0)
    result=dict(training_started=False,forward_executed=False,backward_executed=False,
        optimizer_constructed=False,real_base_action_weights_loaded_strictly=True,
        full_action_state_roundtrip_bitwise_equal=True,temporary_checkpoint_removed=True,
        checkpoint_bytes=size,action_tensor_count=len(first.state_dict()),base_fingerprint=identity)
    (out/'real_checkpoint_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
