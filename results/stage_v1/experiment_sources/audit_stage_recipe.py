"""CPU-only real data/architecture audit for C/P. Never constructs a world model or optimizer."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from finetune_action_head import configure, build_splits
from stage_finetune_utils import build_phase_index, draw_schedule
from action_training_scope import configure_action_scope
from glancewam.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c-audit',required=True);p.add_argument('--p-audit',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    c,pdir=Path(a.c_audit),Path(a.p_audit)
    splitc=json.loads((c/'split_manifest.json').read_text());splitp=json.loads((pdir/'split_manifest.json').read_text())
    phasec=json.loads((c/'phase_manifest.json').read_text());phasep=json.loads((pdir/'phase_manifest.json').read_text())
    assert splitc==splitp and phasec==phasep
    ca=json.loads((c/'train_args.json').read_text());pa=json.loads((pdir/'train_args.json').read_text())
    common=lambda x:{k:v for k,v in x.items() if k not in ('out','sampling_mode')}
    assert common(ca)==common(pa)
    assert ca['sampling_mode']=='uniform' and pa['sampling_mode']=='phase'
    for key,value in dict(train_scope='crossattn_decoder',lr=3e-6,max_steps=100,seed=42,
                          aug_prob=0.,batch_size=1,grad_accum=8,checkpoint_format='action').items():
        assert ca[key]==value,(key,ca[key])
    args=argparse.Namespace(**ca)
    cfg,stats=configure(args)
    assert not cfg.framework.action_model.get('t_loss_weighting',False)
    datasets,train,val,manifest=build_splits(cfg,args,stats)
    tags,pools,phase=build_phase_index(datasets,train,val)
    assert manifest==splitc and phase==phasec
    # Architecture construction only, entirely on CPU. No pretrained weights loaded.
    torch.manual_seed(42);torch.set_num_threads(4)
    head=FlowmatchingActionHead(cfg)
    wrapper=nn.Module();wrapper.action_model=head
    trainable,frozen=configure_action_scope(wrapper,'crossattn_decoder')
    assert all('.attn1.' in n or n.startswith('action_model.action_decoder.') for n,p in trainable)
    assert all(p.device.type=='cpu' for p in head.parameters())
    parameters=dict(trainable_parameters=sum(p.numel() for n,p in trainable),
        action_parameters=sum(p.numel() for p in head.parameters()),
        trainable_names=[n for n,p in trainable],frozen_names=[n for n,p in frozen],
        action_state_bytes=sum(t.numel()*t.element_size() for t in head.state_dict().values()))
    (out/'parameter_scope.json').write_text(json.dumps(parameters,indent=2))
    # Decode just one real sample to verify the official 8x7 action packing,
    # labels and cached language; no feature/model forward follows this read.
    from finetune_action_head import EpisodeFrames
    sample=EpisodeFrames(datasets,[train[0]],42,phase_tags=tags)[0]
    import numpy as np
    action=np.asarray(sample['action'])
    assert action.shape==(8,7) and np.isfinite(action).all()
    assert np.all((action[:,6]==0)|(action[:,6]==1))
    selected=datasets[train[0][0]]
    episode,frame=map(int,selected.all_steps[train[0][1]])
    rowdata=selected.get_trajectory_data(episode)
    # The current export has separate raw columns; use the official modality slices.
    field=selected.lerobot_modality_meta.action['gripper']
    g=np.stack(rowdata[field.original_key or 'gripper'])[:,field.start:field.end].reshape(-1)
    expected=g[np.clip(np.arange(frame,frame+8),0,len(g)-1)]
    assert np.array_equal(action[:,6],expected)
    result=dict(training_started=False,model_forward_executed=False,optimizer_constructed=False,
        c_p_split_identical=True,c_p_phase_manifest_identical=True,c_p_config_differs_only_out_and_sampler=True,
        train_windows=len(train),validation_windows=len(val),
        train_pool_counts=phase['train_pool_counts'],
        validation_phase_counts={k:sum(k in x for x in phase['validation_tags']) for k in ['grasp','place']},
        sample_action_shape=list(action.shape),sample_image_slots=len(sample['image']),
        checkpoint_normalization=manifest[selected.dataset_name],
        trainable_parameters=parameters['trainable_parameters'],action_state_bytes=parameters['action_state_bytes'],
        loss='Unmodified upstream seven-dimensional joint FM MSE; no extra channel/phase multiplier',
        loss_selection='Macro normalized clean grasp/place FM, clean global <= 1.05x step0, strict improvement over step0',
        phase_manifest_sha256=hashlib.sha256((c/'phase_manifest.json').read_bytes()).hexdigest())
    (out/'audit_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
