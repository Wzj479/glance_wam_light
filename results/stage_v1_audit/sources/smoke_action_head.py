"""Real-data inference / frozen-backbone action-head update for GlanceWAM.

Pinned upstream: 6a51bdc5139e72e8ad525779db0c989e1af5abab.
Run as: python day1/smoke_action_head.py --mode infer|train --ckpt ...
This is a feasibility check, not a convergence or benchmark result.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('WANDB_MODE', 'disabled')
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from glancewam.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from glancewam.model.framework.share_tools import read_mode_config
from glancewam.model.framework.wam.GlanceWAM import GlanceWAM


def digest(named_tensors):
    """Hash every byte; CPU copy one tensor at a time, no second GPU model."""
    h = hashlib.sha256()
    for name, t in named_tensors:
        h.update(name.encode())
        h.update(str((tuple(t.shape), t.dtype)).encode())
        raw = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        h.update(raw.tobytes())
    return h.hexdigest()


def mem():
    return dict(allocated_gib=torch.cuda.memory_allocated() / 2**30,
                reserved_gib=torch.cuda.memory_reserved() / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def load(a):
    d, stats = read_mode_config(Path(a.ckpt))
    cfg = OmegaConf.create(d)
    cfg.framework.world_model.compile_dit = False
    cfg.framework.world_model.compile_vae = False
    cfg.framework.action_model.repeated_diffusion_steps = 1
    cfg.trainer.sdpa_backend = 'auto'
    cfg.datasets.vla_data.t5_cache.attach_embeds = True
    cfg.datasets.vla_data.t5_cache.enabled = True
    cfg.datasets.vla_data.t5_cache.max_length = 512
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 0
    model = GlanceWAM(cfg)
    sd = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    result = model.load_state_dict(sd, strict=False)
    allowed = tuple(model._reconstructable_prefixes)
    bad_missing = [k for k in result.missing_keys if not k.startswith(allowed)]
    bad_extra = [k for k in result.unexpected_keys if not k.startswith(allowed)]
    if bad_missing or bad_extra:
        raise RuntimeError({'missing': bad_missing, 'unexpected': bad_extra})
    del sd
    model.norm_stats = stats
    model.requires_grad_(False)
    model.to(device='cuda', dtype=torch.bfloat16).eval()
    # Use the official dataset and normalization; a plain DataLoader avoids
    # upstream persistent_workers=True being invalid when num_workers=0.
    ds = get_vla_dataset(cfg.datasets.vla_data)
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=collate_fn, shuffle=False)
    batch = next(iter(loader))
    if 'state' in batch[0]:
        raise RuntimeError('Expected the released stateless LIBERO contract; investigate data config.')
    if 'lang_embed' not in batch[0]:
        raise RuntimeError('Missing real UMT5 cache embedding; do not substitute a zero tensor.')
    return model, batch, cfg


class FeaturesReady(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['infer', 'train'], required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--out', default='results/day1')
    a = p.parse_args()
    if a.steps < 1:
        p.error('--steps must be positive')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), 'CUDA is not available'
    assert torch.cuda.is_bf16_supported(), 'BF16 is not supported'
    Accelerator()  # initialize Accelerate logging state; no DeepSpeed trainer import
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    model, batch, cfg = load(a)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'
    report = dict(mode=a.mode, commit=commit, checkpoint=str(Path(a.ckpt).resolve()),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, instruction=batch[0]['lang'],
                  action_target_shape=list(np.asarray(batch[0]['action']).shape),
                  load_memory=mem(), batch_size=1,
                  repeated_diffusion_steps=1, learning_rate=a.lr)
    OmegaConf.save(cfg, out / f'{a.mode}_effective_config.yaml')
    if a.mode == 'infer':
        sample = {'image': batch[0]['image'][0], 'lang': batch[0]['lang'], 'seed': 42}
        # Guarantee no uncached instruction can silently load the 11 GB text encoder.
        assert model._eval_text_from_table([sample['lang']]) is not None, 'Resident text cache miss'
        rows = []
        for i in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                prediction = model.predict_action([sample])
            torch.cuda.synchronize()
            actions = prediction['normalized_actions']
            assert actions.shape == (1, model.action_horizon, 7), actions.shape
            assert np.isfinite(actions).all()
            assert np.isfinite(prediction['goal_latent']).all()
            rows.append(dict(iteration=i, seconds=time.perf_counter() - start, **mem()))
        # All three calls regenerate the goal: latency is NOT a fast cached-goal control tick.
        np.savez(out / 'inference_sample.npz', **prediction)
        report.update(rows=rows, action_shape=list(actions.shape),
                      generated_goal_shape=list(prediction['goal_latent'].shape),
                      timing_scope='observation encoding + new goal generation + action sampling')
    else:
        model.action_model.float().requires_grad_(True)
        # Freeze goal_time_proj/tap_embed too; train exactly action_model.*.
        named = [(n, t) for n, t in model.named_parameters() if t.requires_grad]
        assert named and all(n.startswith('action_model.') for n, _ in named)
        frozen = [(n, t) for n, t in model.named_parameters() if not t.requires_grad]
        frozen_before = digest(frozen)
        action_before = digest(model.action_model.named_parameters())
        optimizer = torch.optim.AdamW([t for _, t in named], lr=a.lr, foreach=False)
        report.update(total_parameters=sum(t.numel() for t in model.parameters()),
                      trainable_parameters=sum(t.numel() for _, t in named))
        rows = []
        for step in range(a.steps):
            model.eval()  # keep the entire frozen feature path in eval mode
            model.action_model.eval()
            captured = {}

            def capture(_module, inputs):
                captured['args'] = tuple(x.detach() if torch.is_tensor(x) else x for x in inputs)
                # Stop before the original head/video loss; optimize the head explicitly below.
                raise FeaturesReady()

            hook = model.action_model.register_forward_pre_hook(capture)
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    model(batch)
            except FeaturesReady:
                pass
            finally:
                hook.remove()
            if 'args' not in captured:
                raise RuntimeError('The upstream action-head interface changed')
            assert not model.backbone.training
            args = captured['args']
            assert all(not x.requires_grad for x in args if torch.is_tensor(x))
            model.action_model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model.action_model(*args)
            assert torch.isfinite(loss).all(), 'Nonfinite action loss'
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([t for _, t in named], 1.0, error_if_nonfinite=True)
            assert grad_norm.item() > 0, 'No nonzero action-head gradient'
            assert all(t.grad is None for _, t in frozen), 'Gradient leaked to frozen parameters'
            optimizer.step()  # Adam states are created here; include them in the memory measurement
            torch.cuda.synchronize()
            rows.append(dict(step=step, action_loss=loss.item(), grad_norm=grad_norm.item(),
                             seconds=time.perf_counter() - start, **mem()))
            print(json.dumps(rows[-1]), flush=True)
            del captured, args, loss
        frozen_after = digest(frozen)
        action_after = digest(model.action_model.named_parameters())
        assert frozen_before == frozen_after, 'Frozen parameter hash changed'
        assert action_before != action_after, 'Action-head parameters did not update'
        report.update(rows=rows, frozen_parameters_unchanged=True, action_parameters_changed=True,
                      frozen_sha256=frozen_after,
                      timing_scope='real batch frozen-feature extraction + head forward/backward + optimizer step')
        # Smoke artifact only: not an official full checkpoint and not directly deployable.
        torch.save({'action_model': {k: v.detach().cpu() for k, v in model.action_model.state_dict().items()},
                    'base_checkpoint': str(Path(a.ckpt).resolve()), 'steps': a.steps},
                   out / 'action_head_smoke.pt')
    report['final_memory'] = mem()
    report['passed'] = True
    (out / f'{a.mode}_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
