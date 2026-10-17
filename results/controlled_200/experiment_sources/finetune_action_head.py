"""Single-A100 frozen-world-model action-head SFT for pinned GlanceWAM.

Use from GlanceWAM root: python day1/finetune_action_head.py --help
Official commit: 6a51bdc5139e72e8ad525779db0c989e1af5abab.
Validation is held-out hindsight-goal flow-matching loss, NOT closed-loop success.
The pretrained checkpoint may already have seen the held-out episodes.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
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
from torch.utils.data import DataLoader, Dataset, get_worker_info

from finetune_utils import (augment_sample, checkpoint_action_metadata, deployment_keys, load_episode_partition, photometric_window,
                            split_episodes, split_step_indices, stable_seed)
from smoke_action_head import digest, mem
from glancewam.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from glancewam.dataloader.lerobot_datasets import collate_fn, make_LeRobotSingleDataset
from glancewam.model.framework.share_tools import read_mode_config
from glancewam.model.framework.wam.GlanceWAM import GlanceWAM

PIN = '6a51bdc5139e72e8ad525779db0c989e1af5abab'
T5_ID = 'Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers'


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def atomic_torch(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def log(out, row):
    with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


class EpisodeFrames(Dataset):
    """Index actual single datasets; never use the random mixture __getitem__."""
    def __init__(self, datasets, entries, seed, epoch=0):
        self.datasets, self.entries, self.seed, self.epoch = datasets, entries, seed, epoch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        di, local_index = self.entries[index]
        ds = self.datasets[di]
        episode, frame = ds.all_steps[local_index]
        # Upstream goal offsets use a private per-worker RNG: seed per sample to
        # reproduce validation goals and resumed training with different worker timing.
        seed = stable_seed(self.seed, self.epoch, ds.dataset_name, int(episode), int(frame))
        info = get_worker_info()
        ds._future_rng_worker = info.id if info else -1
        ds._future_rng = np.random.default_rng(seed)
        sample = ds[local_index]
        if 'lang_embed' not in sample or 'state' in sample:
            raise RuntimeError('Expected real cached UMT5 embeddings and stateless LIBERO samples')
        sample['_sample_seed'] = seed
        return sample


def build_splits(cfg, a, checkpoint_stats):
    data_cfg = cfg.datasets.vla_data
    datasets, train_entries, val_entries, manifest = [], [], [], {}
    seen = set()
    if a.episode_split:
        supplied = json.loads(Path(a.episode_split).read_text(encoding='utf-8'))
        expected = {name for name, _, _ in DATASET_NAMED_MIXTURES[a.data_mix]}
        if set(supplied['datasets']) != expected:
            raise ValueError('Explicit split datasets must exactly match --data-mix')
    for name, _weight, robot_type in DATASET_NAMED_MIXTURES[a.data_mix]:
        if name in seen:
            continue
        seen.add(name)
        ds = make_LeRobotSingleDataset(
            Path(data_cfg.data_root_dir), name, robot_type, data_cfg=data_cfg, eval_split='train')
        # SingleDataset defaults to per-suite bounds; pretrained inference uses
        # merged checkpoint bounds. In particular goal yaw.min differs from franka.
        # Set metadata BEFORE packing float16 actions, avoiding a lossy re-normalize.
        action_stats = checkpoint_stats['franka']['action']
        ds.set_transforms_metadata(checkpoint_action_metadata(ds.metadata, action_stats))
        # Split only eligible episodes actually present in all_steps.
        eligible = {int(t) for t, _ in ds.all_steps}
        if a.episode_split:
            tr, va = load_episode_partition(a.episode_split, name, eligible)
        else:
            tr, va = split_episodes(eligible, a.val_fraction, stable_seed(a.seed, name))
        ti, vi = split_step_indices(ds.all_steps, tr, va, a.val_stride)
        di = len(datasets)
        datasets.append(ds)
        train_entries.extend((di, i) for i in ti)
        val_entries.extend((di, i) for i in vi)
        manifest[name] = dict(train_episode_ids=tr, val_episode_ids=va,
                              train_frames=len(ti), val_frames=len(vi),
                              action_normalization_source='checkpoint.franka.action',
                              action_min=action_stats['min'], action_max=action_stats['max'],
                              action_mask=action_stats['mask'])
    # Fixed representative subset across all selected datasets; full episode IDs remain recorded.
    rng = np.random.default_rng(stable_seed(a.seed, 'val_subset'))
    if len(val_entries) > a.val_samples:
        chosen = sorted(rng.choice(len(val_entries), a.val_samples, replace=False).tolist())
        val_entries = [val_entries[i] for i in chosen]
    manifest['validation_windows'] = [dict(dataset=datasets[d].dataset_name,
        episode=int(datasets[d].all_steps[i][0]), frame=int(datasets[d].all_steps[i][1]))
        for d, i in val_entries]
    manifest['scope'] = 'Adaptation holdout only; pretrained model may have seen these episodes'
    if a.episode_split:
        manifest['explicit_split_source'] = dict(path=str(Path(a.episode_split).resolve()),
            sha256=hashlib.sha256(Path(a.episode_split).read_bytes()).hexdigest())
    return datasets, train_entries, val_entries, manifest


@torch.no_grad()
@torch.autocast('cuda', dtype=torch.bfloat16)
def extract_frozen_features(model, batch, goal_dropout=0.):
    """Explicit frozen feature path, matching upstream single-horizon attention contract.

    Keep original three-slot window and time embeddings, but omit video loss and
    action-head computation. No hook, exception, or graph through the video DiT.
    Match the official caller's autocast scope for VAE and goal projection too;
    autocasting only the DiT changes the VAE features before the frozen backbone.
    """
    if model.n_future_frames != 1 or model.n_goal_frames != 1:
        raise ValueError('This recipe supports the released single-horizon LIBERO checkpoint only')
    if model.backbone.training:
        raise RuntimeError('Frozen backbone must stay in eval mode')
    images = [e['image'] for e in batch]
    if any(len(im) != model.window_frames for im in images):
        raise ValueError('Observation/future/goal image window does not match checkpoint')
    nh = model.num_history_frames
    first = model._encode_video_pass(images, nh)
    goal = model._encode_video_pass(images, nh + 1)[:, :, model.obs_latent_frames:]
    clean = torch.cat([first, goal], dim=2)
    goal_start = clean.shape[2] - 1
    # Current + goal cannot attend to future slot under the official 3-class mask.
    # Fixed sigma .5 in the ignored future slot avoids consuming training RNG.
    xt = clean.clone()
    xt[:, :, model.obs_latent_frames:goal_start] = (
        .5 * clean[:, :, model.obs_latent_frames:goal_start])
    timestep = torch.zeros(clean.shape[0], clean.shape[2], device=clean.device)
    timestep[:, model.obs_latent_frames:goal_start] = 500.
    text = model._maybe_stack_cached_text(batch)
    if text is None:
        raise RuntimeError('UMT5 cache missing; refusing to load an extra text encoder')
    with torch.autocast('cuda', dtype=torch.bfloat16), model._sdpa_ctx_factory():
        features, _ = model._run_dit(xt.to(model.backbone.transformer.dtype), timestep,
                                     text, n_goal_frames=1)
    obs_hidden, goal_hidden = model._slice_obs_goal(features, goal_start, clean.shape[2])
    if model.goal_time_proj is not None:
        offsets = np.stack([np.atleast_1d(e['goal_offset_rows']).astype(np.float32) for e in batch])
        goal_hidden = model._add_goal_time_diff(goal_hidden, torch.from_numpy(offsets).to(goal_hidden.device))
    if goal_dropout:
        keep = (torch.rand(len(batch), 1, device=goal_hidden.device) >= goal_dropout).to(goal_hidden.dtype)
        goal_hidden = goal_hidden * keep.unsqueeze(-1)
    cond = torch.cat([obs_hidden, goal_hidden], dim=1).detach()
    actions = torch.as_tensor(np.stack([e['action'] for e in batch]), device=cond.device,
                              dtype=cond.dtype)[:, -model.action_horizon:]
    # The backbone's feature hooks otherwise retain the last batch on GPU.
    model.backbone._intermediate_features.clear()
    return cond, actions


def action_loss(model, batch, train=False):
    cond, targets = extract_frozen_features(model, batch, model.goal_dropout_p if train else 0.)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model.action_model(cond, targets, None)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite flow-matching action loss')
    return loss


@torch.no_grad()
def verify_feature_parity(model, batch):
    """One-time integration check: compare to actual official forward's head input."""
    model.eval()
    captured = {}

    class Captured(Exception):
        pass

    def capture(_module, inputs):
        captured['cond'], captured['targets'] = inputs[0].detach(), inputs[1].detach()
        raise Captured()

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1729)  # upstream VAE uses latent_dist.sample(), not mode()
        hook = model.action_model.register_forward_pre_hook(capture)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(batch)
        except Captured:
            pass
        finally:
            hook.remove()
        torch.manual_seed(1729)
        actual, targets = extract_frozen_features(model, batch)
        torch.testing.assert_close(actual, captured['cond'], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(targets, captured['targets'], atol=0, rtol=0)
    return dict(condition_shape=list(actual.shape), targets_shape=list(targets.shape), passed=True)


@torch.no_grad()
def validate(model, ds, a):
    model.eval()
    sums = {'clean': 0., 'corrupt': 0.}
    # Isolate validation RNG; identical samples, goal offsets and FM noise every evaluation.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for index in range(len(ds)):
            sample = ds[index]
            altered = dict(sample)
            altered['image'] = photometric_window(sample['image'], a.val_brightness,
                a.val_saturation, tuple(a.val_color_gain))
            for key, item in [('clean', sample), ('corrupt', altered)]:
                torch.manual_seed(stable_seed(a.seed, 'val_noise', index))
                sums[key] += action_loss(model, [item]).item()
    return {k + '_fm_loss': v / len(ds) for k, v in sums.items()}


def save_checkpoint(model, out, step, label, metadata):
    state = model.state_dict()
    keys = deployment_keys(list(state), tuple(model._save_include_prefixes),
                           tuple(model._reconstructable_prefixes))
    # Export video DiT too: the robot-trained backbone differs from raw SkyReels.
    payload = {k: state[k].detach().cpu() for k in keys}
    dest = out / 'checkpoints' / f'{label}.pt'
    atomic_torch(dest, payload)
    atomic_json(dest.with_suffix('.json'), dict(step=step, **metadata))
    del payload, state
    gc.collect()
    return dest


def configure(a):
    raw, stats = read_mode_config(Path(a.ckpt))
    cfg = OmegaConf.create(raw)
    wm = cfg.framework.world_model
    if a.base_model:
        wm.base_wm = str(Path(a.base_model).resolve())
    wm.compile_dit = False
    wm.compile_vae = False
    wm.skip_text_encoder = True
    wm.resident_text_table = True
    wm.truncate_at_extract = False
    dc = cfg.datasets.vla_data
    if a.data_root:
        dc.data_root_dir = str(Path(a.data_root).resolve())
    else:
        dc.data_root_dir = str(Path(dc.data_root_dir).resolve())
    dc.data_mix = a.data_mix
    dc.camera_concat = wm.camera_concat
    dc.include_state = False
    dc.image_aug = False
    dc.eval_holdout_episodes = 0
    for key in ('holdout_episode_fold', 'max_episodes_per_dataset', 'advantage_label_id'):
        if dc.get(key) is not None and dc.get(key) not in ('', 0):
            raise ValueError(f'Unsupported upstream filtering option {key}; start from day1_base')
    if 'vae_cache' in dc:
        dc.vae_cache.enabled = False
    dc.t5_cache = dict(enabled=True, attach_embeds=True, model_id=T5_ID, max_length=512)
    dc.action_horizon = cfg.framework.action_model.action_horizon
    dc.num_history_frames = wm.num_history_frames
    dc.per_device_batch_size = a.batch_size
    cfg.framework.action_model.repeated_diffusion_steps = 1
    cfg.trainer.sdpa_backend = 'auto'
    cfg.trainer.ema.enabled = False
    cfg.trainer.pretrained_checkpoint = None
    cfg.run_id = Path(a.out).name
    cfg.output_dir = str(Path(a.out).resolve())
    return cfg, stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-root')
    p.add_argument('--base-model')
    p.add_argument('--data-mix', default='libero_goal')
    p.add_argument('--episode-split', help='Audited explicit split JSON; overrides random episode splitting')
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--val-fraction', type=float, default=.1)
    p.add_argument('--val-samples', type=int, default=32)
    p.add_argument('--val-stride', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--train-scope', choices=['all', 'crossattn_decoder'], default='all')
    p.add_argument('--aug-prob', type=float, default=.5)
    p.add_argument('--brightness', nargs=2, type=float, default=[.6, 1.4])
    p.add_argument('--saturation', nargs=2, type=float, default=[.6, 1.4])
    p.add_argument('--color-gain', nargs=2, type=float, default=[.9, 1.1])
    p.add_argument('--val-brightness', type=float, default=.7)
    p.add_argument('--val-saturation', type=float, default=.6)
    p.add_argument('--val-color-gain', nargs=3, type=float, default=[.9, 1., 1.1])
    p.add_argument('--allow-different-commit', action='store_true')
    a = p.parse_args()
    for key in ('max_steps', 'batch_size', 'grad_accum', 'eval_every', 'save_every', 'val_samples', 'val_stride'):
        if getattr(a, key) <= 0:
            p.error(f'{key} must be positive')
    if not 0 <= a.aug_prob <= 1 or a.workers < 0 or a.lr <= 0 or a.warmup_steps < 0:
        p.error('Invalid augmentation probability, workers, LR or warmup')
    for key in ('brightness', 'saturation', 'color_gain'):
        lo, hi = getattr(a, key)
        if not 0 < lo <= hi:
            p.error(f'Invalid {key} range')
    if not 0 < a.val_fraction < 1:
        p.error('val-fraction must be between 0 and 1')
    if a.val_brightness <= 0 or a.val_saturation < 0 or min(a.val_color_gain) <= 0:
        p.error('Invalid validation corruption')
    return a


def main():
    a = parse_args()
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new --out directory; existing experiments will not be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This script is single-GPU; use python, not multi-process torchrun')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN and not a.allow_different_commit:
        raise RuntimeError(f'Expected upstream {PIN}, got {commit}')
    Accelerator()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    cfg, stats = configure(a)
    datasets, train_entries, val_entries, manifest = build_splits(cfg, a, stats)
    if len(train_entries) < a.batch_size:
        raise ValueError('Training split is smaller than batch-size')
    split_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    atomic_json(out / 'split_manifest.json', manifest)
    atomic_json(out / 'train_args.json', vars(a))
    atomic_json(out / 'dataset_statistics.json', stats)  # preserve pretrained action scale exactly
    cfg.finetune_recipe = dict(name='frozen_wam_action_sft', seed=a.seed,
        split_sha256=split_hash, upstream_commit=commit, train_args=vars(a))
    OmegaConf.save(cfg, out / 'config.yaml')
    model = GlanceWAM(cfg)
    weights = torch.load(a.ckpt, map_location='cpu', weights_only=True)
    result = model.load_state_dict(weights, strict=False)
    recon = tuple(model._reconstructable_prefixes)
    if any(not k.startswith(recon) for k in result.missing_keys + result.unexpected_keys):
        raise RuntimeError(f'Checkpoint mismatch: {result}')
    del weights
    model.norm_stats = stats
    model.requires_grad_(False).to(device='cuda', dtype=torch.bfloat16).eval()
    model.action_model.float().requires_grad_(True)
    if a.train_scope == 'crossattn_decoder':
        model.action_model.requires_grad_(False)
        count = 0
        for block in model.action_model.model.transformer_blocks:
            # GlanceWAM alternates cross/self-attention blocks; both are named attn1.
            if block.cross_attention_dim is not None:
                block.attn1.requires_grad_(True)
                count += 1
        if not count:
            raise RuntimeError('No cross-attention blocks found; inspect the action-head version')
        model.action_model.action_decoder.requires_grad_(True)
    # The published stateless LIBERO head owns a state encoder but never uses it.
    if model.action_model.state_encoder is not None:
        model.action_model.state_encoder.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    if not trainable or any(not n.startswith('action_model.') for n, _ in trainable):
        raise RuntimeError('Unexpected trainable parameter set')
    frozen_before = digest(frozen)
    action_before = digest(model.action_model.named_parameters())
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=a.lr,
                                 weight_decay=a.weight_decay, foreach=False)
    val_ds = EpisodeFrames(datasets, val_entries, a.seed, epoch=-1)
    epoch, batch_position, iterator = 0, 0, None
    train_ds = None

    def next_batch():
        nonlocal epoch, batch_position, iterator, train_ds
        while True:
            if iterator is None:
                train_ds = EpisodeFrames(datasets, train_entries, a.seed, epoch)
                generator = torch.Generator().manual_seed(stable_seed(a.seed, 'order', epoch))
                loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, generator=generator,
                    num_workers=a.workers, collate_fn=collate_fn, persistent_workers=False, drop_last=True)
                iterator = iter(loader)
                batch_position = 0
            try:
                batch = next(iterator)
                batch_position += 1
                return batch
            except StopIteration:
                epoch += 1
                iterator = None

    metadata = dict(upstream_commit=commit, split_sha256=split_hash,
        train_scope=a.train_scope,
        base_checkpoint=str(Path(a.ckpt).resolve()), trainable_parameters=sum(p.numel() for _, p in trainable),
        frozen_sha256=frozen_before, seed=a.seed, validation_scope='hindsight-goal flow matching loss')
    log(out, dict(event='start', total_parameters=sum(p.numel() for p in model.parameters()),
                  train_windows=len(train_entries), val_windows=len(val_entries), **metadata))
    parity = verify_feature_parity(model, [val_ds[0]])
    log(out, dict(event='official_feature_parity', **parity))
    baseline = validate(model, val_ds, a)
    log(out, dict(event='validation', step=0, **baseline, **mem()))
    best = .5 * (baseline['clean_fm_loss'] + baseline['corrupt_fm_loss'])
    # Best starts at the unmodified base, so a failed adaptation does not silently
    # replace the baseline with a worse checkpoint. Last always contains final updates.
    save_checkpoint(model, out, 0, 'best', dict(selection_score=best, **metadata))
    torch.cuda.reset_peak_memory_stats()
    first_check = True
    for step in range(1, a.max_steps + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.eval()
        model.action_model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(a.grad_accum):
            batch = next_batch()
            batch = [augment_sample(e, np.random.default_rng(stable_seed(a.seed, step, micro, j)),
                a.aug_prob, a.brightness, a.saturation, a.color_gain) for j, e in enumerate(batch)]
            loss = action_loss(model, batch, train=True)
            (loss / a.grad_accum).backward()
            losses.append(loss.item())
            del batch, loss
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1., error_if_nonfinite=True)
        if first_check:
            if norm.item() == 0 or any(p.grad is not None for _, p in frozen):
                raise RuntimeError('Zero action gradient or gradient leakage into frozen parameters')
        warm = min(1., step / max(1, a.warmup_steps))
        progress = max(0., (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps))
        lr = a.lr * warm * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        log(out, dict(event='train', step=step, epoch=epoch, batch_position=batch_position,
            action_fm_loss=float(np.mean(losses)), grad_norm=norm.item(), lr=lr,
            seconds=time.perf_counter() - start, **mem()))
        if first_check:
            if digest(frozen) != frozen_before or digest(model.action_model.named_parameters()) == action_before:
                raise RuntimeError('Freeze/update integrity check failed')
            log(out, dict(event='integrity', step=step, frozen_unchanged=True, action_updated=True))
            first_check = False
        if step % a.eval_every == 0 or step == a.max_steps:
            scores = validate(model, val_ds, a)
            log(out, dict(event='validation', step=step, **scores, **mem()))
            score = .5 * (scores['clean_fm_loss'] + scores['corrupt_fm_loss'])
            if score < best:
                best = score
                save_checkpoint(model, out, step, 'best', dict(selection_score=score, **metadata))
        if step % a.save_every == 0:
            save_checkpoint(model, out, step, f'step_{step:06d}', metadata)
    if digest(frozen) != frozen_before:
        raise RuntimeError('Frozen parameters changed during training')
    save_checkpoint(model, out, a.max_steps, 'last', metadata)
    atomic_json(out / 'summary.json', dict(completed_steps=a.max_steps, best_selection_score=best,
        frozen_unchanged=True, **metadata, **mem()))


if __name__ == '__main__':
    main()
