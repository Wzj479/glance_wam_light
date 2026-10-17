"""Run from the pinned GlanceWAM root. Make a separate day-one checkpoint view."""
import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', default='glancewam_bundle')
    p.add_argument('--base-model', default='models/skyreels-base')
    p.add_argument('--output', default='results/day1_base')
    p.add_argument('--data-mix', default='libero_goal',
                   help='Use only the downloaded suite for the first GPU validation')
    a = p.parse_args()
    bundle = Path(a.bundle).resolve()
    source = bundle / 'checkpoints/glancewam_libero'
    ckpt = source / 'checkpoints/steps_15000_pytorch_model_ema.pt'
    assert ckpt.is_file(), ckpt
    base = Path(a.base_model).resolve()
    assert (base / 'transformer/config.json').is_file(), base
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(source / 'config.yaml')
    cfg.framework.world_model.base_wm = str(base)
    cfg.framework.world_model.compile_dit = False
    cfg.framework.world_model.compile_vae = False
    cfg.framework.world_model.skip_text_encoder = True
    cfg.framework.world_model.resident_text_table = True
    cfg.datasets.vla_data.data_root_dir = str(bundle / 'datasets')
    cfg.datasets.vla_data.data_mix = a.data_mix
    cfg.datasets.vla_data.camera_concat = cfg.framework.world_model.camera_concat
    cfg.datasets.vla_data.include_state = False
    cfg.datasets.vla_data.per_device_batch_size = 1
    cfg.datasets.vla_data.num_workers = 1
    # Cache keys must remain the original HF ID, not the local model directory.
    cfg.datasets.vla_data.t5_cache.enabled = True
    cfg.datasets.vla_data.t5_cache.attach_embeds = True
    cfg.datasets.vla_data.t5_cache.model_id = 'Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers'
    cfg.datasets.vla_data.t5_cache.max_length = 512
    cfg.framework.action_model.repeated_diffusion_steps = 1
    cfg.trainer.sdpa_backend = 'auto'
    cfg.trainer.ema.enabled = False
    cfg.trainer.pretrained_checkpoint = None
    OmegaConf.save(cfg, out / 'config.yaml')
    shutil.copyfile(source / 'dataset_statistics.json', out / 'dataset_statistics.json')
    target = out / 'checkpoints' / ckpt.name
    target.parent.mkdir(exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == ckpt
    elif target.exists():
        raise FileExistsError(f'Refusing to replace {target}')
    else:
        target.symlink_to(ckpt)
    print(json.dumps({'checkpoint': str(target), 'config': str(out / 'config.yaml')}, indent=2))


if __name__ == '__main__':
    main()
