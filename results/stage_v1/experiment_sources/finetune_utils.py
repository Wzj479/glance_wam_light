"""Dependency-light, testable data split / photometric augmentation helpers."""
import hashlib
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def stable_seed(*parts):
    text = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:4], 'little')


def split_episodes(ids, val_fraction, seed):
    ids = sorted(set(map(int, ids)))
    if len(ids) < 2 or not 0 < val_fraction < 1:
        raise ValueError('Need >=2 episodes per dataset and 0 < val_fraction < 1')
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    n_val = min(len(ids) - 1, max(1, int(round(len(ids) * val_fraction))))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    assert not set(train).intersection(val)
    return train, val


def split_step_indices(all_steps, train_ids, val_ids, val_stride):
    """All future/goal frames remain within their original episode in upstream."""
    train_ids, val_ids = set(train_ids), set(val_ids)
    if train_ids & val_ids:
        raise ValueError('Train/validation episodes overlap')
    train, val = [], []
    for i, (episode, frame) in enumerate(all_steps):
        if int(episode) in train_ids:
            train.append(i)
        elif int(episode) in val_ids and int(frame) % val_stride == 0:
            val.append(i)
    if not train or not val:
        raise ValueError('Empty train/validation frame split; inspect episode lengths/stride')
    return train, val


def load_episode_partition(path, dataset_name, eligible):
    """Use audited episode IDs verbatim; reject accidental leakage or missing data."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Expected explicit episode split schema_version=1')
    item = manifest['datasets'][dataset_name]
    train, val = item['train_episode_ids'], item['val_episode_ids']
    if not train or not val:
        raise ValueError('Explicit split must have nonempty train and validation episodes')
    if any(type(e) is not int or e < 0 for e in train + val):
        raise ValueError('Episode IDs must be nonnegative integers')
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError('Duplicate episode IDs in explicit split')
    if set(train) & set(val):
        raise ValueError('Explicit train and validation episodes overlap')
    missing = (set(train) | set(val)) - set(eligible)
    if missing:
        raise ValueError(f'Audited episodes absent from official loader: {sorted(missing)}')
    return sorted(train), sorted(val)


def checkpoint_action_metadata(metadata, action_stats):
    """Align LIBERO loader min/max to the checkpoint's deployment action domain."""
    keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    lo = np.asarray(action_stats['min'], dtype=np.float32)
    hi = np.asarray(action_stats['max'], dtype=np.float32)
    mask = np.asarray(action_stats.get('mask'), dtype=bool)
    if lo.shape != (7,) or hi.shape != (7,) or mask.shape != (7,):
        raise ValueError('Expected seven-dimensional checkpoint action statistics with mask')
    if not np.array_equal(mask, [True] * 6 + [False]):
        raise ValueError('LIBERO recipe requires first-six min/max normalization and raw gripper')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi[:6] <= lo[:6]):
        raise ValueError('Invalid checkpoint action bounds')
    result = copy.deepcopy(metadata)
    for i, key in enumerate(keys):
        field = result.statistics.action[key]
        if np.asarray(field.min).shape != (1,) or np.asarray(field.max).shape != (1,):
            raise ValueError(f'Expected scalar LIBERO action field: {key}')
        field.min = lo[i:i + 1].copy()
        field.max = hi[i:i + 1].copy()
    return result


def photometric_window(images, brightness=1.0, saturation=1.0, gains=(1., 1., 1.)):
    """Same factors for every camera/time in a window; no label/geometry mutation."""
    if brightness <= 0 or saturation < 0 or min(gains) <= 0:
        raise ValueError('Invalid photometric factor')
    result = []
    for image in images:
        if isinstance(image, Image.Image):
            im = image.convert('RGB')
        else:
            array = np.asarray(image)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(f'Expected uint8 HWC RGB, got {array.shape}, {array.dtype}')
            im = Image.fromarray(array)
        im = ImageEnhance.Brightness(im).enhance(brightness)
        im = ImageEnhance.Color(im).enhance(saturation)
        array = np.asarray(im, dtype=np.float32) * np.asarray(gains, dtype=np.float32)
        result.append(Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)))
    return result


def augment_sample(sample, rng, probability, brightness, saturation, color_gain):
    if rng.random() >= probability:
        return sample
    out = dict(sample)
    out['image'] = photometric_window(
        sample['image'], rng.uniform(*brightness), rng.uniform(*saturation),
        tuple(rng.uniform(*color_gain, size=3)),
    )
    return out


def deployment_keys(state_keys, include_prefixes, reconstructable_prefixes):
    """Keep official trained tensors and reject silently dropped new modules."""
    kept = [k for k in state_keys if k.startswith(include_prefixes)]
    unknown = [k for k in state_keys if k not in kept and not k.startswith(reconstructable_prefixes)]
    if unknown:
        raise ValueError(f'Unknown checkpoint prefixes: {unknown[:10]}')
    if not any(k.startswith('backbone.transformer.') for k in kept):
        raise ValueError('Export is missing the robot-trained video backbone')
    if not any(k.startswith('action_model.') for k in kept):
        raise ValueError('Export is missing the action model')
    return kept
