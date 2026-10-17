"""Phase indexing and deterministic sampling; no images, model or torch import."""
import numpy as np


def gripper_phases(gripper):
    g = np.asarray(gripper, dtype=np.float32).reshape(-1)
    if not len(g) or not np.isfinite(g).all() or not np.all((g == 0) | (g == 1)):
        raise ValueError('Expected raw demonstration gripper labels: 1=open, 0=closed')
    close = np.flatnonzero(np.diff(g) == -1) + 1
    release = np.flatnonzero(np.diff(g) == 1) + 1
    # This audited recipe is for the ten single-grasp bowl demonstrations.
    if len(close) != 1 or len(release) != 1 or close[0] >= release[0]:
        raise ValueError('Expected exactly one close followed by one release per demonstration')
    c, r = int(close[0]), int(release[0])
    frames = np.arange(len(g))
    return dict(close=c, release=r, length=len(g),
                grasp=(frames >= c-8) & (frames <= c+24),
                place=(frames >= r-24) & (frames <= r+8))


def build_phase_index(datasets, train_entries, val_entries):
    """Read unnormalized Parquet gripper labels through the official dataset API."""
    cache, episodes = {}, []
    selected = sorted(set(train_entries) | set(val_entries))
    for di, index in selected:
        ds = datasets[di]
        episode, frame = map(int, ds.all_steps[index])
        key = (di, episode)
        if key not in cache:
            df = ds.get_trajectory_data(episode)
            if not np.array_equal(np.asarray(df['frame_index']), np.arange(len(df))):
                raise ValueError('Parquet frame order is not contiguous within an episode')
            field = ds.lerobot_modality_meta.action['gripper']
            raw = np.stack(df[field.original_key or 'gripper'])
            if raw.ndim != 2 or field.end-field.start != 1:
                raise ValueError('Unexpected raw gripper column/slice')
            phase = gripper_phases(raw[:, field.start:field.end])
            cache[key] = phase
            episodes.append(dict(dataset=ds.dataset_name, episode=episode,
                close=phase['close'], release=phase['release'], frames=phase['length']))
        if frame < 0 or frame >= cache[key]['length']:
            raise ValueError('Official dataset frame index outside its own episode')
    tags = {}
    for di, index in selected:
        episode, frame = map(int, datasets[di].all_steps[index])
        phase = cache[di, episode]
        tags[di, index] = [name for name in ('grasp', 'place') if phase[name][frame]]
    pools = dict(uniform=list(range(len(train_entries))), grasp=[], place=[])
    for position, entry in enumerate(train_entries):
        for name in tags[entry]:
            pools[name].append(position)
    if not pools['grasp'] or not pools['place']:
        raise ValueError('Empty training phase pool')
    for name in ('grasp', 'place'):
        if not any(name in tags[e] for e in val_entries):
            raise ValueError(f'Fixed validation windows contain no {name} samples')
    manifest = dict(schema_version=1, episodes=episodes,
        definitions=dict(grasp=[-8, 24], place=[-24, 8], bounds='inclusive, episode clipped'),
        overlaps='A window may have both validation tags; each sampled training window gets one unchanged FM loss',
        train_pool_counts={k: len(v) for k,v in pools.items()},
        train_pool_entries={k: [dict(dataset=datasets[train_entries[j][0]].dataset_name,
            episode=int(datasets[train_entries[j][0]].all_steps[train_entries[j][1]][0]),
            frame=int(datasets[train_entries[j][0]].all_steps[train_entries[j][1]][1])) for j in v]
            for k,v in pools.items()},
        validation_tags=[tags[e] for e in val_entries])
    return tags, pools, manifest


def draw_schedule(pools, mode, count, seed):
    if mode not in ('uniform', 'phase') or count <= 0 or count % 4:
        raise ValueError('Use uniform/phase sampling and a positive multiple-of-four draw count')
    rng = np.random.default_rng(seed)
    indices, sources = [], []
    for _ in range(count // 4):
        names = ['uniform'] * 4 if mode == 'uniform' else rng.permutation(
            ['uniform', 'uniform', 'grasp', 'place']).tolist()
        for name in names:
            if not pools[name]:
                raise ValueError(f'Empty sampling pool: {name}')
            indices.append(int(rng.choice(pools[name])))
            sources.append(name)
    return indices, sources


def phase_selection(scores, baseline, tolerance=.05):
    """Normalized macro phase score, subject to a normal-image non-regression gate."""
    values = [scores['clean_fm_loss'], baseline['clean_fm_loss']]
    for name in ('grasp', 'place'):
        values += [scores['clean_'+name+'_fm_loss'], baseline['clean_'+name+'_fm_loss']]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Selection losses must be finite and positive')
    score = float(np.mean([scores['clean_'+name+'_fm_loss'] / baseline['clean_'+name+'_fm_loss']
                           for name in ('grasp', 'place')]))
    eligible = scores['clean_fm_loss'] <= baseline['clean_fm_loss'] * (1+tolerance)
    return score, bool(eligible)
