"""Fixed, paired closed-loop evaluation contracts. No simulator dependency."""
import hashlib
import json
import math

CONDITIONS = {
    'clean': dict(brightness=1., saturation=1., gains=[1., 1., 1.]),
    'dark': dict(brightness=.7, saturation=1., gains=[1., 1., 1.]),
    'color': dict(brightness=1., saturation=1., gains=[.9, 1., 1.1]),
}
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)


def validate_protocol(p):
    if p.get('schema_version') != 1 or p.get('suite') not in MAX_STEPS:
        raise ValueError('Unsupported protocol version or suite')
    for key in ('task_ids', 'episode_ids'):
        values = p.get(key, [])
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f'Invalid {key}')
    if type(p.get('seed')) is not int or p['seed'] < 0:
        raise ValueError('Invalid seed')
    if not p.get('conditions'):
        raise ValueError('Missing conditions')
    for name, c in p['conditions'].items():
        b, s, gains = c['brightness'], c['saturation'], c['gains']
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError('Condition names must be identifiers')
        if len(gains) != 3 or not all(math.isfinite(v) for v in [b, s, *gains]):
            raise ValueError('Invalid corruption factors')
        if b <= 0 or s < 0 or min(gains) <= 0:
            raise ValueError('Invalid corruption factors')
    return p


def protocol_hash(p):
    validate_protocol(p)
    return hashlib.sha256(json.dumps(p, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def paired_ids(p):
    return {(task, episode) for task in p['task_ids'] for episode in p['episode_ids']}


def validate_episode_results(records, p):
    expected = paired_ids(p)
    observed = [(r['task_id'], r['episode_id']) for r in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Incomplete, duplicated or unexpected evaluation episodes')
    if any(type(r['success']) is not bool for r in records):
        raise ValueError('Success must be a boolean')


def paired_delta(base, candidate):
    b = {(r['task_id'], r['episode_id']): r['success'] for r in base}
    c = {(r['task_id'], r['episode_id']): r['success'] for r in candidate}
    if len(b) != len(base) or len(c) != len(candidate) or b.keys() != c.keys() or not b:
        raise ValueError('Pairing mismatch')
    gains = sum(not b[k] and c[k] for k in b)
    losses = sum(b[k] and not c[k] for k in b)
    n = len(b)
    return dict(episodes=n, recovered=gains, regressed=losses,
                success_delta_pp=100 * (gains - losses) / n)
