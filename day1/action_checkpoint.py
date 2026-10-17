"""Action-only exports bound to immutable base weights and their lexical config view."""
import hashlib
import json
import os
from pathlib import Path
import torch

FORMAT = 'glancewam_action_v1'


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def base_fingerprint(path):
    # Never resolve a weights symlink: sidecars belong to its caller's view.
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    root = path.parents[1]
    return dict(path=str(path), sha256=file_sha256(path),
                config_sha256=file_sha256(root/'config.yaml'),
                statistics_sha256=file_sha256(root/'dataset_statistics.json'))


def save_action_checkpoint(action_model, destination, base, metadata):
    destination = Path(destination)
    if not destination.name.endswith('.action.pt'):
        raise ValueError('Action-only checkpoints must use the .action.pt suffix')
    state = {k: v.detach().cpu().clone() for k,v in action_model.state_dict().items()}
    if not state or any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Empty or nonfinite action state')
    payload = dict(checkpoint_format=FORMAT, base=base, action_state_dict=state, metadata=metadata)
    temporary = destination.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)
    sidecar = {**metadata, 'checkpoint_format':FORMAT,
               'checkpoint_sha256':file_sha256(destination), 'base':base}
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(sidecar,indent=2),encoding='utf-8')
    temporary.replace(destination.with_suffix('.json'))
    return destination


def read_action_checkpoint(path, base_override=None):
    path = Path(path)
    sidecar = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if sidecar.get('checkpoint_format') != FORMAT or file_sha256(path) != sidecar['checkpoint_sha256']:
        raise ValueError('Invalid action checkpoint sidecar or checksum')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('checkpoint_format') != FORMAT or payload['base'] != sidecar['base']:
        raise ValueError('Action checkpoint/base metadata mismatch')
    actual = base_fingerprint(base_override or payload['base']['path'])
    for key in ('sha256','config_sha256','statistics_sha256'):
        if actual[key] != payload['base'][key]:
            raise ValueError(f'Wrong base checkpoint/view: {key}')
    if not payload['action_state_dict'] or any(not torch.isfinite(t).all()
            for t in payload['action_state_dict'].values()):
        raise ValueError('Nonfinite or empty action checkpoint')
    return actual['path'], payload


def apply_action_checkpoint(action_model, payload):
    target = action_model.state_dict()
    source = payload['action_state_dict']
    if set(target) != set(source) or any(target[k].shape != source[k].shape for k in target):
        raise ValueError('Action architecture/key/shape mismatch')
    action_model.load_state_dict(source, strict=True)


def load_policy_checkpoint(path, framework):
    """Shared closed-loop loader; full legacy checkpoints remain supported."""
    if str(path).endswith('.action.pt'):
        base, payload = read_action_checkpoint(path)
        model = framework.from_pretrained(base)
        apply_action_checkpoint(model.action_model, payload)
        return model
    return framework.from_pretrained(path)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='CPU integrity check of an action-only checkpoint; no model load')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--base-override', help='Relocated lexical base view; all content hashes must still match')
    a = p.parse_args()
    base, payload = read_action_checkpoint(a.checkpoint, a.base_override)
    print(json.dumps(dict(valid=True, base=base, metadata=payload['metadata'],
        action_tensor_count=len(payload['action_state_dict']),
        action_state_bytes=sum(t.numel()*t.element_size() for t in payload['action_state_dict'].values())),indent=2))
