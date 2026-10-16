"""Create a separate low-texture-memory LIBERO copy; preserve physical assets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='LIBERO repo root')
    p.add_argument('--out', required=True, help='New diagnostic repo root')
    p.add_argument('--texture-size', type=int, default=1024)
    a = p.parse_args()
    source, dest = Path(a.source).resolve(), Path(a.out).resolve()
    if dest == source or source in dest.parents or a.texture_size < 64:
        p.error('Use separate source/destination paths and size >=64')
    base = source / 'libero/libero'
    if not (base / 'assets').is_dir():
        p.error('Source does not contain LIBERO assets')
    excluded = ('.git', '.venv', '__pycache__', 'datasets', 'results', 'checkpoints')
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns(*excluded))
    changed = []
    for original in sorted((base / 'assets').rglob('*')):
        if not original.is_file() or original.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        target = dest / original.relative_to(source)
        with Image.open(original) as image:
            before = image.size
            if max(before) <= a.texture_size:
                continue
            image.thumbnail((a.texture_size, a.texture_size), Image.Resampling.LANCZOS)
            image.save(target)
            changed.append(dict(path=original.relative_to(source).as_posix(),
                original_size=list(before), diagnostic_size=list(image.size),
                original_sha256=sha(original), diagnostic_sha256=sha(target)))
    changed_paths = {item['path'] for item in changed}
    checked = 0
    for original in source.rglob('*'):
        if not original.is_file() or any(x in original.relative_to(source).parts for x in excluded):
            continue
        rel = original.relative_to(source)
        if rel.as_posix() not in changed_paths:
            assert sha(original) == sha(dest / rel), str(rel)
            checked += 1
    profile = dict(mode='cpu_diagnostic_textures', texture_max_size=a.texture_size,
                   source=str(source), destination=str(dest), unchanged_files_checked=checked,
                   physics_files_unchanged=True, changed_textures=changed,
                   scope='Control/replay diagnostics; not official visual benchmark scores')
    (dest / 'cpu_asset_profile.json').write_text(json.dumps(profile, indent=2))
    import yaml
    config = dest / 'cpu_config'
    config.mkdir()
    root = dest / 'libero/libero'
    (config / 'config.yaml').write_text(yaml.safe_dump(dict(
        benchmark_root=str(root), bddl_files=str(root / 'bddl_files'),
        init_states=str(root / 'init_files'), assets=str(root / 'assets'), datasets=str(dest))))
    print(json.dumps(dict(changed_textures=len(changed), unchanged_files_checked=checked,
                          config=str(config), profile=str(dest / 'cpu_asset_profile.json'))))


if __name__ == '__main__':
    main()
