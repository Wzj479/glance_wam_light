"""Package the assets that cannot be rebuilt from any public source.

GlanceWAM code, LIBERO code and assets, the base checkpoint and the LIBERO dataset
all come from public repositories or Hugging Face. Two things do not: the
stage_v1 fine-tuned action checkpoints, which are the product of this project's own
sampling run, and the archived per-episode action arrays, which only that checkpoint
can regenerate under the archived protocol.

The action arrays are what the state33 diagnostic's zero-intervention control is
gated on, so losing them does not degrade the release, it invalidates it. This script
collects the minimum irreproducible set, records a SHA256 manifest, and writes a
tarball that can be verified after transfer.

last.action.pt is excluded deliberately: only best is evaluated, and the extra
595MB buys nothing. base_seed42 needs no packaging because it is downloadable.
"""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

RUN_DIRS = ('P_seed42', 'C_seed42')
RUN_METADATA = ('config.yaml', 'dataset_statistics.json', 'parameter_manifest.json',
                'phase_manifest.json', 'split_manifest.json', 'train_args.json', 'summary.json')
CHECKPOINT_FILES = ('best.action.pt', 'best.action.json')
EXCLUDED = {'last.action.pt': 'only best is evaluated; the extra 595MB is not used',
            'last.action.json': 'same as last.action.pt'}


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def collect(root):
    """Relative paths of the irreproducible set, verified to exist and be non-empty."""
    root = Path(root)
    wanted = []
    for run in RUN_DIRS:
        for name in CHECKPOINT_FILES + RUN_METADATA:
            wanted.append(f'results/stage_v1/{run}/checkpoints/{name}'
                          if name in CHECKPOINT_FILES else f'results/stage_v1/{run}/{name}')
    wanted += [str(p.relative_to(root)) for p in sorted(root.glob('results/stage_v1/development/**/*_actions.npy'))]
    missing = [name for name in wanted if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f'missing irreproducible assets: {missing}')
    empty = [name for name in wanted if (root / name).stat().st_size == 0]
    if empty:
        raise ValueError(f'zero-byte assets would defeat the purpose: {empty}')
    return wanted


def build(root, out_tar):
    files = collect(root)
    manifest = {name: dict(bytes=(Path(root) / name).stat().st_size, sha256=sha256(Path(root) / name))
                for name in files}
    manifest_path = Path(root) / 'irreproducible_SHA256.json'
    manifest_path.write_text(json.dumps(dict(
        note='Assets that no public source can rebuild; verify after transfer.',
        excluded=EXCLUDED,
        files=manifest), indent=2), encoding='utf-8')
    total = sum(entry['bytes'] for entry in manifest.values())
    # The .pt files are already zip-compressed, so compression costs CPU for nothing.
    with tarfile.open(out_tar, 'w') as archive:
        for name in files:
            archive.add(Path(root) / name, arcname=name)
        archive.add(manifest_path, arcname=manifest_path.name)
    print(json.dumps(dict(files=len(files), bytes=total,
                          tar=str(out_tar), tar_bytes=Path(out_tar).stat().st_size), indent=2))
    return manifest


def verify(tar):
    with tarfile.open(tar) as archive:
        payload = json.loads(archive.extractfile('irreproducible_SHA256.json').read())
        checked = failed = 0
        for name, expected in payload['files'].items():
            member = archive.extractfile(name)
            if member is None:
                print(f'MISSING {name}')
                failed += 1
                continue
            digest = hashlib.sha256()
            for block in iter(lambda: member.read(1 << 20), b''):
                digest.update(block)
            if digest.hexdigest() != expected['sha256']:
                print(f'CORRUPT {name}')
                failed += 1
            checked += 1
    print(json.dumps(dict(checked=checked, failed=failed,
                          verdict='OK' if not failed else 'FAILED'), indent=2))
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='GlanceWAM directory')
    parser.add_argument('--out', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        raise SystemExit(0 if verify(args.out) else 1)
    build(args.root, args.out)


if __name__ == '__main__':
    main()