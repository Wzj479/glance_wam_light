"""Download the pinned official LIBERO files without loading models or pickle."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--host', default='huggingface.co')
    args = p.parse_args()
    manifest = json.loads(Path(__file__).with_name('demo_download_manifest.json').read_text())
    root = Path(args.root)
    records = []
    for item in manifest['files']:
        rel = item['path'][len(manifest['prefix']) + 1:]
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('lfs', {}).get('oid')
        if not (dest.exists() and dest.stat().st_size == item['size'] and
                (expected is None or digest(dest) == expected)):
            url = (f"https://{args.host}/datasets/LinhanWang/GlanceWAM/resolve/"
                   f"{manifest['revision']}/{item['path']}")
            for attempt in range(4):
                try:
                    part = dest.with_suffix(dest.suffix + '.part')
                    print('DOWNLOAD', rel, item['size'], flush=True)
                    with urllib.request.urlopen(url, timeout=40) as response, part.open('wb') as f:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            f.write(block)
                    assert part.stat().st_size == item['size'], 'Size mismatch'
                    assert expected is None or digest(part) == expected, 'LFS SHA256 mismatch'
                    part.replace(dest)
                    break
                except Exception as error:
                    print('RETRY', attempt, type(error).__name__, str(error), flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2)
        actual = digest(dest)
        records.append(dict(path=rel, size=dest.stat().st_size, sha256=actual,
                            official_lfs_sha256=expected))
        print('VERIFIED', rel, actual, flush=True)
    (root / 'download_receipt.json').write_text(json.dumps(dict(
        revision=manifest['revision'], host=args.host, files=records), indent=2))


if __name__ == '__main__':
    main()
