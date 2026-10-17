"""Download pinned GPU validation inputs; verify every available LFS SHA256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/root/autodl-tmp/wam_glance_project/GlanceWAM')
ENDPOINT = 'https://hf-mirror.com'
DATA_REV = '0b9208dff852914f94be879f702cd40118255bce'

def api(path):
    for attempt in range(4):
        try:
            return json.loads(subprocess.check_output(['curl', '-fsSL', '--connect-timeout', '10',
                '--max-time', '45', ENDPOINT+path], text=True))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)

base_id = 'Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers'
base_info = api('/api/models/'+base_id)
base_rev = base_info['sha']
jobs = []
base_files = api('/api/models/'+base_id+'/tree/'+base_rev+'?recursive=true&expand=false')
for item in base_files:
    if item.get('type') == 'file' and (item['path'].startswith(('transformer/', 'vae/', 'scheduler/')) or item['path'] == 'model_index.json'):
        jobs.append((base_id, base_rev, item, ROOT/'models/skyreels-base'/item['path'], False))
data_id = 'LinhanWang/GlanceWAM'
for prefix in ['checkpoints/glancewam_libero', 'datasets/libero_goal_no_noops_1.0.0_lerobot']:
    items = api('/api/datasets/'+data_id+'/tree/'+DATA_REV+'/'+prefix+'?recursive=true&expand=false')
    assert isinstance(items, list)
    for item in items:
        if item.get('type') == 'file':
            jobs.append((data_id, DATA_REV, item, ROOT/'glancewam_bundle'/item['path'], True))
assert len(jobs) > 15, len(jobs)
receipt_dir = ROOT/'results/gpu_validation'
receipt_dir.mkdir(parents=True, exist_ok=True)
planned = [{'repo': repo, 'revision': rev, 'path': item['path'], 'bytes': item.get('size'),
            'lfs': item.get('lfs'), 'destination': str(dest.relative_to(ROOT))}
           for repo, rev, item, dest, _ in jobs]
(receipt_dir/'download_plan.json').write_text(json.dumps(planned, indent=2))
print(json.dumps({'planned_files': len(jobs), 'planned_gib': sum(i.get('size',0) for _,_,i,_,_ in jobs)/2**30,
                  'base_revision': base_rev}), flush=True)

def fetch(job):
    repo, rev, item, dest, dataset = job
    expected = item.get('lfs', {}).get('oid', '').removeprefix('sha256:')
    dest.parent.mkdir(parents=True, exist_ok=True)
    def valid():
        return dest.exists() and dest.stat().st_size == item['size'] and (
            not expected or hashlib.sha256(dest.read_bytes()).hexdigest() == expected)
    if not valid():
        url = ENDPOINT+('/datasets/' if dataset else '/')+repo+'/resolve/'+rev+'/'+item['path']+'?download=true'
        tmp = dest.with_name(dest.name+'.part')
        subprocess.run(['curl','-fL','--retry','4','--retry-delay','3','--connect-timeout','15',
                        '--max-time','1800','--silent','--show-error',url,'-o',str(tmp)],check=True)
        assert tmp.stat().st_size == item['size'], (item['path'], tmp.stat().st_size, item['size'])
        tmp.replace(dest)
        assert valid(), 'Hash mismatch: '+item['path']
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    result = {'path': str(dest.relative_to(ROOT)), 'bytes': dest.stat().st_size, 'sha256': sha,
              'lfs_verified': bool(expected), 'repo_revision': rev}
    print(json.dumps(result), flush=True)
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    receipts = list(pool.map(fetch, jobs))
(receipt_dir/'download_receipt.json').write_text(json.dumps({'base_revision':base_rev,
    'data_revision':DATA_REV, 'files':receipts, 'all_complete':True}, indent=2))
print('ALL_DOWNLOADS_VERIFIED', flush=True)
