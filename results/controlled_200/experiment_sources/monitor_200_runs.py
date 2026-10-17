"""Record process start and completion times without changing running trainers."""
import json
import os
from pathlib import Path
import time

root=Path('/root/autodl-tmp/wam_glance_project')
out=root/'GlanceWAM/results/controlled_200'
starts={}
records={}
boot=float(next(l.split()[1] for l in Path('/proc/stat').read_text().splitlines() if l.startswith('btime ')))
hz=os.sysconf(os.sysconf_names['SC_CLK_TCK'])
while len(records)<2:
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            argv=(folder/'cmdline').read_bytes().split(b'\0')
            if not any(x.endswith(b'finetune_action_head.py') for x in argv):continue
            for name in ['clean','aug']:
                target=('results/controlled_200/'+name+'_seed42').encode()
                if target in argv and name not in starts:
                    fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
                    # First field after ')' is stat field 3; starttime is field 22.
                    started=boot+int(fields[19])/hz
                    starts[name]={'pid':int(folder.name),'process_start_unix':started}
                    print(json.dumps({'event':'found', 'model':name,**starts[name]}),flush=True)
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    for name in starts:
        summary=out/(name+'_seed42')/'summary.json'
        if name not in records and summary.exists():
            payload=json.loads(summary.read_text())
            if payload.get('completed_steps')==200:
                completed=summary.stat().st_mtime
                records[name]={**starts[name],'summary_written_unix':completed,
                    'process_start_to_summary_seconds':completed-starts[name]['process_start_unix']}
                print(json.dumps({'event':'completed','model':name,**records[name]}),flush=True)
    (out/'run_wall_times.json').write_text(json.dumps({'starts':starts,'completed':records},indent=2))
    time.sleep(2)
