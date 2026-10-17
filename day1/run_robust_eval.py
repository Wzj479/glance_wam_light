"""Start one GPU policy at a time, run fixed conditions, aggregate paired results."""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from eval_protocol import paired_delta, protocol_hash, validate_episode_results


def checkpoint_view_path(path):
    # Config/statistics belong to the caller's run directory, even when the
    # large weights file is a symlink into the original download bundle.
    ckpt = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    for name in ('config.yaml', 'dataset_statistics.json'):
        if not (ckpt.parents[1] / name).is_file():
            raise FileNotFoundError(ckpt.parents[1] / name)
    return ckpt


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            return True
    except OSError:
        return False


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


def aggregate(out, protocol, names):
    summaries, lines = {}, ['model\tcondition\tsuccesses\tepisodes\trate\tdelta_pp_vs_base']
    for name in names:
        summaries[name] = {}
        for condition in protocol['conditions']:
            result = json.loads((out / name / condition / 'results.json').read_text())
            if result['protocol_sha256'] != protocol_hash(protocol):
                raise ValueError('Protocol hash mismatch')
            validate_episode_results(result['episode_results'], protocol)
            summaries[name][condition] = result
    rows = []
    for name in names:
        for condition in protocol['conditions']:
            result = summaries[name][condition]
            row = dict(model=name, condition=condition, successes=result['successes'],
                       episodes=result['episodes'], success_rate=result['success_rate'])
            if 'base' in summaries:
                base = summaries['base'][condition]['episode_results']
                candidate = result['episode_results']
                # Guard against accidentally pairing different physical initial states.
                for r in candidate:
                    b = next(x for x in base if (x['task_id'], x['episode_id']) == (r['task_id'], r['episode_id']))
                    if b['init_state_sha256'] != r['init_state_sha256'] or b['env_seed'] != r['env_seed']:
                        raise ValueError('Physical initial state pairing mismatch')
                    if b.get('post_settle_state_sha256') != r.get('post_settle_state_sha256'):
                        raise ValueError('Post-settle state pairing mismatch')
                row.update(paired_delta(base, candidate))
            row['per_task'] = {str(tid): dict(
                successes=sum(r['success'] for r in result['episode_results'] if r['task_id'] == tid),
                episodes=len(protocol['episode_ids'])) for tid in protocol['task_ids']}
            rows.append(row)
            lines.append(f"{name}\t{condition}\t{row['successes']}\t{row['episodes']}\t"
                         f"{row['success_rate']:.3f}\t{row.get('success_delta_pp', 0.):.2f}")
    (out / 'summary.json').write_text(json.dumps(dict(protocol_sha256=protocol_hash(protocol), rows=rows), indent=2))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--model', action='append', required=True, help='name=/absolute/or/relative/checkpoint.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--port', type=int, default=6694)
    p.add_argument('--model-python', default='.venv/bin/python')
    p.add_argument('--libero-python', default='../LIBERO/.venv/bin/python')
    p.add_argument('--libero-home', default='../LIBERO')
    p.add_argument('--server-timeout', type=int, default=900)
    p.add_argument('--save-videos', action='store_true')
    a = p.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Run on the Linux GPU server')
    if ',' in a.gpu:
        raise ValueError('Specify one GPU for this single-card evaluator')
    protocol_path = Path(a.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    protocol_hash(protocol)
    models = {}
    for pair in a.model:
        name, path = pair.split('=', 1)
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', name) or name in models:
            raise ValueError('Model names must be unique simple identifiers')
        ckpt = checkpoint_view_path(path)
        models[name] = str(ckpt)
    for python in (a.model_python, a.libero_python):
        if not Path(python).is_file():
            raise FileNotFoundError(python)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new --out directory; partial results will not be mixed')
    if port_open(a.port):
        raise RuntimeError(f'Port {a.port} is already occupied')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out / 'models.json').write_text(json.dumps(models, indent=2))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = a.gpu
    env['PYTHONPATH'] = os.pathsep.join([str(Path.cwd()), str(Path(a.libero_home).resolve()), env.get('PYTHONPATH', '')])
    env['LIBERO_CONFIG_PATH'] = str(Path(a.libero_home).resolve() / 'libero')
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('PYOPENGL_PLATFORM', 'egl')
    env.setdefault('OMP_NUM_THREADS', '4')
    cuda_lib = Path('.venv/lib/python3.11/site-packages/nvidia/cu12/lib').resolve()
    if cuda_lib.is_dir():
        env['LD_LIBRARY_PATH'] = str(cuda_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    # Single visible GPU; override this variable when EGL enumeration differs.
    if env['MUJOCO_GL'] == 'egl':
        env.setdefault('GLANCEWAM_EGL_DEVICE_ID', '0')
    for name, ckpt in models.items():
        model_out = out / name
        model_out.mkdir()
        server, client = None, None
        with (model_out / 'server.log').open('w') as server_log:
            try:
                server = subprocess.Popen([a.model_python, 'day1/paired_policy_server.py',
                    '--ckpt', ckpt, '--port', str(a.port), '--metrics', str(model_out / 'server_metrics.jsonl')],
                    env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + a.server_timeout
                while not port_open(a.port):
                    if server.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f'Server startup failed; inspect {model_out / "server.log"}')
                    time.sleep(1)
                for condition in protocol['conditions']:
                    condition_out = model_out / condition
                    condition_out.mkdir()
                    command = [a.libero_python, 'day1/eval_robust_libero.py', '--protocol', str(protocol_path),
                        '--condition', condition, '--ckpt', ckpt, '--model-name', name,
                        '--out', str(condition_out), '--port', str(a.port)]
                    if a.save_videos:
                        command.append('--save-videos')
                    with (condition_out / 'client.log').open('w') as client_log:
                        client = subprocess.Popen(command, env=env, stdout=client_log,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        while client.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError('Policy server exited during evaluation')
                            time.sleep(1)
                        if client.returncode:
                            raise RuntimeError(f'Evaluation failed; inspect {condition_out / "client.log"}')
            finally:
                stop(client)
                stop(server)
    aggregate(out, protocol, list(models))


if __name__ == '__main__':
    main()
