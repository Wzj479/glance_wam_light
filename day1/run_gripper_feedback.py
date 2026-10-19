"""Serve one checkpoint and run the gripper-intervention diagnostic on it.

Mirrors run_robust_eval.py's process setup, but drives the single-state diagnostic
driver instead of the suite evaluator, so a subset run cannot be mistaken for a
suite result. Everything after --driver-args is forwarded to the driver.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path


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


def checkpoint_view_path(path):
    # Config and statistics belong to the caller's run directory even when the
    # weights file is a symlink into the original download bundle.
    ckpt = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    for name in ('config.yaml', 'dataset_statistics.json'):
        if not (ckpt.parents[1] / name).is_file():
            raise FileNotFoundError(ckpt.parents[1] / name)
    return ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model-name', default='P')
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=6694)
    p.add_argument('--gpu', default='0')
    p.add_argument('--model-python', default='.venv/bin/python')
    p.add_argument('--libero-python', default='/root/wam_sim_env/bin/python')
    p.add_argument('--libero-home', default='/root/autodl-tmp/wam_glance_project/LIBERO')
    p.add_argument('--server-timeout', type=int, default=900)
    p.add_argument('--driver-args', nargs=argparse.REMAINDER, default=[],
                   help='Everything after this flag is forwarded to diagnose_gripper_feedback.py')
    a = p.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Run on the Linux GPU server')
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', a.model_name):
        raise ValueError('Model name must be a simple identifier')
    protocol_path = Path(a.protocol).resolve()
    ckpt = checkpoint_view_path(a.ckpt)
    for python in (a.model_python, a.libero_python):
        if not Path(python).is_file():
            raise FileNotFoundError(python)
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a new --out directory; partial results will not be mixed')
    if port_open(a.port):
        raise RuntimeError(f'Port {a.port} is already occupied')

    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = a.gpu
    env['PYTHONPATH'] = os.pathsep.join([str(Path.cwd()), str(Path(a.libero_home).resolve()),
                                         env.get('PYTHONPATH', '')])
    env['LIBERO_CONFIG_PATH'] = str(Path(a.libero_home).resolve() / 'libero')
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('PYOPENGL_PLATFORM', 'egl')
    env.setdefault('OMP_NUM_THREADS', '8')
    cuda_lib = Path('.venv/lib/python3.11/site-packages/nvidia/cu12/lib').resolve()
    if cuda_lib.is_dir():
        env['LD_LIBRARY_PATH'] = str(cuda_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    if env['MUJOCO_GL'] == 'egl':
        env.setdefault('GLANCEWAM_EGL_DEVICE_ID', '0')

    server, client = None, None
    with (out / 'server.log').open('w') as server_log:
        try:
            server = subprocess.Popen([a.model_python, 'day1/paired_policy_server.py',
                '--ckpt', str(ckpt), '--port', str(a.port),
                '--metrics', str(out / 'server_metrics.jsonl')],
                env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + a.server_timeout
            while not port_open(a.port):
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('Server startup failed; inspect the server log')
                time.sleep(1)
            command = [a.libero_python, 'day1/diagnose_gripper_feedback.py',
                '--protocol', str(protocol_path), '--condition', a.condition,
                '--ckpt', str(ckpt), '--model-name', a.model_name,
                '--out', str(out / 'driver'), '--port', str(a.port)] + list(a.driver_args)
            print(' '.join(command), flush=True)
            with (out / 'client.log').open('w') as client_log:
                client = subprocess.Popen(command, env=env, stdout=client_log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError('Policy server exited during the run')
                    time.sleep(1)
                if client.returncode:
                    raise RuntimeError('Diagnostic failed; inspect the client log')
        finally:
            stop(client)
            stop(server)

    summary = json.loads((out / 'driver' / 'results.json').read_text())
    result = summary['episode_results'][0]
    print(json.dumps(dict(model=summary['model'], condition=summary['condition'],
                          hold_close_inclusive=summary['hold_close_inclusive'],
                          episodes=summary['episodes'], successes=summary['successes'],
                          success_rate=summary['success_rate'], steps=result['steps'],
                          success=result['success'],
                          gripper_overridden_steps=result['gripper_overridden_steps'],
                          divergence_step_vs_reference=result['divergence_step_vs_reference'],
                          action_sha256=result['action_sha256'],
                          init_state_sha256=result['init_state_sha256'],
                          post_settle_state_sha256=result['post_settle_state_sha256'],
                          protocol_sha256=summary['protocol_sha256']), indent=2))


if __name__ == '__main__':
    main()