"""Official model/server with per-request RNG isolation for paired evaluation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')
sys.path.insert(0, str(Path.cwd()))

import torch
from accelerate import Accelerator
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from glancewam.model.framework.base_framework import baseframework


class PairedPolicy:
    def __init__(self, model, metrics):
        self.model, self.metrics = model, Path(metrics)

    def predict_action(self, examples, **kwargs):
        if len(examples) != 1 or 'seed' not in examples[0]:
            raise ValueError('Paired evaluation requires batch=1 and an explicit seed')
        seed = int(examples[0]['seed'])
        # The example seed alone seeds goal/action sampling in upstream, but NOT
        # VAE latent_dist.sample(). Seed the entire request as well.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = self.model.predict_action(examples=examples, **kwargs)
            torch.cuda.synchronize()
        row = dict(seed=seed, seconds=time.perf_counter() - start,
            goal_refresh='goal_latent' not in examples[0],
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        with self.metrics.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--metrics', required=True)
    p.add_argument('--port', type=int, default=6694)
    a = p.parse_args()
    Accelerator()
    torch.set_num_threads(4)
    model = baseframework.from_pretrained(a.ckpt).to(device='cuda', dtype=torch.bfloat16).eval()
    metadata = dict(env='libero', paired_rng=True)
    if getattr(model, 'goal_enabled', False):
        metadata['goal_horizon_rows'] = float(model._default_goal_horizon)
    torch.cuda.reset_peak_memory_stats()
    server = WebsocketPolicyServer(policy=PairedPolicy(model, a.metrics),
        host='127.0.0.1', port=a.port, max_batch=1, batch_wait_ms=0, metadata=metadata)
    server.serve_forever()


if __name__ == '__main__':
    main()
