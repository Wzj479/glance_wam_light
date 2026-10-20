"""Score the world model's imagined goal against the goal state that was actually reached.

During the closed loop, diagnose_gripper_feedback.py --dump-goal saves, at every query, the
policy-view image pair and the goal latent the world model produced (`client._cached_goal`).
The goal latent is the world model's "imagination half": a prediction of the task goal
conditioned on the current observation, in normalized VAE latent space.

This script measures one non-circular thing: how far is the imagined goal from the real goal?

  g*  = the mean VAE latent of the terminal frames of SUCCESSFUL runs (bowl on the plate).
        Fixed in advance, from outcomes the scored trajectory did not decide.
  d_t = || goal_t - g* || / || g* ||   at each query t.

Two modes for goal_t:
  default        use the goal latent the closed loop cached (may be cadence-stale: piecewise
                 constant within a chunk).
  --fresh        re-run predict_action on each saved frame to get a fresh goal per frame -- a
                 genuine per-observation prediction, no chunk-cadence stepping.

d_t asks "given what the camera shows now, how accurately does the world model imagine the
task goal?" -- a function of the observation alone, not of whether the run succeeds. The
question the project cares about: does d_t stay low / converge on successful trajectories and
stay high or rise on failing ones?

Order matters: encode the reference frames BEFORE any predict_action call (the eval path
leaves the VAE encoders unusable for a later _encode_images).
"""
import argparse, glob, json, os, sys
from pathlib import Path

for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k, '4')
sys.path.insert(0, str(Path.cwd() / 'day1'))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
from glancewam.model.framework.base_framework import baseframework
from glancewam.dataloader.image_tools import to_pil_preserve
from action_checkpoint import load_policy_checkpoint

CKPT = 'results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt'


def frames_of(run):
    for d in (Path(run) / 'driver' / 'frames', Path(run) / 'frames'):
        if d.is_dir():
            return sorted(d.glob('step_*.npz'))
    return []


def encode_frame(model, view0, view1):
    hist = model.num_history_frames
    with torch.no_grad():
        batch = model._maybe_stitch_camera_views([to_pil_preserve([view0, view1])])
        latent, _ = model.backbone._encode_images([[im] * hist for im in batch], num_frames=hist)
    return latent.float()[0].cpu()  # [16,1,28,56]


def fresh_goal(model, view0, view1, lang, seed):
    res = model.predict_action(examples=[dict(image=[view0, view1], lang=str(lang), seed=int(seed))])
    return torch.as_tensor(np.asarray(res['goal_latent'])).float().cpu()[0]  # [16,1,28,56]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ref', nargs='+', required=True, help='Successful run dirs; g* = mean of their terminal-frame latents')
    ap.add_argument('--runs', nargs='+', required=True, help='Run dirs to score')
    ap.add_argument('--fresh', action='store_true', help='Recompute a fresh goal per frame via predict_action')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    model = load_policy_checkpoint(CKPT, baseframework).to(device='cuda', dtype=torch.bfloat16).eval()

    # g*: mean terminal-frame latent over all reference runs. Encode FIRST (VAE gotcha).
    ref_latents = []
    for ref in a.ref:
        fr = frames_of(ref)
        if not fr:
            continue
        last = np.load(fr[-1], allow_pickle=True)
        ref_latents.append(encode_frame(model, last['view0'], last['view1']))
    if not ref_latents:
        raise SystemExit('no reference frames found')
    g_star = torch.stack(ref_latents).mean(0)
    out = dict(reference=a.ref, n_ref=len(ref_latents), fresh=a.fresh,
               g_star_shape=list(g_star.shape), g_star_norm=round(float(g_star.norm()), 4), runs={})

    for run in a.runs:
        fr = frames_of(run)
        if not fr:
            out['runs'][run] = dict(error='no frames')
            continue
        rows = []
        for f in fr:
            d = np.load(f, allow_pickle=True)
            step = int(f.stem.split('_')[-1])
            if a.fresh:
                goal = fresh_goal(model, d['view0'], d['view1'], d['lang'][0], d['seed'][0])
            else:
                if not bool(d['has_goal'][0]) or d['goal'].ndim < 4:
                    continue
                goal = torch.as_tensor(d['goal']).float()
            dval = float((goal - g_star).norm() / g_star.norm())
            cos = float(torch.nn.functional.cosine_similarity(
                goal.flatten().unsqueeze(0), g_star.flatten().unsqueeze(0), dim=1))
            rows.append(dict(step=step, d=round(dval, 4), cosine=round(cos, 4)))
        ds = [r['d'] for r in rows]
        out['runs'][run] = dict(
            n_frames=len(rows),
            d_first=ds[0] if ds else None, d_last=ds[-1] if ds else None,
            d_mean=round(float(np.mean(ds)), 4) if ds else None,
            d_min=round(float(np.min(ds)), 4) if ds else None,
            d_max=round(float(np.max(ds)), 4) if ds else None,
            trend=round(ds[-1] - ds[0], 4) if len(ds) >= 2 else None,
            curve=rows)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('GOAL_SCORE ' + json.dumps({k: out[k] for k in ('reference', 'n_ref', 'fresh', 'g_star_shape', 'g_star_norm')}))
    for run, r in out['runs'].items():
        if 'error' in r:
            print(f'  {Path(run).name:<16} ERROR {r["error"]}')
        else:
            print(f'  {Path(run).name:<16} n={r["n_frames"]:<3} d_first={r["d_first"]} d_last={r["d_last"]} '
                  f'd_mean={r["d_mean"]} trend={r["trend"]}')


if __name__ == '__main__':
    main()
