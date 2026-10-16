"""Select tasks using base-model pilot only; reserve disjoint final initial states."""
import argparse
import copy
import json
from pathlib import Path

from eval_protocol import protocol_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', required=True, help='run_robust_eval output directory')
    p.add_argument('--out', required=True)
    p.add_argument('--num-tasks', type=int, default=3)
    p.add_argument('--clean-min', type=float, default=2 / 3)
    p.add_argument('--episode-start', type=int, default=10)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.num_tasks < 1 or a.episodes < 1 or not 0 <= a.clean_min <= 1:
        p.error('Invalid selection parameters')
    root = Path(a.pilot)
    protocol = json.loads((root / 'protocol.json').read_text())
    report = json.loads((root / 'summary.json').read_text())
    if report['protocol_sha256'] != protocol_hash(protocol):
        raise ValueError('Pilot report/protocol mismatch')
    rows = {r['condition']: r for r in report['rows'] if r['model'] == 'base'}
    corrupt_names = [c for c in protocol['conditions'] if c != 'clean']
    if 'clean' not in rows or not corrupt_names:
        raise ValueError('Pilot needs base model, clean and at least one corrupt condition')
    ranks = []
    for tid in protocol['task_ids']:
        def rate(condition):
            r = rows[condition]['per_task'][str(tid)]
            return r['successes'] / r['episodes']
        clean = rate('clean')
        drop = clean - sum(rate(c) for c in corrupt_names) / len(corrupt_names)
        if clean >= a.clean_min and drop > 0:
            ranks.append(dict(task_id=tid, pilot_clean_rate=clean, pilot_mean_corrupt_drop=drop))
    ranks.sort(key=lambda r: (-r['pilot_mean_corrupt_drop'], -r['pilot_clean_rate'], r['task_id']))
    if len(ranks) < a.num_tasks:
        raise ValueError(f'Only {len(ranks)} sensitive tasks qualify; inspect pilot or revise pilot severity. '
                         'Do not select tasks using fine-tuned final results.')
    final = copy.deepcopy(protocol)
    final['task_ids'] = [r['task_id'] for r in ranks[:a.num_tasks]]
    final['episode_ids'] = list(range(a.episode_start, a.episode_start + a.episodes))
    final['seed'] = a.seed
    if set(final['episode_ids']) & set(protocol['episode_ids']):
        raise ValueError('Pilot and final initial-state IDs must be disjoint')
    digest = protocol_hash(final)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2), encoding='utf-8')
    out.with_suffix('.selection.json').write_text(json.dumps(dict(
        source_pilot_sha256=protocol_hash(protocol), final_sha256=digest,
        eligible_task_ranking=ranks, chosen=ranks[:a.num_tasks]), indent=2), encoding='utf-8')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
