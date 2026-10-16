"""Create a fixed pilot or confirmatory evaluation protocol without a simulator."""
import argparse
import json
from pathlib import Path

from eval_protocol import CONDITIONS, MAX_STEPS, protocol_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--suite', choices=list(MAX_STEPS), default='libero_goal')
    p.add_argument('--task-ids', default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--episode-start', type=int, default=0)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    if a.episodes < 1 or a.episode_start < 0:
        p.error('Invalid episode range')
    protocol = dict(schema_version=1, suite=a.suite,
        task_ids=[int(x) for x in a.task_ids.split(',')],
        episode_ids=list(range(a.episode_start, a.episode_start + a.episodes)),
        seed=a.seed, conditions=CONDITIONS)
    digest = protocol_hash(protocol)
    dest = Path(a.out)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps(dict(protocol=str(dest), sha256=digest,
        episodes_per_model=len(protocol['task_ids']) * a.episodes * len(CONDITIONS)), indent=2))


if __name__ == '__main__':
    main()
