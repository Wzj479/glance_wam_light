"""Re-run agreement between an expanded evaluation and the archive it extends.

Reads only results.json from two finished evaluations. No environment, model or
physics execution, so it runs without a GPU and cannot perturb anything.

Widening a protocol from 25-44 to 0-44 re-executes the episodes the archive already
covers. Those repeats are the only evidence that the policy server is deterministic
here, which is the same property the gripper diagnostic gates on with its zero
intervention control. aggregate() in run_robust_eval.py pairs models against each
other inside one output directory and deliberately says nothing about earlier runs,
so this comparison has to be requested explicitly.

Agreement is required on the physical initial state (env_seed, init_state_sha256,
post_settle_state_sha256) and on the outcome (success, steps). A disagreement on the
physical fields means the two runs are not the same experiment at all; a
disagreement on success or steps means inference is not deterministic.
"""
import argparse
import json
from pathlib import Path

# Fields that make two runs of the same episode the same experiment.
PHYSICAL_FIELDS = ('env_seed', 'init_state_sha256', 'post_settle_state_sha256')
OUTCOME_FIELDS = ('success', 'steps')


def read_results(source, model, condition):
    path = Path(source) / model / condition / 'results.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    for key in ('protocol_sha256', 'episode_results'):
        if key not in result:
            raise ValueError(f'{path} is missing {key!r}; not a finished evaluation')
    return result


def index_by_episode(result):
    records = {}
    for record in result['episode_results']:
        key = (record['task_id'], record['episode_id'])
        if key in records:
            raise ValueError(f'Duplicate episode {key}')
        records[key] = record
    return records


def compare_pair(archive_record, expansion_record):
    """Field-level differences, split into physical and outcome groups."""
    physical, outcome = [], []
    for field in PHYSICAL_FIELDS:
        if archive_record.get(field) != expansion_record.get(field):
            physical.append(field)
    for field in OUTCOME_FIELDS:
        if archive_record.get(field) != expansion_record.get(field):
            outcome.append(field)
    return physical, outcome


def verify(archive, expansion, models, conditions):
    report = {'models': models, 'conditions': conditions, 'archive': str(archive),
              'expansion': str(expansion), 'protocol_sha256': {}, 'shared_episodes': {},
              'new_episodes': {}, 'differences': [], 'protocol_note': ''}
    archive_hashes, expansion_hashes = set(), set()
    for model in models:
        for condition in conditions:
            old = read_results(archive, model, condition)
            new = read_results(expansion, model, condition)
            archive_hashes.add(old['protocol_sha256'])
            expansion_hashes.add(new['protocol_sha256'])
            old_index, new_index = index_by_episode(old), index_by_episode(new)
            shared = sorted(set(old_index) & set(new_index))
            key = f'{model}/{condition}'
            report['shared_episodes'][key] = len(shared)
            report['new_episodes'][key] = sorted(ep for _, ep in set(new_index) - set(old_index))
            if not shared:
                raise ValueError(
                    f'{key} shares no episode with the archive; a widened protocol must '
                    're-run the archived episodes so determinism can be observed')
            for episode in shared:
                physical, outcome = compare_pair(old_index[episode], new_index[episode])
                if physical or outcome:
                    report['differences'].append(dict(
                        model=model, condition=condition, task_id=episode[0],
                        episode_id=episode[1], physical=physical, outcome=outcome))
    report['protocol_sha256'] = {'archive': sorted(archive_hashes),
                                 'expansion': sorted(expansion_hashes)}
    if len(archive_hashes) != 1 or len(expansion_hashes) != 1:
        raise ValueError('Each source must come from a single protocol')
    if archive_hashes == expansion_hashes:
        report['protocol_note'] = ('archive and expansion hashes are identical, so the '
                                   'expansion did not widen anything')
    else:
        report['protocol_note'] = ('hashes differ because the episode list widened; the '
                                   'shared episodes are still directly comparable')
    report['reproduced'] = not report['differences']
    report['checked_records'] = sum(report['shared_episodes'].values())
    return report


def render(report):
    lines = [f"archive   : {report['archive']}",
             f"expansion : {report['expansion']}",
             f"protocol  : {report['protocol_note']}"]
    for key, count in report['shared_episodes'].items():
        new = report['new_episodes'][key]
        lines.append(f'{key}: {count} shared episodes re-checked, {len(new)} newly covered')
    if report['reproduced']:
        lines.append(f"REPRODUCED: {report['checked_records']} shared episode records agree "
                     'on initial state, success and step count')
    else:
        lines.append(f"NOT REPRODUCED: {len(report['differences'])} of "
                     f"{report['checked_records']} shared episode records disagree")
        for difference in report['differences']:
            lines.append(f"  {difference['model']}/{difference['condition']} "
                         f"task {difference['task_id']} episode {difference['episode_id']}: "
                         f"physical={difference['physical']} outcome={difference['outcome']}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, help='Archived evaluation directory')
    p.add_argument('--expansion', required=True, help='Widened evaluation directory')
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--conditions', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--label', default='')
    a = p.parse_args()
    report = verify(a.archive, a.expansion, a.models, a.conditions)
    report['label'] = a.label
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(render(report))
    # A disagreement here voids every success rate in the expansion, so it stops the
    # run the same way plan_stage3.py stops an unreproduced control.
    return 0 if report['reproduced'] else 2


if __name__ == '__main__':
    raise SystemExit(main())