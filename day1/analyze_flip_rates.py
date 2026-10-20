"""Run-to-run flip rates for repeated closed-loop episodes.

One closed-loop episode on a given initial state can land on either of two
outcomes: the task succeeds in a few dozen steps, or the policy stalls until
the step limit. Which one happens is decided by numerical noise in the GPU
forward pass, not by the model, so a single episode is not evidence about a
model. This script measures that directly: it replays the same protocol
several times and reports, per (model, episode), how often each outcome
appears and how many distinct action traces were produced.

The archive is the reference for what was recorded once. A model whose
archived outcome never reproduces is not a stable model-specific result.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_episodes(source):
    """Read one completed model run into {episode_id: record}.

    `source` is the run's results.json. A run counts only when that file
    exists, so a repetition that was interrupted mid-way is not silently
    counted as a shorter set of repetitions. The action trace of every episode
    is hashed from the file the run wrote, because the expansion driver does
    not put a trace hash in the record itself.
    """
    path = Path(source)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = {}
    for record in payload.get('episode_results', []):
        episode_id = int(record['episode_id'])
        trace = path.parent / f"task_{int(record['task_id']):02d}_episode_{episode_id:03d}_actions.npy"
        digest = None
        if trace.is_file():
            hasher = hashlib.sha256()
            with trace.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b''):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        records[episode_id] = dict(record, action_sha256=digest)
    return records


def flip_summary(runs, archive=None):
    """Per-episode outcome distribution over repeated runs.

    `runs` is a list of {episode_id: record} maps, one per repetition.
    Returns {episode_id: {...}} with the observed outcomes, the spread against
    the archived outcome and the number of distinct action traces.
    """
    per_episode = {}
    for run in runs:
        for episode_id, record in run.items():
            per_episode.setdefault(episode_id, []).append(record)
    summary = {}
    for episode_id, records in sorted(per_episode.items()):
        outcomes = Counter(bool(r['success']) for r in records)
        steps = sorted(int(r['steps']) for r in records)
        traces = {r.get('action_sha256') for r in records if r.get('action_sha256')}
        entry = dict(
            runs=len(records),
            successes=outcomes[True],
            failures=outcomes[False],
            success_rate=outcomes[True] / len(records),
            step_min=steps[0], step_max=steps[-1],
            distinct_step_values=len(set(steps)),
            distinct_action_sha256=len(traces),
        )
        if archive is not None and episode_id in archive:
            archived = archive[episode_id]
            entry['archived_success'] = bool(archived['success'])
            entry['archived_steps'] = int(archived['steps'])
            entry['matches_archive'] = outcomes[bool(archived['success'])]
            entry['archive_agreement_rate'] = outcomes[bool(archived['success'])] / len(records)
        summary[episode_id] = entry
    return summary


def attribution(summary):
    """Which per-model outcomes are stable enough to attribute to the model.

    An outcome that flips between runs carries no model-specific information:
    the run-to-run spread of the same model is as large as the difference
    between models. Only episodes where every repetition agrees may be
    reported as a model-specific result.
    """
    per_model = {}
    for model, episodes in summary.items():
        stable_failure = []
        stable_success = []
        unstable = []
        for episode_id, entry in episodes.items():
            if entry['successes'] == 0:
                stable_failure.append(episode_id)
            elif entry['failures'] == 0:
                stable_success.append(episode_id)
            else:
                unstable.append(episode_id)
        per_model[model] = dict(
            stable_failure_episodes=sorted(stable_failure),
            stable_success_episodes=sorted(stable_success),
            flipping_episodes=sorted(unstable),
            stable_failure_count=len(stable_failure),
            flipping_count=len(unstable),
        )
    return per_model


def state_stability(summary):
    """Per-state cross-model view: which states carry a stable conclusion.

    A state is stable for the whole set only when every model agrees on the
    outcome in every repetition. A state where any model flips is unstable for
    that model and cannot support a model-specific claim; the remaining states
    can.
    """
    states = {}
    for model, episodes in summary.items():
        for episode_id, entry in episodes.items():
            state = states.setdefault(episode_id, {})
            state[model] = dict(successes=entry['successes'], failures=entry['failures'],
                                runs=entry['runs'],
                                stable=entry['successes'] in (0, entry['runs']))
    out = {}
    for episode_id, models in states.items():
        flipping = sorted(m for m, v in models.items() if not v['stable'])
        outcomes = set()
        for v in models.values():
            if v['successes'] == v['runs']:
                outcomes.add('success')
            elif v['successes'] == 0:
                outcomes.add('failure')
            else:
                outcomes.add('unstable')
        out[episode_id] = dict(
            per_model={m: v['successes'] for m, v in sorted(models.items())},
            runs=models[next(iter(models))]['runs'],
            flipping_models=flipping,
            stable_overall=not flipping,
            # Agreement means every model reaches the same stable outcome; a model
            # that flips does not agree with anyone.
            all_models_agree=outcomes in ({'success'}, {'failure'}),
            shared_failure=outcomes == {'failure'},
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True,
                        help='Directory holding one subdirectory per repetition')
    parser.add_argument('--model', action='append', required=True,
                        help='Name=relative/sub/dir/model')
    parser.add_argument('--archive', help='Archived run directory for the same states')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    root = Path(args.runs)
    reps = sorted(p for p in root.iterdir() if p.is_dir())
    if not reps:
        raise SystemExit(f'no repetitions under {root}')

    models = {}
    for spec in args.model:
        name, _, relative = spec.partition('=')
        models[name] = relative

    summary = {}
    for name, relative in models.items():
        runs = []
        for rep in reps:
            source = rep / relative / 'results.json'
            if source.is_file():
                runs.append(read_episodes(source))
        if not runs:
            continue
        archive = None
        if args.archive:
            archived = Path(args.archive) / relative / 'results.json'
            if archived.is_file():
                archive = read_episodes(archived)
        summary[name] = flip_summary(runs, archive)

    report = dict(
        repetitions=len(reps),
        models=sorted(summary),
        episodes={name: {str(k): v for k, v in episodes.items()}
                  for name, episodes in summary.items()},
        attribution=attribution(summary),
        states={str(k): v for k, v in sorted(state_stability(summary).items())},
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
