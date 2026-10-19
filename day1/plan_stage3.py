"""Turn the state33 diagnostic verdict into the concrete next experiment.

Reads verdict.json produced by analyze_gripper_feedback.py and emits the branch to
take, the question that branch asks, and the exact commands to run. No model, no
environment and no GPU: the point is that the decision is made off the host before
spending a GPU window, and that a run whose prerequisites failed is refused rather
than explained away.

The branches encode a question, not an expected answer. What the diagnostic cannot
distinguish is stated as such, because a spec that pretends to know its own result
is worse than no spec.
"""
import argparse
import json
from pathlib import Path

# Failures that belong to exactly one model, from results/failure_structure. Only these
# can say anything about the fine-tuned models; the shared ones are base fragility.
MODEL_SPECIFIC_STATES = {26: ['base'], 32: ['aug'], 33: ['P'], 39: ['aug']}
SHARED_STATES = [28, 34, 35, 42]

SEALED = 'states 45-49 stay held back in every branch'


def prerequisites(verdict):
    """Blocking checks. A diagnostic that failed these cannot support any conclusion."""
    blockers = []
    control = verdict.get('control')
    if control is not None and not control.get('reproduced'):
        blockers.append(
            'the zero-intervention control did not reproduce the archived actions, so a '
            'difference in the intervention run cannot be attributed to the gripper override')
    if verdict.get('success_predicates_agree') is False:
        blockers.append('the environment termination flag and the state-log predicate disagree, '
                        'so the success reading is inconsistent and must be resolved first')
    if verdict.get('gripper_overridden_steps', 0) == 0 and verdict.get('hold_close_inclusive'):
        blockers.append('the gripper window was requested but no step was overridden')
    return blockers


def branch(verdict):
    """Map the transport axis and the success axis onto the next question."""
    outcome = verdict['outcome']
    if outcome == 'transported_and_placed':
        return dict(
            branch='A_placement_closed',
            question=('The bowl was carried and placed even though the model released the gripper '
                      'on its own after the window. Which part of the release decision is worth '
                      'keeping as-is?'),
            why_not_conclusive=('This is one held-back-adjacent initial state under one condition. '
                                'It shows the intervention was sufficient here, not that a general '
                                'rule follows.'),
            next_steps=[
                'Re-run the same protocol with --hold-close omitted and confirm the success '
                'disappears, which it must if the window is what did it.',
                'Repeat the window on states 32 and 39, the other model-specific failures, to '
                'see whether one window closes all of them or each is its own case.',
                'Read the chunk log around the placement to check whether the model released '
                'from the correct pose or only by luck.'])
    if outcome == 'transported_not_placed':
        return dict(
            branch='B_transport_only',
            question=('Holding the gripper restores carrying but not placing. This matches the '
                      '39/aug signature already seen offline, so the placement stage needs its own '
                      'probe rather than more gripper work.'),
            why_not_conclusive=('The diagnostic only held the gripper; it deliberately added no '
                                'release rule, so nothing here tests a placement fix.'),
            next_steps=[
                'Fix the placement stage as the independent variable: vary only the release step '
                'and hold the transport window fixed.',
                'Compare against 39/aug on the same axes, since the signature already matches.',
                'Keep the state log contact columns; placement failure needs contact, not '
                                'command magnitudes.'])
    return dict(
        branch='C_no_transport',
        question=('Holding the gripper closed through the window did not carry the bowl at all, '
                  'so the binding constraint is upstream of the release. Whether the grasp pose is '
                  'reachable is still unknown.'),
        why_not_conclusive=('Transport failing does not distinguish an unreachable pose from a plan '
                            'that never attempts one. Those need different probes.'),
        next_steps=[
            'Read the state log for the window: whether contact was ever established at all, and '
            'the contact gap length, using the end-anchored columns.',
            'Probe reachability separately with an approach-and-hold experiment that fixes the '
            'gripper closed for the whole episode and removes the arm command entirely.',
            'Do not attribute this to the action module without checking states 32 and 39, '
            'which are the other model-specific failures.'])


def common_context():
    return dict(
        model_specific_states=MODEL_SPECIFIC_STATES,
        shared_states_note=('states ' + ', '.join(str(s) for s in SHARED_STATES) +
                            ' fail for every model in both families; they are base-policy '
                            'fragility and no model-specific conclusion can be drawn from them'),
        sealed_states=SEALED,
        measurement_note=('judge model-specific failures with the end-anchored contact columns, '
                          'not the command-magnitude signature, which is a correlation only'))


REQUIRED_KEYS = ('outcome', 'verdict', 'hold_close_inclusive', 'gripper_overridden_steps',
                 'divergence_step_vs_reference', 'success_predicates_agree')


def load_verdict(path):
    """Refuse an incomplete verdict explicitly; this tool gates the expensive runs."""
    data = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f'verdict is missing required keys: {missing}')
    if data['outcome'] not in ('transported_and_placed', 'transported_not_placed', 'not_transported'):
        raise ValueError(f'unknown outcome {data["outcome"]!r}; regenerate the verdict '
                         'with analyze_gripper_feedback.py')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verdict', required=True, help='verdict.json from analyze_gripper_feedback.py')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    verdict = load_verdict(args.verdict)
    blockers = prerequisites(verdict)
    plan = dict(
        source_verdict=dict(outcome=verdict['outcome'], verdict=verdict['verdict'],
                            success=verdict.get('success'),
                            hold_close_inclusive=verdict.get('hold_close_inclusive'),
                            divergence_step_vs_reference=verdict.get('divergence_step_vs_reference'),
                            max_lift_mm=verdict.get('max_lift_mm'),
                            final_xy_mm=verdict.get('final_xy_mm'),
                            lift_at_last_finger_contact_mm=verdict.get('lift_at_last_finger_contact_mm')),
        prerequisites_blocking=blockers,
        executable=not blockers,
        **common_context(),
        **(dict(branch='BLOCKED', question='prerequisites failed', next_steps=[
            'Resolve the blockers before any follow-up. A result whose control run did not '
            'reproduce the archive cannot be attributed to the intervention.'])
            if blockers else branch(verdict)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2))
    return 0 if plan['executable'] else 2


if __name__ == '__main__':
    raise SystemExit(main())