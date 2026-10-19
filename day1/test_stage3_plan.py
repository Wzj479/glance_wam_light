"""Tests for the stage-three decision spec; no model, environment or GPU."""
import unittest

from plan_stage3 import MODEL_SPECIFIC_STATES, SHARED_STATES, branch, common_context, prerequisites


def verdict(**overrides):
    base = dict(outcome='not_transported', verdict='no_transport_under_intervention', success=False,
                hold_close_inclusive=[56, 75], gripper_overridden_steps=20,
                divergence_step_vs_reference=64, success_predicates_agree=True,
                control=dict(reproduced=True), max_lift_mm=4.29, final_xy_mm=137.0,
                lift_at_last_finger_contact_mm=4.29)
    base.update(overrides)
    return base


class PrerequisiteTests(unittest.TestCase):
    def test_a_control_that_did_not_reproduce_blocks_everything(self):
        blockers = prerequisites(verdict(control=dict(reproduced=False)))
        self.assertTrue(blockers)
        self.assertIn('cannot be attributed', blockers[0])

    def test_disagreeing_success_predicates_block(self):
        self.assertTrue(prerequisites(verdict(success_predicates_agree=False)))

    def test_a_window_that_overrode_nothing_blocks(self):
        self.assertTrue(prerequisites(verdict(gripper_overridden_steps=0)))

    def test_a_control_run_absent_is_not_treated_as_a_failure(self):
        self.assertEqual(prerequisites(verdict(control=None)), [])

    def test_a_clean_verdict_has_no_blockers(self):
        self.assertEqual(prerequisites(verdict()), [])


class BranchTests(unittest.TestCase):
    def test_transport_only_asks_about_the_placement_stage(self):
        chosen = branch(verdict(outcome='transported_not_placed'))
        self.assertEqual(chosen['branch'], 'B_transport_only')
        self.assertIn('placement', chosen['question'])

    def test_carried_and_placed_repeats_on_other_model_specific_states(self):
        chosen = branch(verdict(outcome='transported_and_placed'))
        self.assertEqual(chosen['branch'], 'A_placement_closed')
        self.assertTrue(any('32' in step and '39' in step for step in chosen['next_steps']))

    def test_no_transport_asks_about_reachability_upstream(self):
        chosen = branch(verdict(outcome='not_transported'))
        self.assertEqual(chosen['branch'], 'C_no_transport')
        self.assertIn('reachab', chosen['question'])

    def test_every_branch_states_its_own_limits(self):
        for outcome in ('transported_and_placed', 'transported_not_placed', 'not_transported'):
            chosen = branch(verdict(outcome=outcome))
            self.assertTrue(chosen['why_not_conclusive'].strip())

    def test_the_sealed_states_are_a_plan_level_invariant_not_a_branch_note(self):
        # One global statement is enough; duplicating it per branch would let them drift apart.
        self.assertIn('45-49', common_context()['sealed_states'])
        for outcome in ('transported_and_placed', 'transported_not_placed', 'not_transported'):
            chosen = branch(verdict(outcome=outcome))
            self.assertNotIn('45', ' '.join(chosen['next_steps']))

    def test_no_branch_proposes_using_a_sealed_state(self):
        sealed = {'45', '46', '47', '48', '49'}
        for outcome in ('transported_and_placed', 'transported_not_placed', 'not_transported'):
            text = ' '.join(branch(verdict(outcome=outcome))['next_steps'])
            self.assertFalse(any(f'state {s}' in text or f'states {s}' in text for s in sealed))


class ContextTests(unittest.TestCase):
    def test_shared_states_are_excluded_from_model_specific_claims(self):
        self.assertTrue(set(SHARED_STATES).isdisjoint(MODEL_SPECIFIC_STATES))

    def test_state_33_is_the_only_P_specific_failure(self):
        p_states = [s for s, m in MODEL_SPECIFIC_STATES.items() if m == ['P']]
        self.assertEqual(p_states, [33])

    def test_context_tells_the_reader_not_to_use_the_correlation(self):
        self.assertIn('correlation', common_context()['measurement_note'])


class VerdictLoadingTests(unittest.TestCase):
    def test_incomplete_verdict_is_refused_with_the_missing_keys_named(self):
        import json
        import tempfile
        from pathlib import Path

        from plan_stage3 import load_verdict
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v.json'
            path.write_text(json.dumps({'outcome': 'not_transported'}))
            with self.assertRaises(ValueError) as caught:
                load_verdict(path)
            self.assertIn('hold_close_inclusive', str(caught.exception))

    def test_unknown_outcome_points_back_at_the_verdict_generator(self):
        import json
        import tempfile
        from pathlib import Path

        from plan_stage3 import load_verdict
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v.json'
            path.write_text(json.dumps(verdict(outcome='looks_fine')))
            with self.assertRaises(ValueError) as caught:
                load_verdict(path)
            self.assertIn('analyze_gripper_feedback.py', str(caught.exception))

    def test_a_complete_verdict_loads_unchanged(self):
        import json
        import tempfile
        from pathlib import Path

        from plan_stage3 import load_verdict
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'v.json'
            path.write_text(json.dumps(verdict()))
            self.assertEqual(load_verdict(path)['outcome'], 'not_transported')


if __name__ == '__main__':
    unittest.main()