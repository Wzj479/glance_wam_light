"""Tests that the held-back initial states are mechanically blocked, not just noted."""
import unittest

from eval_protocol import RESERVED_STATES, protocol_hash, reserved_states, validate_protocol


def protocol(episode_ids, task_ids=(8,), suite='libero_goal'):
    return dict(schema_version=1, suite=suite, seed=42, task_ids=list(task_ids),
                episode_ids=list(episode_ids),
                conditions={'clean': dict(brightness=1., saturation=1., gains=[1., 1., 1.])})


class ReservationScopeTests(unittest.TestCase):
    def test_task8_holds_back_exactly_the_documented_states(self):
        self.assertEqual(RESERVED_STATES[('libero_goal', 8)], frozenset({45, 46, 47, 48, 49}))

    def test_other_tasks_and_suites_hold_nothing_back(self):
        self.assertEqual(reserved_states(protocol([25], task_ids=[0])), set())
        self.assertEqual(reserved_states(protocol([25], suite='libero_10')), set())


class EnforcementTests(unittest.TestCase):
    def test_protocol_selecting_a_held_back_state_is_rejected(self):
        for episode in (45, 47, 49):
            with self.assertRaises(ValueError) as caught:
                validate_protocol(protocol([33, episode]))
            self.assertIn('held back', str(caught.exception))

    def test_the_documented_evaluation_range_still_validates(self):
        validate_protocol(protocol(range(25, 45)))

    def test_ordinary_failure_states_are_allowed(self):
        for episode in (26, 28, 32, 33, 34, 35, 39, 42, 44):
            validate_protocol(protocol([episode]))

    def test_reserved_states_reported_for_a_task_include_only_that_task(self):
        self.assertEqual(reserved_states(protocol([33])), {45, 46, 47, 48, 49})


class HashTests(unittest.TestCase):
    def test_a_reserved_state_changes_the_hash_before_it_is_rejected(self):
        # Rejection happens at validation; hashing a reserved protocol must still not
        # silently produce a value that could be recorded as a legitimate run.
        clean = protocol([33])
        with self.assertRaises(ValueError):
            protocol_hash(protocol([33, 45]))
        self.assertTrue(protocol_hash(clean))


if __name__ == '__main__':
    unittest.main()