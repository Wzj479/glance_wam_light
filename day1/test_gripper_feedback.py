"""Tests for the gripper-intervention diagnostic's pure logic; no simulator or model."""
import json
import unittest

import numpy as np

from diagnose_gripper_feedback import (
    GRIPPER_CLOSE, chunk_records, first_divergence_step, override_gripper, parse_window, slip_series,
    subset_protocol)
from eval_protocol import protocol_hash
from finetune_utils import stable_seed


def trajectory(gripper_values, arm=None):
    arm = np.zeros((len(gripper_values), 6)) if arm is None else np.asarray(arm, dtype=float)
    return np.concatenate([arm, np.asarray(gripper_values, dtype=float).reshape(-1, 1)], axis=1)


class WindowTests(unittest.TestCase):
    def test_absent_window_is_a_control_run(self):
        self.assertIsNone(parse_window(None))
        self.assertEqual(parse_window([56, 75]), (56, 75))

    def test_reversed_or_negative_window_rejected(self):
        for bad in ([75, 56], [-1, 10]):
            with self.assertRaises(ValueError):
                parse_window(bad)
        with self.assertRaises(ValueError):
            parse_window([1, 2, 3])

    def test_window_may_extend_past_the_episode(self):
        # Holding to a large END is well defined: every remaining step is covered.
        self.assertEqual(parse_window([56, 10 ** 6]), (56, 10 ** 6))
        action = np.array([.1, .2, .3, .04, .05, .06, -1.0])
        self.assertTrue(override_gripper(action, 299, (56, 10 ** 6))[1])


class OverrideTests(unittest.TestCase):
    def test_only_the_gripper_channel_changes_inside_the_window(self):
        action = np.array([.1, .2, .3, .04, .05, .06, -1.0])
        for step in (56, 60, 75):
            executed, overridden = override_gripper(action, step, (56, 75))
            self.assertTrue(overridden)
            np.testing.assert_array_equal(executed[:6], action[:6])
            self.assertEqual(executed[6], GRIPPER_CLOSE)

    def test_window_edges_are_inclusive_and_exclusive_outside(self):
        action = trajectory([-1.0] * 4).tolist()[-1]
        action = np.array(action)
        self.assertTrue(override_gripper(action, 55, (56, 75))[1] is False)
        self.assertTrue(override_gripper(action, 56, (56, 75))[1] is True)
        self.assertTrue(override_gripper(action, 75, (56, 75))[1] is True)
        self.assertTrue(override_gripper(action, 76, (56, 75))[1] is False)

    def test_archived_open_run_is_fully_converted_to_closed(self):
        # State 33 under P opens at action 56 and never closes again.
        archived = trajectory([-1.0] * 10, np.zeros((10, 6)))
        held = np.stack([override_gripper(row, i, (56, 75))[0] for i, row in enumerate(archived)])
        self.assertTrue((held[55:, 6] == GRIPPER_CLOSE).all())
        np.testing.assert_array_equal(held[56:, :6], archived[56:, :6])
        self.assertEqual(int((held[:56, 6] != archived[:56, 6]).sum()), 0)

    def test_input_action_is_never_mutated(self):
        action = np.array([.1, .2, .3, .04, .05, .06, -1.0])
        override_gripper(action, 60, (56, 75))
        self.assertEqual(action[6], -1.0)

    def test_non_binarised_gripper_is_rejected(self):
        action = np.array([.1, .2, .3, .04, .05, .06, 0.37])
        with self.assertRaises(ValueError):
            override_gripper(action, 60, (56, 75))
        with self.assertRaises(ValueError):
            override_gripper(np.zeros(6), 60, (56, 75))

    def test_disabled_window_leaves_every_step_untouched(self):
        action = np.array([.1, .2, .3, .04, .05, .06, -1.0])
        executed, overridden = override_gripper(action, 60, None)
        self.assertFalse(overridden)
        np.testing.assert_array_equal(executed, action)


class DivergenceTests(unittest.TestCase):
    def test_gripper_only_difference_is_not_reported_as_divergence(self):
        archived = trajectory([-1.0] * 6)
        held = np.stack([override_gripper(row, i, (2, 3))[0] for i, row in enumerate(archived)])
        self.assertIsNone(first_divergence_step(held, archived))

    def test_first_arm_difference_is_reported(self):
        archived = trajectory([-1.0] * 6)
        changed = archived.copy()
        changed[4, 2] += 0.01
        self.assertEqual(first_divergence_step(changed, archived), 4)

    def test_without_reference_there_is_nothing_to_compare(self):
        self.assertIsNone(first_divergence_step(trajectory([1.0]), None))

    def test_shorter_run_compares_only_the_shared_prefix(self):
        archived = trajectory([-1.0] * 6)
        self.assertIsNone(first_divergence_step(archived[:4], archived))

    def test_malformed_reference_rejected(self):
        with self.assertRaises(ValueError):
            first_divergence_step(trajectory([1.0]), np.zeros((4, 6)))


class ChunkTests(unittest.TestCase):
    def test_rows_carry_goal_provenance_and_the_full_block(self):
        block = np.arange(16, dtype=float).reshape(2, 8)
        rows = chunk_records([dict(step=56, goal_source='cached', goal_horizon_rows=40.0,
                                   seed=7, chunk_position=0, actions=block)])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row['query_step'], row['goal_source'], row['chunk_position']), (56, 'cached', 0))
        self.assertEqual(len(json.loads(row['chunk_actions'])), 2)

    def test_empty_query_list_yields_no_rows(self):
        self.assertEqual(chunk_records([]), [])


class SlipTests(unittest.TestCase):
    def test_first_state_has_no_predecessor(self):
        self.assertEqual(slip_series([[0., 0., 0.]]).tolist(), [0.0])

    def test_slip_is_the_norm_of_the_offset_change(self):
        offsets = [[0., 0., 0.], [0.003, 0., 0.], [0.003, 0.004, 0.]]
        np.testing.assert_allclose(slip_series(offsets), [0.0, 3.0, 4.0])


class SubsetProtocolTests(unittest.TestCase):
    def setUp(self):
        self.parent = dict(schema_version=1, suite='libero_goal', seed=42, task_ids=[8],
                           episode_ids=[25, 32, 33, 44],
                           conditions={'clean': dict(brightness=1., saturation=1., gains=[1., 1., 1.]),
                                       'color_strong': dict(brightness=1., saturation=1., gains=[.7, 1., 1.3])})

    def test_derived_protocol_keeps_the_single_episode(self):
        derived = subset_protocol(self.parent, 33)
        self.assertEqual(derived['episode_ids'], [33])
        self.assertEqual(set(derived['conditions']), set(self.parent['conditions']))

    def test_per_episode_inputs_do_not_depend_on_the_episode_list(self):
        parent = subset_protocol(self.parent, 33)
        derived = subset_protocol(self.parent, 33, condition='color_strong')
        for protocol in (parent, derived):
            self.assertEqual(stable_seed(protocol['seed'], protocol['suite'], 8, 33) % (2 ** 31), 474245375)
        self.assertEqual(parent['conditions']['color_strong'], derived['conditions']['color_strong'])

    def test_condition_subset_keeps_the_corruption_exact(self):
        self.assertEqual(subset_protocol(self.parent, 33, condition='color_strong')
                         ['conditions']['color_strong']['gains'], [.7, 1., 1.3])

    def test_hash_changes_so_a_subset_is_never_confused_with_the_suite(self):
        self.assertNotEqual(protocol_hash(self.parent), protocol_hash(subset_protocol(self.parent, 33)))

    def test_episode_outside_the_parent_is_rejected(self):
        with self.assertRaises(ValueError):
            subset_protocol(self.parent, 40)
        with self.assertRaises(ValueError):
            subset_protocol(self.parent, 33, condition='dark')


if __name__ == '__main__':
    unittest.main()