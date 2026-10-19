"""Tests for the failure-attribution inventory; no simulator, model or GPU."""
import unittest

import numpy as np

from inventory_failure_structure import (
    UNIT_NOTE, attribute, gripper_timeline, motion_profile, motion_separation)


def actions(gripper, translations=None):
    count = len(gripper)
    arm = np.zeros((count, 6)) if translations is None else np.asarray(translations, dtype=float)
    return np.concatenate([arm, np.asarray(gripper, dtype=float).reshape(-1, 1)], axis=1)


def from_steps(gripper, steps):
    """Build an action array whose per-step translation magnitudes equal ``steps``."""
    positions = [[0.0, 0.0, 0.0]]
    for step in steps:
        positions.append([positions[-1][0], positions[-1][1], positions[-1][2] + float(step)])
    return actions(gripper, positions)


def episode(model, condition, episode_id, success, gripper, translations=None):
    close, opened = gripper_timeline(actions(gripper))
    median, peak = motion_profile(actions(gripper, translations))
    return dict(model=model, condition=condition, task_id=8, episode_id=episode_id,
                success=success, steps=len(gripper), gripper_close_steps=close,
                gripper_open_steps=opened, median_command_mag=median, peak_command_mag=peak)


class GripperTimelineTests(unittest.TestCase):
    def test_flips_are_reported_as_step_indices(self):
        # Archived state33 under P: closes at 44 and opens at 56.
        grip = [-1] * 44 + [1] * 12 + [-1] * 5
        close, opened = gripper_timeline(actions(grip))
        self.assertEqual(close, [44])
        self.assertEqual(opened, [56])

    def test_no_transition_yields_empty_lists(self):
        self.assertEqual(gripper_timeline(actions([1] * 10)), ([], []))
        self.assertEqual(gripper_timeline(actions([-1] * 10)), ([], []))

    def test_chattering_gripper_is_visible_as_repeated_transitions(self):
        grip = [1, -1, 1, -1, 1]
        close, opened = gripper_timeline(actions(grip))
        self.assertEqual((close, opened), ([2, 4], [1, 3]))


class MotionProfileTests(unittest.TestCase):
    def test_peak_uses_the_largest_single_step(self):
        median, peak = motion_profile(from_steps([1] * 5, [0.0, 0.9, 0.0, 0.0]))
        self.assertAlmostEqual(peak, 900.0, places=6)
        self.assertAlmostEqual(median, 0.0, places=6)

    def test_single_step_action_has_no_motion(self):
        self.assertEqual(motion_profile(actions([1.0])), (0.0, 0.0))


class AttributionTests(unittest.TestCase):
    def test_state_failing_for_every_model_is_shared(self):
        failures = {('color_strong', 28): ['base', 'C', 'P'], ('color_strong', 33): ['P']}
        shared, specific = attribute(failures, ['base', 'C', 'P'])
        self.assertEqual([(x['condition'], x['episode_id']) for x in shared], [('color_strong', 28)])
        self.assertEqual(specific, {('color_strong', 33): ['P']})

    def test_a_single_model_failure_is_kept_separate_from_shared_ones(self):
        failures = {('color_strong', 26): ['base'], ('color_strong', 34): ['base', 'C', 'P']}
        shared, specific = attribute(failures, ['base', 'C', 'P'])
        self.assertEqual(len(shared), 1)
        self.assertEqual(list(specific.values()), [['base']])


def episode_from_steps(model, condition, episode_id, success, steps, gripper=(1,) * 5):
    """An episode whose translation magnitudes follow ``steps``, rotation zeroed."""
    arm = np.concatenate([from_steps(list(gripper), steps)[:, :3], np.zeros((len(steps) + 1, 3))], axis=1)
    return episode(model, condition, episode_id, success, list(gripper), arm)


class MotionSeparationTests(unittest.TestCase):
    # Successes move steadily; failures hold still and then spike.
    SMOOTH = [0.4, 0.4, 0.4, 0.4]
    SPIKY = [1.2, 0.2, 0.2, 0.2]

    def _rows(self, good, bad, per_side=3):
        rows = []
        for index in range(per_side):
            rows.append(episode_from_steps('P', 'color_strong', index, True, good))
            rows.append(episode_from_steps('P', 'color_strong', 100 + index, False, bad))
        return rows

    def test_spiky_low_median_failures_are_reported_as_separated(self):
        cells, overall = motion_separation(self._rows(self.SMOOTH, self.SPIKY))
        self.assertTrue(cells[0]['separated'])
        self.assertLess(overall['failure_median_command_mag'], overall['success_median_command_mag'])
        self.assertGreater(overall['failure_peak_command_mag'], overall['success_peak_command_mag'])

    def test_reversed_profile_is_not_called_separated(self):
        cells, _ = motion_separation(self._rows(self.SPIKY, self.SMOOTH))
        self.assertFalse(cells[0]['separated'])

    def test_cells_with_too_few_episodes_stay_undecided(self):
        cells, _ = motion_separation(self._rows(self.SMOOTH, self.SPIKY, per_side=1))
        self.assertIsNone(cells[0]['separated'])
        self.assertIn('undecided_because', cells[0])

    def test_pooled_counts_cover_both_outcomes(self):
        _, overall = motion_separation(self._rows(self.SMOOTH, self.SPIKY))
        self.assertEqual(overall['successes'], 3)
        self.assertEqual(overall['failures'], 3)

    def test_unit_note_forbids_reading_magnitudes_as_metres(self):
        self.assertIn('not a verified physical distance', UNIT_NOTE)


if __name__ == '__main__':
    unittest.main()