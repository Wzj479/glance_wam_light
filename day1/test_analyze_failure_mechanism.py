import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_failure_mechanism import (collect, gripper_events, mechanism_facts,
                                      read_states)

COLUMNS = ['state_step', 'time_s', 'eef_x', 'eef_y', 'eef_z',
           'bowl_x', 'bowl_y', 'bowl_z', 'plate_x', 'plate_y', 'plate_z',
           'left_pad_contact', 'right_pad_contact', 'both_pad_contact',
           'any_finger_contact', 'relative_slip_mm', 'bowl_plate_contact',
           'xy_distance_m', 'above_plate', 'xy_within_30mm', 'bowl_tilt_deg',
           'success']


def state(step, bowl_z=0.9, both=False, any_finger=False, xy=0.1, success=False):
    row = dict.fromkeys(COLUMNS, 0.0)
    row.update(state_step=step, time_s=step * 0.05, eef_x=0.0, eef_y=0.0, eef_z=1.0,
               bowl_x=0.1, bowl_y=0.0, bowl_z=bowl_z, plate_x=0.2, plate_y=0.0, plate_z=0.85,
               left_pad_contact=both, right_pad_contact=both, both_pad_contact=both,
               any_finger_contact=any_finger, relative_slip_mm=0.0,
               bowl_plate_contact=False, xy_distance_m=xy, above_plate=bowl_z >= 0.85,
               xy_within_30mm=xy <= 0.03, bowl_tilt_deg=0.0, success=success)
    return row


def actions(gripper_values):
    out = np.zeros((len(gripper_values), 7), dtype=np.float32)
    for index, value in enumerate(gripper_values):
        out[index, 6] = value
    return out


class GripperEventTests(unittest.TestCase):
    def test_first_close_and_first_release_are_transition_indices(self):
        grip = [-1, -1, 1, 1, 1, -1, -1, 1]
        self.assertEqual(gripper_events(actions(grip)), (2, 5))

    def test_a_gripper_that_never_closes_reports_nothing(self):
        self.assertEqual(gripper_events(actions([-1] * 4)), (None, None))


class MechanismTests(unittest.TestCase):
    def test_a_successful_episode_is_labelled_success(self):
        rows = [state(0), state(1, bowl_z=0.95, both=True, any_finger=True, xy=0.01, success=True)]
        facts = mechanism_facts(rows, actions([-1, 1]))
        self.assertEqual(facts['mechanism'], 'success')
        self.assertTrue(facts['success'])

    def test_releasing_before_any_both_pad_contact_is_its_own_mechanism(self):
        rows = [state(0), state(1, any_finger=True), state(2)]
        facts = mechanism_facts(rows, actions([-1, 1, -1]))
        self.assertEqual(facts['mechanism'], 'released_before_both_pad_contact')
        self.assertIsNone(facts['first_both_pad_contact_state'])
        # -1, +1, -1: the close transition is at index 1 and the release at index 2.
        self.assertEqual(facts['first_close_action'], 1)
        self.assertEqual(facts['first_release_action'], 2)

    def test_a_grasp_that_never_lifts_is_labelled_separately(self):
        rows = [state(0), state(1, both=True, any_finger=True), state(2, both=True, any_finger=True)]
        facts = mechanism_facts(rows, actions([-1, 1, 1]))
        self.assertEqual(facts['mechanism'], 'grasped_never_lifted')
        self.assertIsNone(facts['first_lift_20mm_state'])

    def test_never_contacting_the_bowl_at_all(self):
        facts = mechanism_facts([state(0), state(1)], actions([-1, -1]))
        self.assertEqual(facts['mechanism'], 'never_both_pad_contact')

    def test_lifting_then_losing_the_contact_is_distinguished(self):
        rows = [state(0), state(1, bowl_z=0.9, both=True, any_finger=True),
                state(2, bowl_z=0.95, both=True, any_finger=True),
                state(3, bowl_z=0.95)]
        facts = mechanism_facts(rows, actions([-1, 1, 1, 1]))
        self.assertEqual(facts['mechanism'], 'lifted_then_lost_contact')
        self.assertEqual(facts['first_lift_20mm_state'], 2)


class CollectTests(unittest.TestCase):
    def test_collect_reads_every_case_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'replay_plan.json').write_text(json.dumps(dict(cases=[['color_strong', 33]])),
                                                    encoding='utf-8')
            case = root / 'color_strong_episode33'
            model = case / 'P'
            model.mkdir(parents=True)
            header = ','.join(COLUMNS)
            body = ','.join(str(value) for value in state(0).values())
            (model / 'states.csv').write_text(header + '\n' + body + '\n', encoding='utf-8')
            np.savez(model / 'replay.npz', actions=actions([-1, 1]))
            facts = collect(root)
        self.assertIn(('color_strong', 33, 'P'), facts)
        self.assertEqual(facts[('color_strong', 33, 'P')]['steps'], 1)


if __name__ == '__main__':
    unittest.main()
