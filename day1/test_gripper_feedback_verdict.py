"""Tests for the closed-loop verdict logic and its schema compatibility."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_contact_transfer import analyze as contact_transfer_analyze
from analyze_gripper_feedback import (
    LIFT_MARKER_MM, actions_match, analyse_run, contact_tail, grasp_profile, overlay_profile,
    slip_profile)
from diagnose_gripper_feedback import slip_series, state_record

FINGER = ('left_pad_contact', 'right_pad_contact', 'both_pad_contact',
          'left_finger_contact', 'right_finger_contact', 'any_finger_contact',
          'finger_contact_without_both_pads')


def state_rows(lift_mm, contact, bowl_xy=(0.0, 0.0), plate_xy=(0.0, 0.0), plate_contact=None):
    """Synthetic state rows in the closed-loop schema.

    ``contact`` is gripper-to-bowl contact. ``plate_contact`` is bowl-to-plate contact,
    which is independent of it and defaults to the gripper pattern so simple cases need
    only one argument.
    """
    plate_contact = contact if plate_contact is None else plate_contact
    rows = []
    for index, (height, touching, resting) in enumerate(zip(lift_mm, contact, plate_contact)):
        bowl = [bowl_xy[0], bowl_xy[1], height / 1000]
        plate = [plate_xy[0], plate_xy[1], 0.0]
        xy = float(((bowl[0] - plate[0]) ** 2 + (bowl[1] - plate[1]) ** 2) ** 0.5)
        touching_plate = resting and bowl[2] >= plate[2] and xy < 0.03
        row = {'state_step': index, 'time_s': index / 20,
               'eef_x': 0.0, 'eef_y': 0.0, 'eef_z': 0.9,
               'eef_qx': 0.0, 'eef_qy': 0.0, 'eef_qz': 0.0, 'eef_qw': 1.0,
               'bowl_x': bowl[0], 'bowl_y': bowl[1], 'bowl_z': bowl[2],
               'plate_x': plate[0], 'plate_y': plate[1], 'plate_z': 0.0,
               'finger_width_m': 0.01,
               'left_pad_contact': touching, 'right_pad_contact': touching,
               'both_pad_contact': touching,
               'left_finger_contact': touching, 'right_finger_contact': touching,
               'any_finger_contact': touching,
               'finger_contact_without_both_pads': False,
               'bowl_in_eef_x_m': 0.0, 'bowl_in_eef_y_m': 0.0, 'bowl_in_eef_z_m': 0.0,
               'relative_slip_mm': 0.0,
               'bowl_plate_contact': touching_plate,
               'xy_distance_m': xy,
               'above_plate': bowl[2] >= plate[2], 'xy_within_30mm': xy < 0.03,
               'success': touching_plate}
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ContactTailTests(unittest.TestCase):
    def test_tail_measures_the_final_gap_not_the_first_one(self):
        flags = np.array([True] * 10 + [False] * 3 + [True] * 7)
        first, last, tail = contact_tail(flags)
        self.assertEqual((first, last, tail), (0, 19, 0))
        self.assertEqual(contact_tail(np.zeros(5, dtype=bool)), (None, None, None))

    def test_trailing_gap_is_reported_in_states(self):
        flags = np.array([True] * 4 + [False] * 6)
        self.assertEqual(contact_tail(flags)[2], 6)


class GraspVerdictTests(unittest.TestCase):
    def test_transport_needs_lift_before_contact_is_lost(self):
        rows = state_rows([0] * 10 + [40] * 10 + [0] * 5, [True] * 15 + [False] * 10)
        profile = grasp_profile(rows)
        self.assertTrue(profile['reached_lift_marker_before_last_finger_contact'])
        self.assertEqual(profile['lift_at_last_finger_contact_mm'], 40.0)

    def test_archived_shape_never_transports(self):
        # The archived state33 P pattern: contact ends while the bowl is barely off
        # the table, so no lift marker is ever cleared before release.
        lift = [0, 2, 4, 4.29, 4, 3, 0, 0, 0]
        rows = state_rows(lift, [False, True, True, True, True, False, False, False, False])
        profile = grasp_profile(rows)
        self.assertFalse(profile['reached_lift_marker_before_last_finger_contact'])
        self.assertIsNone(profile['first_state_above_lift_marker'])
        self.assertAlmostEqual(profile['max_lift_mm'], 4.29, places=2)
        self.assertAlmostEqual(profile['lift_at_last_finger_contact_mm'], 4.0, places=2)
        self.assertEqual(profile['last_finger_contact_state'], 4)

    def test_lift_after_release_does_not_count_as_transport(self):
        rows = state_rows([0] * 10 + [60] * 10, [True] * 5 + [False] * 15)
        self.assertFalse(grasp_profile(rows)['reached_lift_marker_before_last_finger_contact'])

    def test_contact_chatter_is_visible(self):
        rows = state_rows([50] * 12, [True, False] * 6)
        self.assertEqual(grasp_profile(rows)['finger_contact_regions'], 6)


class OverlayTests(unittest.TestCase):
    def test_final_overlay_state_decides_success(self):
        rows = state_rows([0] * 5, [False] * 5, bowl_xy=(0.01, 0.0), plate_contact=[True] * 5)
        profile = overlay_profile(rows)
        self.assertTrue(profile['state_log_success'] and profile['final_within_30mm'])
        self.assertAlmostEqual(profile['min_xy_mm'], 10.0, places=6)

    def test_far_bowl_is_reported_as_failure(self):
        rows = state_rows([0] * 5, [False] * 5, bowl_xy=(0.137, 0.0), plate_contact=[True] * 5)
        self.assertFalse(overlay_profile(rows)['state_log_success'])


class ControlGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, actions):
        run = self.root / 'run'
        (run / 'driver').mkdir(parents=True)
        np.save(run / 'driver' / 'task_08_episode_033_actions.npy', actions)
        return run

    def test_identical_control_reproduces_the_archive(self):
        archived = np.random.default_rng(0).normal(size=(12, 7))
        reference = self.root / 'ref.npy'
        np.save(reference, archived)
        report = actions_match(self._run(archived.copy()), reference)
        self.assertTrue(report['reproduced'])
        self.assertNotIn('first_divergence_step', report)

    def test_single_different_command_blocks_attribution(self):
        archived = np.zeros((12, 7))
        archived[:, 6] = 1.0
        reference = self.root / 'ref.npy'
        np.save(reference, archived)
        altered = archived.copy()
        altered[4, 2] += 1e-3
        report = actions_match(self._run(altered), reference)
        self.assertFalse(report['reproduced'])
        self.assertEqual(report['first_divergence_step'], 4)

    def test_shortened_run_is_not_silently_accepted(self):
        archived = np.zeros((12, 7))
        reference = self.root / 'ref.npy'
        np.save(reference, archived)
        report = actions_match(self._run(archived[:5]), reference)
        self.assertFalse(report['reproduced'])
        self.assertEqual(report['reason'], 'step_count_differs')


class SchemaCompatibilityTests(unittest.TestCase):
    def test_closed_loop_log_is_readable_by_the_frozen_metric_script(self):
        # One set of definitions must serve the offline replays and the closed loop.
        rows = state_rows([0, 10, 30, 50, 50], [False, True, True, True, True])
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / 'hold'
            case.mkdir()
            write_csv(case / 'states.csv', rows)
            metrics = contact_transfer_analyze(case / 'states.csv')
        self.assertTrue(metrics['finger_contact_columns_present'])
        self.assertEqual(metrics['states_with_any_finger_contact'], 4)
        self.assertGreater(metrics['max_lift_mm'], LIFT_MARKER_MM)

    def test_slip_column_matches_the_frozen_analysis_convention(self):
        # The replay script logs slip in world axes while the analysis recomputes it in
        # the EEF frame. The closed-loop log must follow the analysis convention, or the
        # two report different numbers for the same motion.
        rows = state_rows([0, 5, 15, 30], [True] * 4)
        for index, angle in enumerate((0.0, 0.2, 0.5, 0.9)):
            half = angle / 2.0
            rows[index].update(eef_qx=0.0, eef_qy=0.0, eef_qz=float(np.sin(half)), eef_qw=float(np.cos(half)))
            rows[index]['bowl_x'] = 0.02 * index
            rows[index]['bowl_z'] = 0.9 + 0.01 * index
        from scipy.spatial.transform import Rotation
        offsets = np.einsum('nji,nj->ni', Rotation.from_quat(
            np.asarray([[r[f'eef_q{a}'] for a in 'xyzw'] for r in rows])).as_matrix(),
            np.asarray([[r[f'bowl_{a}'] - r[f'eef_{a}'] for a in 'xyz'] for r in rows]))
        for index, value in enumerate(slip_series(offsets)):
            rows[index]['relative_slip_mm'] = float(value)
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / 'held'
            case.mkdir()
            write_csv(case / 'states.csv', rows)
            frozen = contact_transfer_analyze(case / 'states.csv')
        mine = slip_profile(rows)
        self.assertAlmostEqual(frozen['max_relative_slip_mm'], mine['max_relative_slip_mm'], places=9)
        self.assertGreater(mine['max_relative_slip_mm'], 0.0)

    def test_recorded_offset_and_slip_come_from_the_same_series(self):
        offsets = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.001, 0.002, 0.0]])
        slip = slip_series(offsets)
        np.testing.assert_allclose(slip, [0.0, 1.0, 2.0])


class EndToEndReportTests(unittest.TestCase):
    def test_report_is_built_from_logs_without_a_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'run' / 'driver'
            driver.mkdir(parents=True)
            rows = state_rows([0] * 5 + [45] * 8, [False] + [True] * 10 + [False] * 2)
            write_csv(driver / 'task_08_episode_033_state_log.csv', rows)
            write_csv(driver / 'task_08_episode_033_override_log.csv', [
                dict(action_index=k, overridden=(56 <= k <= 75),
                     model_gripper_command=(-1 if 56 <= k <= 75 else 1),
                     executed_gripper_command=1) for k in range(80)])
            write_csv(driver / 'task_08_episode_033_chunk_log.csv', [
                dict(query_index=i, query_step=i * 8, goal_source='generate' if i == 0 else 'cached',
                     goal_horizon_rows=48.0, seed=1, chunk_position=0, chunk_actions='[]')
                for i in range(2)])
            episode = dict(task_id=8, episode_id=33, steps=13, success=False,
                           action_sha256='x', init_state_sha256='y', post_settle_state_sha256='z',
                           divergence_step_vs_reference=64)
            (driver / 'results.json').write_text(json.dumps(dict(
                model='P', condition='color_strong', hold_close_inclusive=[56, 75],
                protocol_sha256='p', successes=0, episodes=1, success_rate=0.0,
                episode_results=[episode])))
            report = analyse_run(root / 'run')
        self.assertEqual(report['hold_close_inclusive'], [56, 75])
        self.assertEqual(report['divergence_step_vs_reference'], 64)
        self.assertEqual(report['goal_sources'], ['cached', 'generate'])
        self.assertEqual(report['gripper_override_first_step'], 56)
        self.assertEqual(report['model_gripper_commands_seen'], [-1, 1])
        self.assertEqual(report['verdict'], 'transport_restored')

    def test_float_gripper_channel_survives_a_csv_round_trip(self):
        # The real writer stores the channel as a float, so a round trip through
        # the CSV yields the text '-1.0'. Reading it back must not raise.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'run' / 'driver'
            driver.mkdir(parents=True)
            rows = state_rows([0] * 5 + [45] * 8, [False] + [True] * 10 + [False] * 2)
            write_csv(driver / 'task_08_episode_033_state_log.csv', rows)
            write_csv(driver / 'task_08_episode_033_override_log.csv', [
                dict(action_index=k, overridden=(56 <= k <= 75),
                     model_gripper_command=float(-1 if 56 <= k <= 75 else 1),
                     executed_gripper_command=float(1)) for k in range(80)])
            write_csv(driver / 'task_08_episode_033_chunk_log.csv', [
                dict(query_index=i, query_step=i * 8, goal_source='generate' if i == 0 else 'cached',
                     goal_horizon_rows=48.0, seed=1, chunk_position=0, chunk_actions='[]')
                for i in range(2)])
            episode = dict(task_id=8, episode_id=33, steps=13, success=False,
                           action_sha256='x', init_state_sha256='y', post_settle_state_sha256='z',
                           divergence_step_vs_reference=64)
            (driver / 'results.json').write_text(json.dumps(dict(
                model='P', condition='color_strong', hold_close_inclusive=[56, 75],
                protocol_sha256='p', successes=0, episodes=1, success_rate=0.0,
                episode_results=[episode])))
            report = analyse_run(root / 'run')
        self.assertEqual(report['model_gripper_commands_seen'], [-1, 1])
        self.assertEqual(report['gripper_override_first_step'], 56)


class OutcomeAxisTests(unittest.TestCase):
    def test_lifting_the_bowl_is_not_the_same_as_placing_it(self):
        # 39/aug lifted 100mm and still failed; the outcome must say so.
        rows = state_rows([0] * 5 + [100] * 8, [False] + [True] * 10 + [False] * 2,
                          bowl_xy=(0.14, 0.0), plate_contact=[False] * 13)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'run' / 'driver'
            driver.mkdir(parents=True)
            write_csv(driver / 'task_08_episode_033_state_log.csv', rows)
            write_csv(driver / 'task_08_episode_033_override_log.csv', [
                dict(action_index=k, overridden=(56 <= k <= 75), model_gripper_command=-1,
                     executed_gripper_command=1) for k in range(80)])
            write_csv(driver / 'task_08_episode_033_chunk_log.csv', [
                dict(query_index=0, query_step=0, goal_source='generate', goal_horizon_rows=48.0,
                     seed=1, chunk_position=0, chunk_actions='[]')])
            episode = dict(task_id=8, episode_id=33, steps=13, success=False, action_sha256='x',
                           init_state_sha256='y', post_settle_state_sha256='z',
                           divergence_step_vs_reference=64)
            (driver / 'results.json').write_text(json.dumps(dict(
                model='P', condition='color_strong', hold_close_inclusive=[56, 75], protocol_sha256='p',
                successes=0, episodes=1, success_rate=0.0, episode_results=[episode])))
            report = analyse_run(root / 'run')
        self.assertEqual(report['verdict'], 'transport_restored')
        self.assertEqual(report['outcome'], 'transported_not_placed')

    def test_placing_without_crossing_the_marker_is_not_transported(self):
        rows = state_rows([0] * 5 + [8] * 8, [False] + [True] * 10 + [False] * 2,
                          plate_contact=[True] * 13)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'run' / 'driver'
            driver.mkdir(parents=True)
            write_csv(driver / 'task_08_episode_033_state_log.csv', rows)
            write_csv(driver / 'task_08_episode_033_override_log.csv', [])
            write_csv(driver / 'task_08_episode_033_chunk_log.csv', [])
            episode = dict(task_id=8, episode_id=33, steps=13, success=True, action_sha256='x',
                           init_state_sha256='y', post_settle_state_sha256='z',
                           divergence_step_vs_reference=None)
            (driver / 'results.json').write_text(json.dumps(dict(
                model='P', condition='color_strong', hold_close_inclusive=None, protocol_sha256='p',
                successes=1, episodes=1, success_rate=1.0, episode_results=[episode])))
            report = analyse_run(root / 'run')
        self.assertEqual(report['outcome'], 'not_transported')
        self.assertTrue(report['success_predicates_agree'])


if __name__ == '__main__':
    unittest.main()