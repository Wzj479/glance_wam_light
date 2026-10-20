import json
import tempfile
import unittest
from pathlib import Path

from analyze_flip_rates import (attribution, flip_summary, read_episodes,
                            state_stability)


def record(episode_id, success, steps, sha=None):
    return dict(task_id=8, episode_id=episode_id, success=success, steps=steps,
                env_seed=1, init_state_sha256='i', post_settle_state_sha256='s',
                action_sha256=sha or f'{episode_id}-{success}-{steps}')


class FlipSummaryTests(unittest.TestCase):
    def test_success_rate_and_trace_count_come_from_the_repetitions(self):
        runs = [
            {26: record(26, True, 84, 'a'), 33: record(33, True, 80, 'b')},
            {26: record(26, True, 84, 'c'), 33: record(33, False, 300, 'd')},
            {26: record(26, True, 85, 'e'), 33: record(33, True, 81, 'f')},
        ]
        summary = flip_summary(runs)
        self.assertEqual(summary[26]['success_rate'], 1.0)
        self.assertEqual(summary[26]['distinct_action_sha256'], 3)
        self.assertEqual(summary[26]['step_min'], 84)
        self.assertEqual(summary[26]['step_max'], 85)
        self.assertEqual(summary[33]['successes'], 2)
        self.assertEqual(summary[33]['failures'], 1)
        self.assertEqual(summary[33]['success_rate'], 2 / 3)

    def test_archive_outcome_is_compared_per_episode(self):
        runs = [{26: record(26, False, 300, 'a')}, {26: record(26, True, 84, 'b')}]
        archive = {26: record(26, False, 300, 'z')}
        summary = flip_summary(runs, archive)
        self.assertFalse(summary[26]['archived_success'])
        self.assertEqual(summary[26]['matches_archive'], 1)
        self.assertEqual(summary[26]['archive_agreement_rate'], 0.5)

    def test_episode_missing_from_the_archive_is_reported_without_it(self):
        summary = flip_summary([{26: record(26, True, 84)}], {33: record(33, True, 80)})
        self.assertNotIn('archived_success', summary[26])


class AttributionTests(unittest.TestCase):
    def test_a_flipped_episode_is_not_attributed_to_the_model(self):
        summary = {
            'P': {26: {'successes': 2, 'failures': 1, 'runs': 3},
                  28: {'successes': 0, 'failures': 3, 'runs': 3}},
        }
        result = attribution(summary)['P']
        self.assertEqual(result['flipping_episodes'], [26])
        self.assertEqual(result['stable_failure_episodes'], [28])
        self.assertEqual(result['stable_failure_count'], 1)

    def test_a_consistent_success_is_neither_failure_nor_flip(self):
        summary = {'P': {26: {'successes': 3, 'failures': 0, 'runs': 3}}}
        result = attribution(summary)['P']
        self.assertEqual(result['flipping_episodes'], [])
        self.assertEqual(result['stable_failure_episodes'], [])
        self.assertEqual(result['stable_success_episodes'], [26])


class StateStabilityTests(unittest.TestCase):
    def test_a_state_where_every_model_agrees_is_stable(self):
        summary = {
            'base': {28: {'successes': 0, 'failures': 5, 'runs': 5}},
            'P': {28: {'successes': 0, 'failures': 5, 'runs': 5}},
        }
        state = state_stability(summary)[28]
        self.assertTrue(state['stable_overall'])
        self.assertTrue(state['all_models_agree'])
        self.assertTrue(state['shared_failure'])
        self.assertEqual(state['flipping_models'], [])

    def test_one_flipping_model_makes_the_state_unstable(self):
        summary = {
            'base': {33: {'successes': 5, 'failures': 0, 'runs': 5}},
            'P': {33: {'successes': 3, 'failures': 2, 'runs': 5}},
        }
        state = state_stability(summary)[33]
        self.assertFalse(state['stable_overall'])
        self.assertEqual(state['flipping_models'], ['P'])
        self.assertFalse(state['all_models_agree'])
        self.assertFalse(state['shared_failure'])

    def test_models_may_agree_on_failure_while_one_flips_on_success(self):
        summary = {
            'base': {26: {'successes': 4, 'failures': 1, 'runs': 5}},
            'C': {26: {'successes': 5, 'failures': 0, 'runs': 5}},
        }
        state = state_stability(summary)[26]
        self.assertFalse(state['stable_overall'])
        self.assertEqual(state['flipping_models'], ['base'])


class ReadEpisodesTests(unittest.TestCase):
    def test_a_run_without_results_json_is_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_episodes(Path(tmp) / 'results.json'), {})

    def test_episode_ids_become_ints_and_traces_are_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.json'
            payload = dict(episode_results=[
                dict(record(26, True, 84), task_id=8),
                dict(record(33, False, 300), task_id=8),
            ])
            path.write_text(json.dumps(payload), encoding='utf-8')
            (Path(tmp) / 'task_08_episode_026_actions.npy').write_bytes(b'first')
            (Path(tmp) / 'task_08_episode_033_actions.npy').write_bytes(b'second')
            rows = read_episodes(path)
        self.assertEqual(sorted(rows), [26, 33])
        self.assertTrue(rows[26]['success'])
        self.assertIsNotNone(rows[26]['action_sha256'])
        self.assertNotEqual(rows[26]['action_sha256'], rows[33]['action_sha256'])

    def test_a_missing_trace_file_leaves_the_hash_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.json'
            path.write_text(json.dumps(dict(episode_results=[dict(record(26, True, 84), task_id=8)])),
                            encoding='utf-8')
            rows = read_episodes(path)
        self.assertIsNone(rows[26]['action_sha256'])


if __name__ == '__main__':
    unittest.main()
