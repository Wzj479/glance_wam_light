"""Tests for the expansion-against-archive determinism check; no simulator, model or GPU."""
import json
import tempfile
import unittest
from pathlib import Path

from verify_expansion_against_archive import compare_pair, render, verify

SHA_A = 'a' * 64
SHA_B = 'b' * 64


def record(episode_id, success=True, steps=300, env_seed=474245375,
           init=SHA_A, post=SHA_B):
    return dict(task_id=8, episode_id=episode_id, env_seed=env_seed,
                init_state_sha256=init, post_settle_state_sha256=post,
                success=success, steps=steps)


def write_run(root, protocol_sha256, models, conditions, episodes):
    """Lay out one evaluation directory in the shape run_robust_eval.py produces."""
    for model in models:
        for condition in conditions:
            target = Path(root) / model / condition
            target.mkdir(parents=True, exist_ok=True)
            (target / 'results.json').write_text(json.dumps(dict(
                protocol_sha256=protocol_sha256,
                episode_results=[record(ep) for ep in episodes])))


class ComparePairTests(unittest.TestCase):
    def test_identical_records_have_no_differences(self):
        physical, outcome = compare_pair(record(33), record(33))
        self.assertEqual((physical, outcome), ([], []))

    def test_physical_and_outcome_differences_are_reported_separately(self):
        physical, outcome = compare_pair(record(33), record(33, success=False, steps=120))
        self.assertEqual(physical, [])
        self.assertEqual(outcome, ['success', 'steps'])

    def test_changed_initial_state_is_physical_not_outcome(self):
        physical, outcome = compare_pair(record(33), record(33, init=SHA_B))
        self.assertEqual(physical, ['init_state_sha256'])
        self.assertEqual(outcome, [])


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def build(self, archive_episodes, expansion_episodes, archive_hash='h-archive',
              expansion_hash='h-expansion'):
        write_run(self.root / 'archive', archive_hash, ['base'], ['color_strong'],
                  archive_episodes)
        write_run(self.root / 'expansion', expansion_hash, ['base'], ['color_strong'],
                  expansion_episodes)
        return (str(self.root / 'archive'), str(self.root / 'expansion'),
                ['base'], ['color_strong'])

    def test_reproduced_when_shared_episodes_agree(self):
        archive, expansion, models, conditions = self.build(range(25, 30), range(0, 30))
        report = verify(archive, expansion, models, conditions)
        self.assertTrue(report['reproduced'])
        self.assertEqual(report['differences'], [])
        self.assertEqual(report['shared_episodes']['base/color_strong'], 5)
        self.assertEqual(report['new_episodes']['base/color_strong'], list(range(25)))
        self.assertIn('REPRODUCED', render(report))

    def test_widened_protocol_is_not_reported_as_a_defect(self):
        archive, expansion, models, conditions = self.build(range(25, 30), range(0, 30))
        report = verify(archive, expansion, models, conditions)
        self.assertTrue(report['reproduced'])
        self.assertNotEqual(report['protocol_sha256']['archive'],
                            report['protocol_sha256']['expansion'])
        self.assertIn('widened', report['protocol_note'])

    def test_success_flips_mark_the_run_as_not_reproduced(self):
        write_run(self.root / 'archive', 'h-archive', ['base'], ['color_strong'], range(25, 30))
        records = [record(ep) for ep in range(0, 30)]
        records[25]['success'] = False
        target = self.root / 'expansion' / 'base' / 'color_strong'
        target.mkdir(parents=True)
        (target / 'results.json').write_text(json.dumps(
            dict(protocol_sha256='h-expansion', episode_results=records)))
        report = verify(str(self.root / 'archive'), str(self.root / 'expansion'),
                        ['base'], ['color_strong'])
        self.assertFalse(report['reproduced'])
        self.assertEqual(report['differences'][0]['episode_id'], 25)
        self.assertEqual(report['differences'][0]['outcome'], ['success'])
        self.assertIn('NOT REPRODUCED', render(report))

    def test_nondeterministic_step_count_is_caught(self):
        records = [record(ep) for ep in range(0, 30)]
        records[28]['steps'] = 299
        write_run(self.root / 'archive', 'h-archive', ['base'], ['color_strong'], range(25, 30))
        target = self.root / 'expansion' / 'base' / 'color_strong'
        target.mkdir(parents=True)
        (target / 'results.json').write_text(json.dumps(
            dict(protocol_sha256='h-expansion', episode_results=records)))
        report = verify(str(self.root / 'archive'), str(self.root / 'expansion'),
                        ['base'], ['color_strong'])
        self.assertFalse(report['reproduced'])
        self.assertEqual(report['differences'][0]['outcome'], ['steps'])

    def test_disjoint_episode_sets_are_refused_rather_than_called_reproduced(self):
        write_run(self.root / 'archive', 'h-archive', ['base'], ['color_strong'], range(25, 45))
        write_run(self.root / 'expansion', 'h-expansion', ['base'], ['color_strong'], range(0, 20))
        with self.assertRaises(ValueError):
            verify(str(self.root / 'archive'), str(self.root / 'expansion'),
                   ['base'], ['color_strong'])

    def test_missing_results_file_names_the_path(self):
        archive, expansion, models, conditions = self.build(range(25, 30), range(0, 30))
        (Path(expansion) / 'base' / 'color_strong' / 'results.json').unlink()
        with self.assertRaises(FileNotFoundError):
            verify(archive, expansion, models, conditions)

    def test_results_without_episode_records_are_refused(self):
        archive, expansion, models, conditions = self.build(range(25, 30), range(0, 30))
        target = Path(expansion) / 'base' / 'color_strong' / 'results.json'
        target.write_text(json.dumps({'protocol_sha256': 'h-expansion'}))
        with self.assertRaises(ValueError):
            verify(archive, expansion, models, conditions)


if __name__ == '__main__':
    unittest.main()