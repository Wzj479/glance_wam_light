import json
import tempfile
import unittest
from pathlib import Path

from analyze_intervention_reps import collect, load_baseline, read_run, summarise, verdict


def write_run(root, state, model, rep, success, steps=80, overridden=20):
    run = Path(root) / f'state{state}_{model}_r{rep}' / 'driver'
    run.mkdir(parents=True)
    (run / 'results.json').write_text(json.dumps(dict(episode_results=[dict(
        episode_id=state, steps=steps, success=success,
        gripper_overridden_steps=overridden)])), encoding='utf-8')


class ReadRunTests(unittest.TestCase):
    def test_a_missing_results_file_is_not_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_run(tmp))

    def test_the_lap_record_is_unpacked(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, 28, 'base', 1, True, steps=71, overridden=20)
            run = read_run(Path(tmp) / 'state28_base_r1')
        self.assertEqual(run['episode_id'], 28)
        self.assertEqual(run['steps'], 71)
        self.assertTrue(run['success'])
        self.assertEqual(run['overridden_steps'], 20)


class CollectTests(unittest.TestCase):
    def test_runs_are_grouped_by_state_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, 28, 'base', 1, True)
            write_run(tmp, 28, 'base', 2, True)
            write_run(tmp, 28, 'C', 1, False)
            grouped = collect(tmp)
        self.assertEqual(sorted(grouped), [(28, 'C'), (28, 'base')])
        self.assertEqual(sorted(grouped[(28, 'base')]), [1, 2])

    def test_unrelated_directories_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'notes').mkdir()
            self.assertEqual(collect(tmp), {})


class VerdictTests(unittest.TestCase):
    def test_a_baseline_that_always_failed_is_recovered(self):
        self.assertEqual(verdict(5, 5, 0, 5), 'recovered')
        self.assertEqual(verdict(0, 5, 0, 5), 'still_failing')
        self.assertEqual(verdict(3, 5, 0, 5), 'partial')

    def test_a_baseline_that_always_succeeded_can_only_be_kept_or_damaged(self):
        self.assertEqual(verdict(5, 5, 5, 5), 'unchanged')
        self.assertEqual(verdict(0, 5, 5, 5), 'damaged')
        self.assertEqual(verdict(2, 5, 5, 5), 'mixed')

    def test_an_unstable_baseline_supports_no_verdict(self):
        self.assertEqual(verdict(5, 5, 3, 5), 'baseline_unstable')


class SummariseTests(unittest.TestCase):
    def test_successes_runs_and_verdict_are_reported(self):
        grouped = {(28, 'base'): {
            1: dict(episode_id=28, steps=71, success=True, overridden_steps=20),
            2: dict(episode_id=28, steps=71, success=True, overridden_steps=20),
            3: dict(episode_id=28, steps=300, success=False, overridden_steps=20),
        }}
        baseline = {str(28): {'base': dict(successes=0, runs=5)}}
        out = summarise(grouped, baseline)['28:base']
        self.assertEqual(out['runs'], 3)
        self.assertEqual(out['successes'], 2)
        self.assertEqual(out['steps'], [71, 71, 300])
        self.assertEqual(out['verdict'], 'partial')
        self.assertEqual(out['overridden_steps'], [20])

    def test_control_state_that_survives_is_unchanged(self):
        grouped = {(25, 'base'): {
            1: dict(episode_id=25, steps=77, success=True, overridden_steps=20),
            2: dict(episode_id=25, steps=77, success=True, overridden_steps=20),
        }}
        baseline = {str(25): {'base': dict(successes=5, runs=5)}}
        self.assertEqual(summarise(grouped, baseline)['25:base']['verdict'], 'unchanged')


class LoadBaselineTests(unittest.TestCase):
    def test_baseline_is_read_per_state_for_one_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'map.json'
            path.write_text(json.dumps(dict(episodes={
                'base': {'28': dict(successes=0, runs=5), '25': dict(successes=5, runs=5)},
                'C': {'36': dict(successes=5, runs=5)}})), encoding='utf-8')
            out = load_baseline(path)
        self.assertEqual(out['28']['base'], dict(successes=0, runs=5))
        self.assertEqual(out['36']['C'], dict(successes=5, runs=5))


if __name__ == '__main__':
    unittest.main()
