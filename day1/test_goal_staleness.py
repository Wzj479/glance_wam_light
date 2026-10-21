"""Tests for the goal-staleness cadence arithmetic and curve aggregation; pure CPU."""
import unittest

from goal_staleness import (
    aggregate_staleness, effective_goal_rows, load_driver_results, refresh_grid,
    should_refresh, staleness_rows)


class RefreshGridTests(unittest.TestCase):
    def test_always_contains_the_trained_cadence(self):
        self.assertIn(48, refresh_grid(48, [2.0, 3.0]))

    def test_multipliers_scale_H_g_and_dedupe(self):
        self.assertEqual(refresh_grid(48, [1.0, 1.0, 2.0, 3.0]), [48, 96, 144])

    def test_nonpositive_multipliers_dropped(self):
        self.assertEqual(refresh_grid(48, [0.0, -1.0, 2.0]), [48, 96])

    def test_bad_H_g_rejected(self):
        with self.assertRaises(ValueError):
            refresh_grid(0, [1.0])


class CadenceTests(unittest.TestCase):
    def test_first_query_always_refreshes(self):
        self.assertTrue(should_refresh(False, 0, 48))

    def test_trained_cadence_refreshes_at_H_g(self):
        self.assertFalse(should_refresh(True, 40, 48))
        self.assertTrue(should_refresh(True, 48, 48))
        self.assertTrue(should_refresh(True, 56, 48))

    def test_longer_interval_holds_the_goal_older(self):
        # Same steps_since_refresh, a longer interval keeps holding.
        self.assertFalse(should_refresh(True, 48, 96))
        self.assertTrue(should_refresh(True, 96, 96))

    def test_bad_interval_rejected(self):
        with self.assertRaises(ValueError):
            should_refresh(True, 0, 0)


class EffectiveRowsTests(unittest.TestCase):
    def test_shrinks_from_H_g_to_zero(self):
        self.assertEqual(effective_goal_rows(48, 0), 48.0)
        self.assertEqual(effective_goal_rows(48, 24), 24.0)

    def test_clamped_at_one_row_past_H_g(self):
        self.assertEqual(effective_goal_rows(48, 48), 1.0)
        self.assertEqual(effective_goal_rows(48, 200), 1.0)


class StalenessRowsTests(unittest.TestCase):
    def test_trained_cadence_has_zero_extra_staleness(self):
        self.assertEqual(staleness_rows(48, 48), 0)

    def test_longer_interval_adds_rows(self):
        self.assertEqual(staleness_rows(96, 48), 48)


class AggregateTests(unittest.TestCase):
    def records(self):
        return [
            dict(refresh_rows=48, success=True, steps=120.0),
            dict(refresh_rows=48, success=False, steps=300.0),
            dict(refresh_rows=96, success=False, steps=300.0),
            dict(refresh_rows=96, success=False, steps=300.0),
        ]

    def test_counts_and_rates_per_interval(self):
        curve = aggregate_staleness(self.records())['curve']
        self.assertEqual([p['refresh_rows'] for p in curve], [48, 96])
        self.assertEqual(curve[0]['episodes'], 2)
        self.assertEqual(curve[0]['successes'], 1)
        self.assertAlmostEqual(curve[0]['success_rate'], 0.5)
        self.assertEqual(curve[1]['success_rate'], 0.0)

    def test_mean_steps_reported(self):
        curve = aggregate_staleness(self.records())['curve']
        self.assertAlmostEqual(curve[0]['mean_steps'], 210.0)
        self.assertAlmostEqual(curve[1]['mean_steps'], 300.0)

    def test_H_g_annotates_extra_staleness(self):
        curve = aggregate_staleness(self.records(), H_g=48)['curve']
        self.assertEqual(curve[0]['staleness_rows'], 0)
        self.assertEqual(curve[1]['staleness_rows'], 48)

    def test_empty_input_gives_empty_curve(self):
        self.assertEqual(aggregate_staleness([])['curve'], [])


class LoadDriverResultsTests(unittest.TestCase):
    def test_reads_episode_results(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.json'
            path.write_text(json.dumps(dict(episode_results=[
                dict(success=True, steps=100), dict(success=False, steps=300)])), encoding='utf-8')
            rows = load_driver_results(path)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]['success'])


if __name__ == '__main__':
    unittest.main()
