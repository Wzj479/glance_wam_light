import copy
import unittest

from eval_protocol import (CONDITIONS, paired_delta, protocol_hash,
                           validate_episode_results, validate_protocol)


def protocol():
    return dict(schema_version=1, suite='libero_goal', task_ids=[0, 2],
                episode_ids=[10, 11], seed=42, conditions=copy.deepcopy(CONDITIONS))


class ProtocolTests(unittest.TestCase):
    def test_protocol_hash_changes_if_seed_or_conditions_change(self):
        a = protocol()
        b = copy.deepcopy(a)
        b['seed'] = 43
        self.assertNotEqual(protocol_hash(a), protocol_hash(b))
        b = copy.deepcopy(a)
        b['conditions']['dark']['brightness'] = .5
        self.assertNotEqual(protocol_hash(a), protocol_hash(b))

    def test_duplicate_episode_ids_rejected(self):
        p = protocol()
        p['episode_ids'] = [10, 10]
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_nan_corruption_rejected(self):
        p = protocol()
        p['conditions']['dark']['brightness'] = float('nan')
        with self.assertRaises(ValueError):
            validate_protocol(p)

    def test_missing_and_duplicated_rollouts_rejected(self):
        rows = [dict(task_id=t, episode_id=e, success=True) for t in [0, 2] for e in [10, 11]]
        validate_episode_results(rows, protocol())
        with self.assertRaises(ValueError):
            validate_episode_results(rows[:-1], protocol())
        with self.assertRaises(ValueError):
            validate_episode_results(rows + [rows[0]], protocol())

    def test_paired_improvement_and_regression(self):
        b = [dict(task_id=0, episode_id=i, success=s) for i, s in enumerate([False, True, False, True])]
        c = [dict(task_id=0, episode_id=i, success=s) for i, s in enumerate([True, False, True, True])]
        result = paired_delta(b, c)
        self.assertEqual(result['recovered'], 2)
        self.assertEqual(result['regressed'], 1)
        self.assertEqual(result['success_delta_pp'], 25.)

    def test_mismatched_pairings_rejected(self):
        with self.assertRaises(ValueError):
            paired_delta([dict(task_id=0, episode_id=0, success=True)],
                         [dict(task_id=0, episode_id=1, success=True)])


if __name__ == '__main__':
    unittest.main()
