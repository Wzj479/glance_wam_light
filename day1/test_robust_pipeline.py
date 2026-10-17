"""Synthetic fixtures test aggregation/selection only; these are not robot results."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from eval_protocol import CONDITIONS, protocol_hash
from run_robust_eval import aggregate, checkpoint_view_path


class PipelineTests(unittest.TestCase):
    def test_checkpoint_symlink_keeps_run_configuration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            source = root / 'source/checkpoints/weights.pt'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'fixture')
            view = root / 'view/checkpoints/weights.pt'
            view.parent.mkdir(parents=True)
            try:
                view.symlink_to(source)
            except OSError:
                self.skipTest('Symlink creation unavailable')
            (view.parents[1] / 'config.yaml').write_text('base_wm: local')
            (view.parents[1] / 'dataset_statistics.json').write_text('{}')
            chosen = checkpoint_view_path(view)
            self.assertEqual(chosen.parents[1], view.parents[1].absolute())
            self.assertNotEqual(chosen, source.resolve())

    def fixture(self, root):
        protocol = dict(schema_version=1, suite='libero_goal', task_ids=[0, 1, 2],
                        episode_ids=[0, 1, 2], seed=7, conditions=copy.deepcopy(CONDITIONS))
        (root / 'protocol.json').write_text(json.dumps(protocol))
        for name in ('base', 'aug'):
            for condition in CONDITIONS:
                dest = root / name / condition
                dest.mkdir(parents=True)
                records = [dict(task_id=t, episode_id=e, env_seed=t * 10 + e,
                    init_state_sha256=f'state-{t}-{e}', success=(condition == 'clean' or name == 'aug'))
                    for t in range(3) for e in range(3)]
                result = dict(protocol_sha256=protocol_hash(protocol), successes=sum(r['success'] for r in records),
                    episodes=9, success_rate=sum(r['success'] for r in records) / 9, episode_results=records)
                (dest / 'results.json').write_text(json.dumps(result))
        return protocol

    def test_full_matrix_and_paired_counts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            protocol = self.fixture(root)
            aggregate(root, protocol, ['base', 'aug'])
            rows = json.loads((root / 'summary.json').read_text())['rows']
            row = next(r for r in rows if r['model'] == 'aug' and r['condition'] == 'dark')
            self.assertEqual(row['recovered'], 9)
            self.assertEqual(row['regressed'], 0)
            self.assertEqual(row['success_delta_pp'], 100)

    def test_changed_physical_state_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            protocol = self.fixture(root)
            path = root / 'aug/dark/results.json'
            result = json.loads(path.read_text())
            result['episode_results'][0]['init_state_sha256'] = 'different'
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                aggregate(root, protocol, ['base', 'aug'])

    def test_selection_reserves_new_initial_states(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            protocol = self.fixture(root)
            aggregate(root, protocol, ['base', 'aug'])
            final = root / 'final.json'
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name('select_pilot_tasks.py')),
                '--pilot', str(root), '--out', str(final)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            selected = json.loads(final.read_text())
            self.assertEqual(selected['task_ids'], [0, 1, 2])
            self.assertFalse(set(selected['episode_ids']) & set(protocol['episode_ids']))

    def test_changed_post_settle_state_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            protocol = self.fixture(root)
            path = root / 'aug/dark/results.json'
            result = json.loads(path.read_text())
            result['episode_results'][0]['post_settle_state_sha256'] = 'different'
            path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                aggregate(root, protocol, ['base', 'aug'])


if __name__ == '__main__':
    unittest.main()
