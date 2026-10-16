"""CPU tests: python -m unittest discover -s day1 -p 'test_finetune_utils.py' -v"""
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from finetune_utils import (augment_sample, checkpoint_action_metadata, deployment_keys, load_episode_partition, photometric_window,
                            split_episodes, split_step_indices)


class EpisodeSplitTests(unittest.TestCase):
    def explicit(self, train, val, eligible):
        text = json.dumps(dict(schema_version=1, datasets={
            'demo': dict(train_episode_ids=train, val_episode_ids=val)}))
        with patch('finetune_utils.Path.read_text', return_value=text):
            return load_episode_partition('split.json', 'demo', eligible)

    def test_audited_partition_is_preserved_without_adding_other_episodes(self):
        self.assertEqual(self.explicit([43, 0], [52], range(100)), ([0, 43], [52]))

    def test_explicit_split_rejects_leakage_duplicates_missing_and_empty_ids(self):
        for train, val, eligible in [([0], [0], [0]), ([0, 0], [1], [0, 1]),
                                     ([0], [2], [0, 1]), ([], [1], [0, 1]),
                                     ([0], ['1'], [0, 1])]:
            with self.subTest(train=train, val=val):
                with self.assertRaises(ValueError):
                    self.explicit(train, val, eligible)

    def test_deterministic_episode_partition(self):
        ids = list(range(20))
        tr, va = split_episodes(ids, .2, 42)
        self.assertEqual((tr, va), split_episodes(ids[::-1], .2, 42))
        self.assertEqual(len(va), 4)
        self.assertEqual(set(tr) | set(va), set(ids))
        self.assertFalse(set(tr) & set(va))

    def test_no_frames_from_validation_episode_in_train(self):
        steps = [(episode, frame) for episode in range(6) for frame in range(16)]
        tr, va = split_episodes(range(6), .3, 42)
        ti, vi = split_step_indices(steps, tr, va, 8)
        self.assertEqual({steps[i][0] for i in ti}, set(tr))
        self.assertEqual({steps[i][0] for i in vi}, set(va))
        self.assertFalse({steps[i][0] for i in ti} & {steps[i][0] for i in vi})
        self.assertTrue(all(steps[i][1] % 8 == 0 for i in vi))

    def test_overlapping_episode_lists_rejected(self):
        with self.assertRaises(ValueError):
            split_step_indices([(0, 0), (1, 0)], [0], [0, 1], 8)

    def test_tiny_dataset_rejected(self):
        with self.assertRaises(ValueError):
            split_episodes([0], .1, 42)


class ActionNormalizationTests(unittest.TestCase):
    def test_checkpoint_bounds_replace_suite_bounds_without_mutating_source(self):
        keys = ['x','y','z','roll','pitch','yaw','gripper']
        source = SimpleNamespace(statistics=SimpleNamespace(action={
            k: SimpleNamespace(min=np.array([-.287142843]), max=np.array([.375])) for k in keys}))
        stats = dict(min=[-.9375]*3+[-.258214295,-.375,-.367500007,0],
                     max=[.9375]*3+[.355714291,.375,.375,1], mask=[True]*6+[False])
        aligned = checkpoint_action_metadata(source, stats)
        normalized_zero_yaw = 2*(0-aligned.statistics.action['yaw'].min)/(aligned.statistics.action['yaw'].max-aligned.statistics.action['yaw'].min)-1
        restored = .5*(normalized_zero_yaw+1)*(stats['max'][5]-stats['min'][5])+stats['min'][5]
        self.assertAlmostEqual(float(restored[0]), 0, places=6)
        self.assertAlmostEqual(float(source.statistics.action['yaw'].min[0]), -.287142843)

    def test_gripper_normalization_mask_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            checkpoint_action_metadata(None, dict(min=[-1]*7, max=[1]*7, mask=[True]*7))


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        pixels = np.full((8, 16, 3), [60, 100, 140], dtype=np.uint8)
        self.image = Image.fromarray(pixels)

    def test_window_consistency_and_no_inplace_mutation(self):
        original = np.asarray(self.image).copy()
        changed = photometric_window([self.image] * 3, .7, .6, (.9, 1., 1.1))
        self.assertTrue(all(np.array_equal(changed[0], im) for im in changed))
        self.assertTrue(np.array_equal(np.asarray(self.image), original))
        self.assertFalse(np.array_equal(changed[0], original))
        self.assertEqual(changed[0].size, self.image.size)

    def test_labels_are_unchanged_and_seed_reproducible(self):
        actions = np.arange(56, dtype=np.float32).reshape(8, 7)
        sample = dict(image=[self.image] * 3, action=actions, lang='pick bowl', goal_offset_rows=12)
        def run():
            return augment_sample(sample, np.random.default_rng(4), 1., (.6, 1.4), (.6, 1.4), (.9, 1.1))
        a, b = run(), run()
        self.assertIs(a['action'], actions)
        self.assertEqual(a['goal_offset_rows'], 12)
        self.assertEqual(a['lang'], 'pick bowl')
        self.assertTrue(np.array_equal(a['image'][0], b['image'][0]))

    def test_clean_control_is_identity(self):
        sample = dict(image=[self.image])
        result = augment_sample(sample, np.random.default_rng(42), 0., (.6, 1.4), (.6, 1.4), (.9, 1.1))
        self.assertIs(result, sample)

    def test_bad_image_type_rejected(self):
        with self.assertRaises(ValueError):
            photometric_window([np.zeros((8, 8, 3), dtype=np.float32)])


class CheckpointContractTests(unittest.TestCase):
    def test_robot_backbone_is_preserved_and_only_reconstructable_components_omitted(self):
        keys = ['backbone.transformer.weight', 'backbone.vae.weight', 'backbone.text_encoder.weight',
                'action_model.weight', 'goal_time_proj.weight', 'tap_embed.weight']
        keep = deployment_keys(keys,
            ('backbone.transformer.', 'action_model.', 'goal_time_proj.', 'tap_embed.'),
            ('backbone.vae.', 'backbone.text_encoder.'))
        self.assertIn('backbone.transformer.weight', keep)
        self.assertIn('action_model.weight', keep)
        self.assertNotIn('backbone.vae.weight', keep)
        self.assertEqual(len(keep), 4)

    def test_unknown_module_is_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            deployment_keys(['backbone.transformer.w', 'action_model.w', 'unexpected.w'],
                            ('backbone.transformer.', 'action_model.'), ())

    def test_head_only_checkpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            deployment_keys(['action_model.w'], ('action_model.',), ())


if __name__ == '__main__':
    unittest.main()
