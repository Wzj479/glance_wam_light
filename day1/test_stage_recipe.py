"""CPU fixtures: sampling leakage, selection safety and action checkpoint loading."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn

from stage_finetune_utils import gripper_phases, draw_schedule, phase_selection, build_phase_index
from action_checkpoint import (base_fingerprint, save_action_checkpoint, read_action_checkpoint,
                               apply_action_checkpoint, load_policy_checkpoint)
from action_training_scope import configure_action_scope, set_action_training_modes


class StageTests(unittest.TestCase):
    def test_phase_boundaries_and_binary_label_rejection(self):
        g=np.r_[np.ones(12),np.zeros(40),np.ones(10)]
        p=gripper_phases(g)
        self.assertEqual((p['close'],p['release']),(12,52))
        self.assertEqual(np.flatnonzero(p['grasp']).tolist(),list(range(4,37)))
        self.assertEqual(np.flatnonzero(p['place']).tolist(),list(range(28,61)))
        for bad in [[-1,1,-1],[1,.2,1],[1,1,1],[1,0,1,0,1]]:
            with self.assertRaises(ValueError):gripper_phases(bad)

    def test_balanced_draws_overlap_and_determinism(self):
        pools=dict(uniform=list(range(10)),grasp=[1,2,3],place=[3,4])
        indices,sources=draw_schedule(pools,'phase',800,42)
        self.assertEqual([sources.count(s) for s in pools],[400,200,200])
        self.assertTrue(all(i in pools[s] for i,s in zip(indices,sources)))
        self.assertEqual((indices,sources),draw_schedule(pools,'phase',800,42))
        self.assertTrue(all(s=='uniform' for s in draw_schedule(pools,'uniform',800,42)[1]))
        self.assertTrue(all(i<10 for i in indices))

    def test_validation_gate_and_step_zero_fallback(self):
        base=dict(clean_fm_loss=1.,clean_grasp_fm_loss=2.,clean_place_fm_loss=4.)
        better=dict(clean_fm_loss=1.04,clean_grasp_fm_loss=1.,clean_place_fm_loss=2.)
        self.assertEqual(phase_selection(better,base),(.5,True))
        better['clean_fm_loss']=1.06
        self.assertEqual(phase_selection(better,base),(.5,False))
        self.assertEqual(phase_selection(base,base),(1.,True))

    def test_pool_builder_uses_only_train_entries(self):
        import pandas as pd
        g=np.r_[np.ones(12),np.zeros(40),np.ones(10)]
        df=pd.DataFrame(dict(frame_index=np.arange(len(g)),action=[np.r_[np.zeros(6),x] for x in g]))
        ds=SimpleNamespace(dataset_name='fixture',all_steps=[(ep,i) for ep in [0,1] for i in range(len(g))],
            lerobot_modality_meta=SimpleNamespace(action={'gripper':SimpleNamespace(original_key='action',start=6,end=7)}),
            get_trajectory_data=lambda ep:df.copy())
        train=[(0,i) for i in range(len(g))];val=[(0,len(g)+i) for i in range(len(g))]
        tags,pools,m=build_phase_index([ds],train,val)
        self.assertTrue(all(x['episode']==0 for p in m['train_pool_entries'].values() for x in p))
        self.assertEqual(len(m['validation_tags']),len(val))
        self.assertTrue(all(i<len(train) for pool in pools.values() for i in pool))
        self.assertIn('place',tags[0,52])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.base=self.root/'base/checkpoints/weights.pt';self.base.parent.mkdir(parents=True)
        self.base.write_bytes(b'base fixture')
        for name in ['config.yaml','dataset_statistics.json']:(self.base.parents[1]/name).write_text('{}')
        self.head=nn.Sequential(nn.Linear(3,4),nn.Linear(4,7))
        self.dest=self.root/'run/checkpoints/last.action.pt';self.dest.parent.mkdir(parents=True)
        for name in ['config.yaml','dataset_statistics.json']:(self.dest.parents[1]/name).write_text('{}')
        save_action_checkpoint(self.head,self.dest,base_fingerprint(self.base),dict(step=100,checkpoint_format='action'))

    def tearDown(self):self.temp.cleanup()

    def test_strict_roundtrip_and_real_loader_branch(self):
        original=copy.deepcopy(self.head.state_dict())
        model=SimpleNamespace(action_model=copy.deepcopy(self.head))
        for p in model.action_model.parameters():p.data.zero_()
        class Framework:
            @staticmethod
            def from_pretrained(path):
                self.assertEqual(Path(path),self.base)
                return model
        self.assertIs(load_policy_checkpoint(self.dest,Framework),model)
        from run_robust_eval import checkpoint_view_path
        self.assertEqual(checkpoint_view_path(self.dest),self.dest.absolute())
        for k,v in model.action_model.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)

    def test_wrong_base_and_corrupt_checkpoint_rejected(self):
        self.base.write_bytes(b'wrong base')
        with self.assertRaises(ValueError):read_action_checkpoint(self.dest)
        self.base.write_bytes(b'base fixture')
        self.dest.write_bytes(self.dest.read_bytes()+b'corrupt')
        with self.assertRaises(ValueError):read_action_checkpoint(self.dest)

    def test_legacy_full_checkpoint_keeps_original_loader(self):
        seen=[]
        class Framework:
            @staticmethod
            def from_pretrained(path):seen.append(path);return 'legacy'
        self.assertEqual(load_policy_checkpoint(str(self.base),Framework),'legacy')
        self.assertEqual(seen,[str(self.base)])

    def test_changed_stats_and_wrong_architecture_rejected(self):
        _,payload=read_action_checkpoint(self.dest)
        with self.assertRaises(ValueError):apply_action_checkpoint(nn.Linear(3,7),payload)
        (self.base.parents[1]/'dataset_statistics.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):read_action_checkpoint(self.dest)

    def test_symlink_uses_view_sidecars(self):
        view=self.root/'view/checkpoints/base.pt';view.parent.mkdir(parents=True)
        try:view.symlink_to(self.base)
        except OSError:self.skipTest('Symlink unavailable')
        for name in ['config.yaml','dataset_statistics.json']:(view.parents[1]/name).write_text('view')
        identity=base_fingerprint(view)
        self.assertEqual(identity['path'],str(view.absolute()))
        self.assertNotEqual(identity['config_sha256'],base_fingerprint(self.base)['config_sha256'])


class UpstreamFlowTests(unittest.TestCase):
    def test_actual_upstream_velocity_mse_and_frozen_gradient_scope(self):
        # Small dimensions, same real upstream head/DiT/forward, CPU only. No optimizer step.
        from omegaconf import OmegaConf
        from glancewam.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, DiTConfig
        DiTConfig['UNIT'] = dict(input_embedding_dim=32,num_attention_heads=4,attention_head_dim=8)
        cfg=OmegaConf.create(dict(framework=dict(action_model=dict(action_model_type='UNIT',
            diffusion_model_cfg=dict(output_dim=32,num_layers=2,cross_attention_dim=16,
                interleave_self_attention=True,dropout=0.,final_dropout=False),
            action_dim=7,action_horizon=8,state_dim=0,hidden_size=32,num_inference_timesteps=2,
            num_target_vision_tokens=2,add_pos_embed=True,max_seq_len=32,
            noise_beta_alpha=1.5,noise_beta_beta=1.,noise_s=.999,num_timestep_buckets=1000))))
        try:head=FlowmatchingActionHead(cfg)
        finally:del DiTConfig['UNIT']
        wrapper=nn.Module();wrapper.action_model=head;wrapper.backbone=nn.Linear(2,2)
        trainable,frozen=configure_action_scope(wrapper,'crossattn_decoder')
        set_action_training_modes(wrapper,'crossattn_decoder')
        self.assertFalse(head.model.transformer_blocks[1].training)
        self.assertFalse(head.model.transformer_blocks[1].attn1.training)
        self.assertTrue(head.model.transformer_blocks[0].attn1.training)
        before={n:p.detach().clone() for n,p in frozen}
        actions=torch.randn(1,8,7);cond=torch.randn(1,5,16);captured={}
        hook=head.action_decoder.register_forward_hook(lambda module,inputs,output:captured.update(pred=output))
        torch.manual_seed(123);noise=torch.randn_like(actions)
        torch.manual_seed(123);loss=head(cond,actions,None);hook.remove()
        expected=((captured['pred'][:,-8:].float()-(actions-noise).float())**2).mean()
        torch.testing.assert_close(loss,expected,rtol=0,atol=0)
        loss.backward()
        for group in ['.attn1.','.action_decoder.']:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in trainable if group in n))
        self.assertTrue(all(p.grad is None for n,p in frozen))
        for n,p in frozen:torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        self.assertTrue(all('.attn1.' in n or '.action_decoder.' in n for n,p in trainable))


if __name__=='__main__':unittest.main()
