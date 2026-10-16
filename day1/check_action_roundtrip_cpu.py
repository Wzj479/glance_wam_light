"""Run official NumPy deployment methods on real actions, without model imports."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from finetune_utils import checkpoint_action_metadata


def extract(path, class_name, name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    container=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name) if class_name else tree
    fn=copy.deepcopy(next(n for n in container.body if isinstance(n,ast.FunctionDef) and n.name==name))
    fn.decorator_list=[]
    scope={'np':np}
    exec(compile('from __future__ import annotations\n'+ast.unparse(fn),str(path),'exec'),scope)
    return scope[name]


def normalized(actions,lo,hi):
    result=np.asarray(actions,np.float32).copy()
    lo,hi=np.asarray(lo,np.float32),np.asarray(hi,np.float32)
    result[:,:6]=2*((result[:,:6]-lo[:6])/(hi[:6]-lo[:6]))-1
    return result.astype(np.float16)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--checkpoint-stats',type=Path,required=True)
    p.add_argument('--suite-stats',type=Path,required=True)
    p.add_argument('--official-interface',type=Path,required=True)
    p.add_argument('--official-eval',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data=np.load(a.source/'matched_demo.npz',allow_pickle=False);actions=data['lerobot_actions']
    checkpoint=json.loads(a.checkpoint_stats.read_text())['franka']['action']
    suite=json.loads(a.suite_stats.read_text())['statistics']['action']
    inverse=extract(a.official_interface,'ModelClient','unnormalize_actions')
    grip=extract(a.official_eval,None,'_binarize_gripper_open')
    # Exercise the exact metadata override used by the revised trainer.
    keys=['x','y','z','roll','pitch','yaw','gripper']
    metadata=SimpleNamespace(statistics=SimpleNamespace(action={k:SimpleNamespace(
        min=np.array([suite['min'][i]]),max=np.array([suite['max'][i]])) for i,k in enumerate(keys)}))
    aligned=checkpoint_action_metadata(metadata,checkpoint)
    lo=[aligned.statistics.action[k].min[0] for k in keys]
    hi=[aligned.statistics.action[k].max[0] for k in keys]
    good_norm=normalized(actions,lo,hi)
    bad_norm=normalized(actions,suite['min'],suite['max'])
    good=inverse(good_norm.astype(np.float32),checkpoint)
    bad=inverse(bad_norm.astype(np.float32),checkpoint)
    commands=good.copy();commands[:,6]=[grip(v)[0] for v in good[:,6]]
    np.testing.assert_array_equal(commands[:,6],data['env_actions'][:,6])
    good_error=np.max(np.abs(commands-data['env_actions']),axis=0)
    bad_error=np.max(np.abs(bad[:,:6]-actions[:,:6]),axis=0)
    assert np.max(good_error[:6])<.0005
    assert bad_error[5]>.01
    zero=np.zeros((1,7));zero[:,6]=1
    zero_wrong=inverse(normalized(zero,suite['min'],suite['max']).astype(np.float32),checkpoint)
    report=dict(status='passed',rows=len(actions),checkpoint_stats_sha256=hashlib.sha256(a.checkpoint_stats.read_bytes()).hexdigest(),
        checkpoint_mask=checkpoint['mask'],suite_min=suite['min'],checkpoint_min=checkpoint['min'],
        max_abs_command_error_aligned_float16=good_error.tolist(),max_abs_first6_error_without_alignment=bad_error.tolist(),
        zero_yaw_command_without_alignment=float(zero_wrong[0,5]),
        max_abs_controller_goal_error_aligned=(good_error[:6]*[.05,.05,.05,.5,.5,.5]).tolist(),
        gripper_mapping_all_rows_match=True,
        official_source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.official_interface,a.official_eval]},
        scope='Official NumPy postprocess functions extracted with AST; neural model not loaded')
    dump=lambda p,v:p.write_text(json.dumps(v,indent=2))
    dump(a.out/'normalization_report.json',report)
    np.savez_compressed(a.out/'normalization_roundtrip.npz',raw=actions,normalized=good_norm,
                        recovered=commands,wrong_recovered=bad)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    axes[0].plot(actions[:,5],label='raw rotation-z command');axes[0].plot(bad[:,5],'--',label='without scale alignment')
    axes[0].plot(commands[:,5],':',label='checkpoint-aligned roundtrip');axes[0].legend();axes[0].set_ylabel('Command unit')
    axes[1].plot((commands[:,5]-actions[:,5])*1000,label='aligned error x1000')
    axes[1].plot((bad[:,5]-actions[:,5])*1000,label='unaligned error x1000')
    axes[1].set(xlabel='Episode frame',ylabel='Error x1000');axes[1].legend();fig.savefig(a.out/'normalization_roundtrip.png',dpi=110)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
