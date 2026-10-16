"""Additional timing plots, verified videos, full and compact result archives."""
import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import imageio_ffmpeg


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args()
    root=a.root.resolve();out=root/'results/control_sensitivity'
    summary=json.loads((out/'summary.json').read_text())
    assert json.loads((out/'video_checks.json').read_text())['status']=='passed'
    # Clarify field names from the first run: env.step stays 20Hz; only command
    # refresh is reduced. MuJoCo's internal integrator frequency is unchanged.
    for path in out.glob('*.json'):
        if path.name=='SHA256SUMS.json':continue
        item=json.loads(path.read_text())
        if 'case' not in item:continue
        if 'simulation_hz' in item:item['control_hz']=item.pop('simulation_hz')
        if 'arm_update_hz' in item:item['command_refresh_hz']=item.pop('arm_update_hz')
        path.write_text(json.dumps(item,indent=2))
    summary['cases']=[json.loads((out/(r['case']+'.json')).read_text()) for r in summary['cases']]
    summary['timebase']='Simulation time; no wall-clock real-time deadline or model inference speed measured'
    summary['peak_numeric_worker_rss_mib']=max(r['peak_rss_mib'] for r in summary['cases'])
    summary['peak_video_worker_rss_mib']=max(json.loads((out/(name+'_video_check.json')).read_text())['peak_rss_mib'] for name in summary['selected_video_cases'])
    source=json.loads((root/'data/matched_grasp_source/match.json').read_text())
    summary['source']={key:source[key] for key in ['source_revision','lerobot_dataset_revision',
        'lerobot_episode','matched_original_demo','extracted_npz_sha256']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    columns=['case','kind','amount','delay_ms','control_hz','command_refresh_hz',
             'final_success','ever_success','bowl_lift_m','final_bowl_plate_xy_distance_m',
             'first_success_step','peak_rss_mib']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summary['cases'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
    per_step=[]
    for result in summary['cases']:
        name=result['case'];z=np.load(out/(name+'.npz'));t=np.arange(len(z['poses']))/20
        for i,ax in enumerate(axes):ax.plot(t,z['poses'][:,i],label=name)
        for i in range(len(t)):
            per_step.append([name,i,t[i],*z['poses'][i,:3],np.sum(np.abs(z['poses'][i,7:])),
                z['objects'][i,2]-z['objects'][0,2],np.linalg.norm(z['objects'][i,:2]-z['objects'][i,3:5])])
    for name,ax in zip(['World x (m)','World y (m)','World z (m)'],axes):ax.set_ylabel(name);ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel('Time from first command (s)')
    fig.savefig(out/'eef_timing_traces.png',dpi=110);plt.close(fig)
    with (out/'per_step_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['case','step','time_s','eef_x_m','eef_y_m','eef_z_m','finger_width_m','bowl_lift_m','bowl_plate_xy_distance_m']);writer.writerows(per_step)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    for name in ['baseline','gripper_delay_1','gripper_delay_2','gripper_delay_4']:
        z=np.load(out/(name+'.npz'));t=np.arange(len(z['actions']))/20
        axes[0].step(t,(1-z['actions'][:,6])/2,where='post',label=name)
        axes[1].plot(np.arange(len(z['poses']))/20,np.sum(np.abs(z['poses'][:,7:]),axis=1)*1000,label=name)
    axes[0].set_ylabel('Command: 1=open, 0=closed');axes[1].set_ylabel('Actual width (mm)')
    axes[1].set_xlabel('Time from first command (s)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'gripper_timing.png',dpi=110);plt.close(fig)
    videos=[]
    for name in summary['selected_video_cases']:
        path=out/(name+'.mp4');n,duration=imageio_ffmpeg.count_frames_and_secs(str(path))
        assert n==summary['cases'][0]['executed_actions']+1
        videos.append(dict(case=name,frames=n,seconds=duration,sha256=sha(path)))
    outcomes={r['case']:r for r in summary['cases']}
    selected_failures=[name for name in summary['selected_video_cases'][1:] if not outcomes[name]['final_success']]
    compared=selected_failures[0] if selected_failures else summary['selected_video_cases'][1]
    comparison=out/'baseline_vs_perturbed.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y',
        '-i',str(out/'baseline.mp4'),'-i',str(out/(compared+'.mp4')),
        '-filter_complex_threads','1','-filter_complex','[0:v][1:v]hstack=inputs=2[v]',
        '-map','[v]','-c:v','libx264','-crf','30','-threads','1','-an',str(comparison)],check=True)
    count,duration=imageio_ffmpeg.count_frames_and_secs(str(comparison))
    assert count==summary['cases'][0]['executed_actions']+1
    videos.append(dict(case='baseline_vs_perturbed',left='baseline',right=compared,
                       frames=count,seconds=duration,sha256=sha(comparison)))
    (out/'video_checks.json').write_text(json.dumps(dict(status='passed',videos=videos,
        rendered_and_nonrendered_states_match_atol=1e-7),indent=2))
    lines=['# 记录动作的时序敏感性实测', '',
        '一条匹配抓取示范；固定初始状态、随机种子和 10 步稳定等待；9 个条件均执行 136 个控制周期（20Hz，6.8 秒）。', '',
        '| 条件 | 最终任务成功 | 最大抬升 mm | 最终碗盘 XY 距离 mm |',
        '| --- | --- | ---: | ---: |']
    for r in summary['cases']:
        lines.append(f"| {r['case']} | {'是' if r['final_success'] else '否'} | {r['bowl_lift_m']*1000:.2f} | {r['final_bowl_plate_xy_distance_m']*1000:.2f} |")
    failed=[r['case'] for r in summary['cases'] if not r['final_success']]
    lines += ['', '本次未满足最终任务谓词的条件：'+('、'.join(failed) if failed else '无')+'。', '',
        '所有频率、延迟和时间轴均为仿真时间；CPU 回放不要求实时执行。',
        '整体动作延迟是固定动作流的起始时移；没有模型、观测延迟或在线重规划。',
        '动作保持重复同一条增量命令，控制器仍每 50ms 更新目标；它不等于保持绝对位姿，也不等于执行多个不同动作组成的 WAM 动作块。',
        '一个条件只跑一条示范；重渲染的视频用于检查物理状态一致性，不构成新增独立评测回合。',
        '视频对照左侧为 baseline，右侧为 '+compared+'。',
        '使用 CPU texture1024 诊断资产，不是原始视觉基准成绩。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.json')
    manifest={p.name:sha(p) for p in files};(out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(root/'wam_control_sensitivity_full.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,'control_sensitivity/'+path.name)
        z.write(out/'SHA256SUMS.json','control_sensitivity/SHA256SUMS.json')
    selected=[p for p in files if p.suffix in ['.png','.mp4','.md'] or p.name in ['summary.json','summary.csv','video_checks.json']]
    with zipfile.ZipFile(root/'wam_control_sensitivity_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in selected:z.write(path,'control_sensitivity/'+path.name)
        z.writestr('control_sensitivity/SHA256SUMS.json',json.dumps({p.name:sha(p) for p in selected},indent=2))
    print(json.dumps(dict(full_files=len(files),videos=videos,
        review_bytes=(root/'wam_control_sensitivity_review.zip').stat().st_size,
        review_sha256=sha(root/'wam_control_sensitivity_review.zip')),indent=2))


if __name__=='__main__':main()
