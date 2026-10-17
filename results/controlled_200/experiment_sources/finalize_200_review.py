import hashlib,json,shutil,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('/root/autodl-tmp/wam_glance_project')
out=root/'GlanceWAM/results/controlled_200'
report=out/'comparison'
data=json.loads((report/'overview.json').read_text())
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
x=np.arange(3)
for j,name in enumerate(['base','clean','aug']):
    counts=[data['closed_loop'][name][c]['successes'] for c in ['clean','dark','color']]
    bars=ax.bar(x+(j-1)*.24,counts,width=.23,label=name)
    ax.bar_label(bars,labels=[f'{n}/10' for n in counts],padding=3,fontsize=9)
ax.set(xticks=x,xticklabels=['clean','dark','color'],ylim=(0,11.5),ylabel='Successful episodes',
       title='Paired grasp evaluation: task 8, initial states 10–19')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
ax.grid(axis='y',alpha=.2)
fig.savefig(report/'closed_loop_success.png',dpi=130);plt.close(fig)
doc=report/'实验记录.md'
text=doc.read_text(encoding='utf-8')
anchor='未使用示范动作回放。'
if '本次主要结果：' not in text:
    text=text.replace(anchor,anchor+'\n\n本次主要结果：三模型在三个视觉条件下均为10/10成功，未观察到后训练的闭环成功率收益。aug相对clean的固定组合扰动验证损失降低约5.3%，正常验证损失升高约8.1%；离线指标变化尚未带来本组闭环成功率差异。')
    text += '\n成功按LIBERO官方任务终止谓词判定；没有增加松爪后持续稳定的独立验收条件。基线权重符号链接保留部署配置目录，相关配对/路径回归测试共11项通过。\n'
doc.write_text(text,encoding='utf-8')
for name in ['analyze_controlled_200.py','finalize_200_review.py']:
    shutil.copy2(root/name,out/'experiment_sources'/name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.pt' and p.name!='SHA256SUMS.json']
(out/'SHA256SUMS.json').write_text(json.dumps({p.relative_to(out).as_posix():sha(p) for p in files},indent=2,ensure_ascii=False),encoding='utf-8')
for kind in ['full','review']:
    archive=root/('wam_clean_aug_200_'+kind+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if not p.is_file() or p.suffix=='.pt':continue
            rel=p.relative_to(out)
            if kind=='review':
                if p.suffix=='.mp4' and rel.parts[0]!='comparison':continue
                if p.suffix=='.npy' or p.name=='SHA256SUMS.json':continue
            z.write(p,'controlled_200/'+rel.as_posix())
        if kind=='review':
            checks={n:sha(out/n[len('controlled_200/'):]) for n in z.namelist()}
            z.writestr('controlled_200/REVIEW_SHA256SUMS.json',json.dumps(checks,indent=2,ensure_ascii=False))
    print(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}))
