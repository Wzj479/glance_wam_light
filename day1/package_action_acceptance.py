"""Verify real outputs and package full results plus a compact video review."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import imageio_ffmpeg


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args();root=a.root.resolve()
    output_dirs=[root/'results/action_interface',root/'results/matched_grasp_replay']
    assert json.loads((output_dirs[0]/'interface_report.json').read_text())['status']=='passed'
    assert json.loads((output_dirs[0]/'normalization_report.json').read_text())['status']=='passed'
    assert json.loads((output_dirs[1]/'replay_report.json').read_text())['repeat_passed']
    videos=[('action_interface','interface_control',150),('matched_grasp_replay','matched_settle10',113),('matched_grasp_replay','matched_settle0',113)]
    review=root/'results/action_acceptance_review';review.mkdir(exist_ok=True)
    video_info=[]
    for folder,name,expected in videos:
        source=root/'results'/folder/(name+'.mp4')
        count,duration=imageio_ffmpeg.count_frames_and_secs(str(source));assert count==expected,(name,count)
        dest=review/(name+'.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(source),
            '-c:v','libx264','-crf','30','-preset','medium','-threads','1','-an',str(dest)],check=True)
        check,_=imageio_ffmpeg.count_frames_and_secs(str(dest));assert check==expected
        video_info.append(dict(name=name,frames=count,seconds=duration,source_bytes=source.stat().st_size,
                               review_bytes=dest.stat().st_size,review_sha256=sha(dest)))
    files=[p for folder in output_dirs for p in folder.iterdir() if p.is_file() and p.suffix in ['.png','.jpg','.json','.npz','.csv','.mp4']]
    manifest={p.relative_to(root).as_posix():sha(p) for p in files}
    (root/'results/action_acceptance_SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
    (review/'video_checks.json').write_text(json.dumps(video_info,indent=2))
    with zipfile.ZipFile(root/'wam_action_acceptance_full.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(root))
        z.write(root/'results/action_acceptance_SHA256SUMS.json','SHA256SUMS.json')
        for name in ['match.json','initial_state.npy','matched_demo.npz','checkpoint_dataset_statistics.json','source_model.xml']:
            path=root/'data/matched_grasp_source'/name;z.write(path,'source/'+name)
    # No duplicate segment movies or large state arrays in the compact review.
    review_manifest={}
    with zipfile.ZipFile(root/'wam_action_acceptance_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if path.suffix in ['.png','.json','.csv']:
                arc=path.relative_to(root).as_posix();z.write(path,arc);review_manifest[arc]=sha(path)
        for path in review.iterdir():
            if path.suffix not in ['.mp4','.json']:continue
            arc='videos/'+path.name;z.write(path,arc);review_manifest[arc]=sha(path)
        for name in ['match.json','checkpoint_dataset_statistics.json']:
            path=root/'data/matched_grasp_source'/name;arc='source/'+name;z.write(path,arc);review_manifest[arc]=sha(path)
        z.writestr('SHA256SUMS.json',json.dumps(review_manifest,indent=2))
    print(json.dumps(dict(full_files=len(manifest),videos=video_info,
        review_zip_bytes=(root/'wam_action_acceptance_review.zip').stat().st_size,
        review_zip_sha256=sha(root/'wam_action_acceptance_review.zip'),
        full_zip_bytes=(root/'wam_action_acceptance_full.zip').stat().st_size),indent=2))


if __name__=='__main__':main()
