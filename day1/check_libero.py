"""Run with the separate LIBERO Python, from the GlanceWAM root."""
from pathlib import Path

import imageio.v2 as imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
try:
    env.seed(42)
    obs = env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step([0., 0., 0., 0., 0., 0., -1.])
    image = obs['agentview_image']
    assert image.shape == (128, 128, 3), image.shape
    dest = Path('results/day1/libero_render.png')
    dest.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(dest, image[::-1])
    print('LIBERO_RENDER_PASS', task.language, str(dest))
finally:
    env.close()
