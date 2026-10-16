# LIBERO CPU 控制测试与固定动作回放

目的：验证机器人控制接口、动作方向、回放一致性和视觉扰动管线。无需 WAM 权重、训练数据或 GPU。该结果属于仿真工程验证，不是策略成功率或鲁棒性结果。

## 启动

已有 LIBERO 环境时：

```bash
export SIM_PYTHON=/绝对路径/LIBERO/.venv/bin/python
export LIBERO_CONFIG_PATH=/绝对路径/LIBERO/libero
bash day1/run_control_cpu.sh --out results/cpu_control_v1
```

Linux 无头渲染需要 `libOSMesa.so.8`；脚本使用 `MUJOCO_GL=osmesa`。依赖版本参考 `install_libero_env.sh`，另外需 scipy、Pillow、imageio、imageio-ffmpeg。输出路径必须不存在，以防覆盖结果。

本次服务器部署路径（仅准备了当前 goal 任务 0 所需的源码和原始资产）：

```bash
cd /root/autodl-tmp/wam_glance_project
export PYTHONPATH="$PWD/LIBERO_CPU_DIAG"
export LIBERO_CONFIG_PATH="$PWD/LIBERO_CPU_DIAG/cpu_config"
export SIM_PYTHON="$PWD/cpu_sim_env/bin/python"
bash day1/run_control_cpu.sh --out "$PWD/results/cpu_control_next" \
  --asset-profile "$PWD/LIBERO_CPU_DIAG/cpu_asset_profile.json"
```

该 CPU 环境借用了已有 Python 3.10 环境的系统包，在自己的 venv 中安装仿真依赖，没有修改已有环境。换机器时建议使用 `install_libero_env.sh` 建独立环境。原始场景资产来自固定提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，下载后按本地官方源码中的 SHA256 核对，完整保留于 `LIBERO_CPU`。初始状态文件使用 `torch.load(weights_only=False)` 兼容新版 PyTorch，仅适用于信任的官方文件。

### 2GB 容器的内存适配

完整原始场景在 2GB 容器中被系统终止，当前使用单独的 `LIBERO_CPU_DIAG` 诊断副本，5 张大纹理降到最长边 1024。其余 140 个文件通过 SHA256 一致性检查，包括网格、XML、源码和初始状态。`cpu_asset_profile.json` 记录每张纹理的原尺寸、新尺寸和哈希；视频标题显示 `CPU-tex1024`。该副本用于控制和回放管线验证，不能当作原始视觉基准成绩。

已将原始资产与诊断资产的零输入回放进行全模拟状态比较：动作完全相同，状态最大绝对差为 0。结果保存于 `original_vs_diagnostic_physics.json`。该对照只覆盖这条零输入轨迹。

CPU 启动脚本默认设置 `NUMBA_DISABLE_JIT=1`，控制器执行同一套 Python/Numpy 函数，避免 JIT 编译峰值；线程数限制为 2，视频编码使用 1 个线程。初始状态由单独子进程读取，退出后才启动渲染，仿真进程不常驻 PyTorch。Linux 下每一组回放在独立子进程执行，退出后完整回收 OSMesa 原生分配。每组新建环境后使用 `hard_reset=False`，复用刚创建的模型进行 reset，再恢复相同初始状态。

创建诊断副本的命令（输出目录必须不存在）：

```bash
python day1/prepare_cpu_assets.py --source /path/LIBERO \
  --out /path/LIBERO_CPU_DIAG --texture-size 1024
```

内存充足后，可将 `PYTHONPATH` 和配置切回原始 `LIBERO_CPU`，去掉 `--asset-profile`，恢复原始纹理；需要比较默认 JIT 执行方式时设置 `NUMBA_DISABLE_JIT=0`。CPU 诊断结果不用于报告正式策略成功率或推理延迟。

## 实验协议

- 默认 `libero_goal`、任务 0、初始状态 0、种子 42，仿真控制步频 20 Hz，双相机 224×224。CPU 离线计算，视频按仿真时间编码为 20 FPS，不表示墙钟时间下实时运行。
- 每个测试重新创建环境，恢复同一官方初始状态，先执行 10 步零运动、夹爪 -1 指令，等待系统稳定。
- 15 组测试：零运动基线；x/y/z/rx/ry/rz 各正负一次；夹爪正负一次。
- 每组：10 步零运动、10 步单轴脉冲、10 步零运动。位姿脉冲默认归一化 0.1，夹爪脉冲 ±1。
- 非夹爪测试全程保持夹爪 -1。这里是持续发送该夹爪指令，不代表夹爪不动。夹爪正指令测试先发 -1 再发 +1；负指令测试先发 +1 再发 -1，避免已经打开时继续发送打开指令造成无响应。后半段维持目标夹爪指令，结合 qpos 和视频确认符号。
- 记录脉冲开始/结束和整段的末端平移、旋转向量、夹爪关节变化。零输入也可能有控制误差，不能要求每个无指令轴绝对为零。
- 固定动作默认依次小幅 z+、x+、y+、y-、x-，共 150 步；这是一段诊断动作，不保证抓取或完成任务。
- 固定序列从同一初始状态执行两遍；保存完整模拟状态，逐元素比较，默认绝对误差阈值 1e-7、相对误差 0。失败时保留报告并以非零状态退出。
- 三种画面来自每一步同一对相机图像：原图；亮度×0.7；RGB 增益 [0.9,1,1.1]。上排主相机，下排腕部相机，列为 clean/dark/color。图像按官方评测旋转 180 度。
- 为保持固定序列，测试即使出现任务 done 也继续运行到预定步数；任务 done 单独记录，不计为策略得分。

## 产物与阅读顺序

1. `controller.json`：实际控制器类型、输入/输出缩放、动作上下界。
2. `poses.csv`：各组每一步末端位置、xyzw 四元数、双指关节位置。
3. `axis_response.png`：六轴正负响应曲线（平移 mm、旋转 deg）；`report.json`：每组脉冲和整段位姿变化、重复回放误差、初始状态和动作哈希、包版本。
4. `zero.mp4`、`x_pos.mp4` 等：15 组诊断视频。
5. `fixed_replay.mp4`：固定动作的三种视觉条件对照视频；`fixed_replay_final.png` 为终帧。
6. `*.npz`：动作、末端位姿和每一步模拟状态。两次固定回放都保存，第二遍不重复编码视频。
7. `fixed_actions.npy`、`initial_state.npy`：可复用动作和初始状态。

先观察零输入漂移，再看正负单轴响应。平移重点对照对应世界坐标位移的符号、幅度和其他轴耦合；旋转用相对四元数转世界旋转向量，避免欧拉角跳变。不要把归一化动作 0.1 写成 0.1 米，也不要在未核对控制器前宣称旋转轴的坐标定义。

自定义已有动作序列时：

```bash
bash day1/run_control_cpu.sh --out results/my_replay --actions /path/actions.npy
```

输入必须是有限值 Nx7 数组，范围 [-1,1]，维度顺序和控制器一致。该选项替换固定回放部分，单轴诊断仍会执行。

视频的三列只是观测处理差异，物理轨迹相同。之后接入 WAM，让不同视觉条件分别影响动作，才开始比较策略鲁棒性。

## 本次实测结果

已在 2GB CPU 容器完成全部 15 组诊断和两次 150 步固定动作回放。单组仿真进程峰值 RSS 为 769.52 MiB；仿真进程未加载 PyTorch。完整运行需数十分钟，已有环境可以直接复用，无需重新安装。

下表是 0.5 秒脉冲结束相对开始时的主轴响应，输入幅度均为归一化 ±0.1：

| 维度 | 正指令响应 | 负指令响应 | 单位 |
|---|---:|---:|---|
| x | 9.860 | -9.913 | mm |
| y | 11.524 | -11.558 | mm |
| z | 11.447 | -11.511 | mm |
| rx | 6.768 | -6.761 | deg |
| ry | 5.638 | -5.666 | deg |
| rz | 6.438 | -6.444 | deg |

12 项主轴方向检查全部通过；其他轴仍有动力学耦合，原始数据保留于 `poses.csv` 和各组 `.npz`。夹爪 +1 使双指间距由约 78.47mm 降至 8.39mm，-1 由约 8.39mm 增至 71.64mm。

两次固定回放的动作序列相同，151 个采样状态向量逐元素一致，最大绝对误差为 0，状态 SHA256 一致。对照视频为 151 帧、20 FPS、7.55 秒、672×484；已验证可解码，并核对服务器与本地视频 SHA256 一致。

服务器实测输出：`/root/autodl-tmp/wam_glance_project/results/cpu_control_diag`。

`wam_cpu_control_results.zip` 包含 16 段视频、所有动作/状态数组、曲线、CSV、报告、环境版本、原始资产零输入参考轨迹和文件哈希清单；为控制下载大小，仅打包固定回放终帧和曲线两张 PNG，其余各组终帧保留在服务器。
