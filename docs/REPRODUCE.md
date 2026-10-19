# 复现入口与已验证边界

## 已部署服务器

本次实测工作目录是 `/root/autodl-tmp/wam_glance_project`，不是 Git 检出目录。它已经包含 `cpu_sim_env/`、`LIBERO_CPU_DIAG/` 和已校验的 `data/`。上传仓库保留相同 `day1/`、`results/` 结构；直接克隆并不自动获得这些依赖。

复跑既有时序实验：

```bash
cd /root/autodl-tmp/wam_glance_project
cpu_sim_env/bin/python -m pytest -q day1/test_control_sensitivity.py
bash day1/run_control_sensitivity_cpu.sh
```

`--resume` 会跳过已有数值回合，但仍重新生成并检查三段视频；改变输入或协议后应使用新的输出目录。每个条件单独进程运行，避免累积原生仿真内存。数值回合峰值 RSS 519.41MiB，视频回合峰值 880.55MiB，本次 cgroup 无 OOM。

其他已完成实验的入口：

```bash
bash day1/run_control_cpu.sh
bash day1/run_demo_audit_cpu.sh
bash day1/run_grasp_audit_cpu.sh
bash day1/run_action_acceptance_cpu.sh
```

## 新机器准备

以下是重建依赖的入口，未在全新机器上端到端重新安装验证。版本参考 `requirements-cpu-observed.txt`；它是实际 CPU 环境的主要包版本快照，不是包含所有依赖的锁文件。GPU 安装脚本 `day1/install_libero_env.sh` 不等于本次低内存 CPU 安装方案。

1. Linux + Python 3.10，准备 OSMesa（Ubuntu 包 `libosmesa6`、`libgl1`）。建立仓库根目录下 `cpu_sim_env/`，安装上述 CPU 包及其依赖。
2. 将固定版本 LIBERO 检出为仓库根目录下 `LIBERO_CPU/`，按上游说明取得 assets，随后执行：

```bash
cpu_sim_env/bin/python day1/prepare_cpu_assets.py \
  --source LIBERO_CPU --out LIBERO_CPU_DIAG --texture-size 1024
```

此步骤复制资产并只缩小纹理、检查其他文件哈希；新输出目录不能已存在。完整资产初次解码阶段的内存可能高于常规回放。当前结果中保留了本次诊断环境配置；换机器后的绝对路径应重新生成，不可原样照搬。

3. 按 `day1/demo_download_manifest.json`、`day1/download_demo_data.py` 的参数和 [数据管线说明](../day1/真实示范数据管线验收.md) 获取固定 revision 的 LeRobot 数据。目录为 `data/libero_goal/`。CPU 验收使用真实 Parquet 和少量从原视频校验抽取的 PNG，不是完整官方训练 loader 验证。完整训练需要完整视频。
4. 对于匹配抓取回放，需将原始 HDF5 下载到本机，再用动作序列匹配导出：

```bash
cpu_sim_env/bin/python day1/match_grasp_source.py \
  --data-root data/libero_goal --episode 0 \
  --local-hdf5 /path/to/put_the_bowl_on_the_plate_demo.hdf5 \
  --out data/matched_grasp_source
```

原 HDF5 路径：数据仓库 `yifengzhu-hf/LIBERO-datasets`、revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`、文件 `libero_goal/put_the_bowl_on_the_plate_demo.hdf5`，SHA256 `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406`。网络正常时脚本也支持 HTTP Range。数据与资产不随 Git 上传。

5. 归一化往返检查另需固定版本 GlanceWAM 的官方接口代码 `data/interface_source/model2libero_interface.py`、`eval_libero.py`，以及官方检查点 `dataset_statistics.json` 放在 `data/matched_grasp_source/checkpoint_dataset_statistics.json`。原输入文件哈希见 [input_receipts.json](input_receipts.json)；获取位置和具体接入见动作接口验收文档。仅跑时序回放不需要导入这些官方代码或加载权重。
6. 完成数据和资产配置后运行 `bash day1/run_control_sensitivity_cpu.sh`。保留单进程、2 线程、`NUMBA_DISABLE_JIT=1`、OSMesa 的配置；不要在 2GiB 容器中同时加载 Torch 和渲染器。

## 产物核验

`publication_manifest.json` 记录本次上传的所有文件大小和 SHA256（不包含清单自身）。各实验内部的历史清单保持原样；它们可能引用未发布的原始输入，核查仓库完整性请以 publication manifest 为准。状态数组和视频都来自服务器实际运行，没有用合成结果替代。

## 不可从公开来源重建的资产

大部分依赖都能重建：GlanceWAM 代码 `linhanwang/GlanceWAM@6a51bdc5`、LIBERO 代码 `Lifelong-Robot-Learning/LIBERO@8f1084e3`（`day1/provenance.json` 记录了完整 commit）、base 权重 `LinhanWang/GlanceWAM`、LIBERO 数据集 `yifengzhu-hf/LIBERO-datasets@f13aa24a3da8c43c7225569f28c562979fa0e35a`，两套 Python 环境的配方在 `day1/install_libero_env.sh`。仿真栈是 mujoco 3.2.3 + robosuite 1.4.0 + bddl 1.0.1，不需要 dm_control；客户端 torch 2.5.1 CPU 版足够。世界模型底座是 `Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers`。

**两项不能重建，因此不进 Git：**

| 资产 | 体量 | 为什么不能重建 |
|---|---|---|
| `results/stage_v1/{P,C}_seed42/checkpoints/best.action.pt` | 每个 595 MB | 本项目自己采样训练的产物，上游没有 |
| `results/stage_v1/development/**/*_actions.npy` | 120 个，共 0.35 MB | 只能由那个检查点在归档协议下重跑产生 |

归档动作数组不是可有可无的附件：状态 33 诊断第 0 步的门禁就是「复现归档的 300 步动作」，缺了它整轮诊断作废。发布仓库的 862 条清单里没有任何文件能替代它们。

`last.action.pt` 被有意排除：只评估 best，多出的 595MB 没有用处。`base_seed42` 无需打包，可从 Hugging Face 下载。

打包与校验（无 GPU，约 15 秒）：

```bash
cd /root/autodl-tmp/wam_glance_project/GlanceWAM
.venv/bin/python /root/autodl-tmp/wam_glance_project/glance_wam_light/day1/package_irreproducible_assets.py \
  --root . --out /root/autodl-tmp/wam_glance_project/irreproducible_assets.tar
# 传到新机器后校验，退出码 0 表示 138 个文件全部一致
.venv/bin/python /root/autodl-tmp/wam_glance_project/glance_wam_light/day1/package_irreproducible_assets.py \
  --root . --out irreproducible_assets.tar --verify
```

预期 SHA256 清单随包内 `irreproducible_SHA256.json` 一并写出，也归档在 [irreproducible_assets](../results/irreproducible_assets/irreproducible_SHA256.json)，可在传输前先行比对。解包时保持 `results/` 相对路径，`paired_policy_server.py` 与评测脚本依赖该布局。

## 后续 GPU 工作

2026-10-16 已在 A100 40GB 上通过真实视频 loader、预训练推理、动作模块三步更新、官方特征一致性、冻结参数哈希、显存及检查点重载检查，见 [GPU 部署验证记录](../results/gpu_validation/GPU部署验证记录.md)。随后完成 clean/aug 各200步和单一抓取任务的90回合配对闭环评测，见 [200步实验记录](../results/controlled_200/comparison/实验记录.md)。三模型在三个条件下均为10/10成功，未观察到成功率提升；尚未评测其他任务或实机。

## 状态33夹爪干预闭环诊断

固定失败动作的开环反事实无法区分「搬运规划有问题」与「回放受限」，因为干预之后的六维动作仍来自原始观测。为此需要一次带反馈的闭环诊断：只覆写夹爪通道，六维动作在干预开始后按当前观测重新查询策略。

推理节奏以部署实现为准并被记录，不做假设。官方客户端每 `action_chunk_size` 步查询一次服务器，其间重放该动作块，因此一个块的六维来自一次推理。窗口前的动作未被改动，进入窗口后第一次查询的观测与存档一致，该次查询复现存档动作块；**分歧步由记录的动作块推导，不写死**。

```bash
cd /root/autodl-tmp/wam_glance_project/GlanceWAM
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PWD:/root/autodl-tmp/wam_glance_project/LIBERO"
export LIBERO_CONFIG_PATH=/root/autodl-tmp/wam_glance_project/LIBERO/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CKPT=results/stage_v1/P_seed42/checkpoints/best.action.pt
REF=results/stage_v1/development/P/color_strong/task_08_episode_033_actions.npy
COMMON="--protocol day1/protocol_state33_color_strong.json --condition color_strong \
  --ckpt $CKPT --model-name P --libero-python /root/wam_sim_env/bin/python --port 6694"

# 第0步：零干预对照。这轮必须复现存档的300步动作，否则整轮实验作废。
.venv/bin/python day1/run_gripper_feedback.py $COMMON --out results/gf33_control \
  --reference-actions $REF --expect-init-sha256 240751828fcffe67789db1c374549b80c4f8427335183d585d6662be054878e0 \
  --expect-post-settle-sha256 fcb8b8a315a62c896845ad0031513b1c0f854c9aacdd646f66ee2b33942e1e1a --save-videos

# 第1步：干预轮。仅覆写动作56–75的夹爪通道；76起不覆写，也不加新的释放门控。
.venv/bin/python day1/run_gripper_feedback.py $COMMON --out results/gf33_hold56_75 \
  --hold-close 56 75 --reference-actions $REF --save-videos
```

单回合协议 `day1/protocol_state33_color_strong.json` 由父协议推导而来。每回合输入只取决于 seed、suite、task、episode 与条件，与 episode 列表无关，因此该子集复现存档回合；其 env_seed 为 474245375，与存档一致。协议哈希有意不同，子集结果不与整批结果混放。

两轮的判定顺序固定：先看第0步是否复现存档动作（未复现则停手并排查确定性），再在干预轮读动作块来源、覆写前夹爪输出、接触与位姿，按「搬运/对齐是否恢复」二分。45–49 保持封存。夹爪符号为 `+1` 闭、`-1` 开，与官方客户端二值化一致；写反符号会静默变成「保持张开」，因此 `override_gripper` 对非二值输入直接报错。

判定在无 GPU 环境下执行，只读日志：

```bash
.venv/bin/python day1/analyze_gripper_feedback.py --run results/gf33_hold56_75 \
  --control-run results/gf33_control \
  --reference-actions $REF --out results/gf33_hold56_75/verdict.json
```

报告把搬运与放置分成两个轴：`verdict` 只回答干预是否让碗被搬起（越过 20mm 抬升标记时夹爪几何仍有接触），`outcome` 才把两者合并。**`transport_restored` 不等于成功**——39/aug 曾把碗抬起 100mm 仍然失败。`control.reproduced` 为 false 时，任何差异都不能归因于夹爪覆写。

滑移口径有一处刻意的差异需要记住：回放脚本日志列记的是世界系，`analyze_contact_transfer.py` 重算的是末端系，两者都被记录并在元数据中注明。闭环驱动跟随分析口径，使离线复核与闭环结果可直接比较；`test_slip_column_matches_the_frozen_analysis_convention` 锁定了这一点。

结果解读受 [失败结构清单](../results/failure_structure/失败结构清单.md) 约束：状态 28、34、35、42 在两个实验族中均为三模型共有失败，属基座策略脆弱性，微调不在其中。状态 33 是全集中唯一与被评模型相关的失败，因此干预轮结果只能对照这一属性解读，不能推广到共有失败状态。

第三段的分流写成规格，判定完直接落规格，不需要临场决定：

```bash
.venv/bin/python day1/plan_stage3.py --verdict results/gf33_hold56_75/verdict.json \
  --out results/gf33_hold56_75/stage3_plan.json
```

按搬运与放置两个轴分三条支路：`transported_and_placed` → 追问释放决策哪部分值得保留；`transported_not_placed` → 搬运恢复但放置仍坏，与 39/aug 同型，改攻放置阶段；`not_transported` → 约束在释放之前，改做可达性探针。任一前置条件不满足则输出 `branch=BLOCKED` 并以退出码 2 停止：零干预对照未复现存档、两个成功判据不一致、或窗口未真正覆写任何一步，都不允许进入第三段。

封存初态的强制点在 `eval_protocol.validate_protocol`，`RESERVED_STATES` 声明 `(libero_goal, 8)` 保留 45–49。任务 8 实际有 50 个初态，所以 45–49 是可达的，仅把注记写进 JSON 不算强制；所有入口都经由该校验，五个归档协议复验全部通过。
