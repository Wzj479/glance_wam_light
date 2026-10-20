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

**逐回合动作数组未随 `results/stage_v1/` 发布。** 该目录在清单里只有 `.log`、`.json`、`.jsonl`，没有 `*_actions.npy`，而 `results/perturbation_pilot/` 的 npy 是收录的。`.gitignore` 也没有忽略 `*.npy`。因此从仓库检出得不到 `results/stage_v1/development/<model>/<condition>/task_XX_episode_YYY_actions.npy`，而它有三个用途：作为回放输入（`replay_failure_cases.py`）、作为失败清单的运动学输入（`inventory_failure_structure.py`）、作为夹爪诊断的对照动作（`--reference-actions`）。这三者都只读服务器归档，**开工前先确认文件在**：

```bash
test -f results/stage_v1/development/P/color_strong/task_08_episode_033_actions.npy
```

若已丢失，可用只含该回合的协议经 `run_robust_eval.py` 重新生成，代价是一次服务器启动。

## 后续 GPU 工作

2026-10-16 已在 A100 40GB 上通过真实视频 loader、预训练推理、动作模块三步更新、官方特征一致性、冻结参数哈希、显存及检查点重载检查，见 [GPU 部署验证记录](../results/gpu_validation/GPU部署验证记录.md)。随后完成 clean/aug 各200步和单一抓取任务的90回合配对闭环评测，见 [200步实验记录](../results/controlled_200/comparison/实验记录.md)。三模型在三个条件下均为10/10成功，未观察到成功率提升；尚未评测其他任务或实机。

### 三个模型的检查点位置不同

stage_v1 族里只有 C 和 P 是自己训出来的，base 是上游预训练权重，没有 `best.action.pt`，也不在 `results/stage_v1/` 下：

| 名称 | 路径 |
|---|---|
| base | `results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt` |
| C | `results/stage_v1/C_seed42/checkpoints/best.action.pt` |
| P | `results/stage_v1/P_seed42/checkpoints/best.action.pt` |

同一份归档以绝对路径记录了三者，见 `results/stage_v1/development/models.json`；直接照抄归档最可靠。写成 `results/stage_v1/base_seed42/...` 会在 `checkpoint_view_path()` 立即抛 `FileNotFoundError`，因为该目录不存在。

三个检查点的父目录都必须同时有 `config.yaml` 和 `dataset_statistics.json`，这是 `checkpoint_view_path()` 的硬要求。

### 扩展协议与归档复现校验

把 stage_v1 的评测集从 25–44 扩到 0–44 时，`run_robust_eval.py` 会遍历协议里的全部条件；协议只写 `color_strong` 即可，45 回合 × 3 模型 = 135 次。`clean` 在归档里是三模型全 20/20 的天花板，零区分度，而复现覆盖度由 color_strong 的 45 × 3 已足够。封存初态 45–49 由 `validate_protocol` 机械拒绝；只有协议声明非空 `held_back_reason` 时才放行，用于样本外检验（见 `results/heldout/`），误列仍会被拒。

`aggregate()` 只在单次 `--out` **内部**做模型间配对校验（`run_robust_eval.py` 中 `if 'base' in summaries` 一段），**不与既有归档比对**。因此扩展与归档的确定性复现必须显式请求：

```bash
day1/verify_expansion_against_archive.py \
  --archive results/stage_v1/development --expansion results/stage_v1/expand_ep0_44 \
  --models base C P --conditions color_strong \
  --out results/failure_structure/expand_ep0_44_reproducibility.json
```

它只读两个 `results.json`，无需 GPU。共享回合必须逐位一致于 `env_seed`、`init_state_sha256`、`post_settle_state_sha256`、`success`、`steps`；物理字段不一致说明不是同一次实验，结果字段不一致说明推理不确定。任一不一致即以退出码 2 停止，与 `plan_stage3.py` 对未复现对照的处理一致。

### 单次运行不能用来做模型归因

2026-10-20 在 A100 40GB 上对 state 33 重复运行同一检查点、同一协议、同一 env_seed 与同一初态
（初态与 settle 后状态的 sha256 与归档逐位一致，说明协议与环境正确）：

| 模型 | 本机重复运行 | 成功次数 | 步数 | 归档 |
|---|---|---|---|---|
| P | 8 次 | **7 / 8** | 79–80（1 次 300 失败） | 300 失败 |
| base | 2 次 | 2 / 2 | 80、81（81 与归档一致） | 81 成功 |
| C | 4 次 | 3 / 4 | 80（1 次 300 失败） | 80 成功 |

每次运行的 `action_sha256` 互不相同，**推理 run-to-run 不确定**。确定性开关不能修复它：
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 让步数稳定在 79 但 sha 仍不同；再加
`torch.use_deterministic_algorithms(True)`，P 的结果**反而翻转为 300 步失败**。

结论：state 33 上模型之间的差异被同一模型自身的运行间方差完全淹没，归档里"P 特有失败、C 成功"
的分裂是单次采样的结果。**任何基于单回合成功率的模型归因都不可靠**，包括本仓库归档导出的
失败清单与 `failure_structure` 归属。要对某个 state 归因，必须先测出该 state 的 run-to-run 翻转率
（每模型重复 N ≥ 5 次），只有翻转率接近 0 的 state 才支撑"模型特有"的表述。

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
  --driver-args --reference-actions $REF \
  --expect-init-sha256 240751828fcffe67789db1c374549b80c4f8427335183d585d6662be054878e0 \
  --expect-post-settle-sha256 fcb8b8a315a62c896845ad0031513b1c0f854c9aacdd646f66ee2b33942e1e1a --save-videos

# 第1步：干预轮。仅覆写动作56–75的夹爪通道；76起不覆写，也不加新的释放门控。
.venv/bin/python day1/run_gripper_feedback.py $COMMON --out results/gf33_hold56_75 \
  --driver-args --hold-close 56 75 --reference-actions $REF --save-videos
```

`run_gripper_feedback.py` 只接受 `--protocol/--condition/--ckpt/--model-name/--out/--port` 一类自有参数；
`--reference-actions`、`--expect-init-sha256`、`--expect-post-settle-sha256`、`--hold-close`、`--save-videos`
属于内层驱动脚本 `diagnose_gripper_feedback.py`，必须写在 `--driver-args` 之后才会被转发。
直接把它们写在 `--driver-args` 之前会以 `unrecognized arguments` 报错。

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

结果解读受 [失败结构清单](../results/failure_structure/失败结构清单.md) 约束：状态 28、34、35、42 在两个实验族中均为三模型共有失败，属基座策略脆弱性，微调不在其中。状态 33 曾被称为全集中唯一与被评模型相关的失败；**该说法已被 2026-10-20 的翻转率测量推翻**（P 在该状态 5/5 成功、C 2/5 失败，见[翻转率测量](../results/flip4/翻转率测量.md)）。28/34/35/42 仍是稳定的共有失败，仍不能推广；但 26/33 只能报告成功率分布，不能用于模型间比较。

第三段的分流写成规格，判定完直接落规格，不需要临场决定：

```bash
.venv/bin/python day1/plan_stage3.py --verdict results/gf33_hold56_75/verdict.json \
  --out results/gf33_hold56_75/stage3_plan.json
```

按搬运与放置两个轴分三条支路：`transported_and_placed` → 追问释放决策哪部分值得保留；`transported_not_placed` → 搬运恢复但放置仍坏，与 39/aug 同型，改攻放置阶段；`not_transported` → 约束在释放之前，改做可达性探针。任一前置条件不满足则输出 `branch=BLOCKED` 并以退出码 2 停止：零干预对照未复现存档、两个成功判据不一致、或窗口未真正覆写任何一步，都不允许进入第三段。

## 从既有动作检查点继续训练

`day1/finetune_action_head.py --init-action-checkpoint <path.action.pt>` 用一个既有动作检查点
初始化可训练作用域，而不是从基座 EMA 的动作头开始。初始化走 `action_checkpoint.read_action_checkpoint`，
它会**重新核对基座权重、基座 config 视图与基座 statistics 三个指纹**，所以在别的基座上训出来的
检查点会被直接拒绝，而不是被悄悄嫁接进来。初始化来源、来源的基座与张量数都会写进导出的 sidecar
（`init_action_checkpoint` / `init_action_base` / `init_action_tensors`），读者能区分"继承来的 run"与
"从基座动作头开始的 run"。

驱动层用环境变量传递，避免改动既有脚本的参数面：

```bash
INIT_ACTION_CHECKPOINT=results/stage_v1/C_seed42/checkpoints/best.action.pt \
OUT_DIR=results/stage_v1/P33_initC_seed42 \
EPISODE_SPLIT=/root/autodl-tmp/wam_glance_project/glance_wam_light/results/grasp_episode_split.json \
bash day1/run_phase_finetune_a100_40g.sh P
```

`EPISODE_SPLIT` 必须写成**基座服务器上的绝对路径**：`split_manifest.json` 记录的是
`Path(...).resolve()` 之后的路径，路径不同会让 `split_sha256` 与 stage_v1 不同，从而失去可比性。
`UPSTREAM_COMMIT` 文件（或 `$WAM_UPSTREAM_COMMIT`）用于 tarball 导出：此时没有 `.git`，
但记录的上游 SHA 必须是真的，不能伪造一个。

## 失败机理的判定口径

`day1/replay_failure_cases.py --physics-only` 把归档轨迹按纯物理重放，逐状态记录接触、位姿、
夹爪与滑移；重放会重新核对初态与 settle 后状态的 sha256，并断言成功/失败与首次成功步与归档一致。
`day1/analyze_failure_mechanism.py` 从这些日志重算机理事实并打标签，**不采信任何叙述**：

| 标签 | 判据 |
|---|---|
| `released_before_both_pad_contact` | 存在释放动作，且释放发生在首次双指垫接触之前（含从未成夹） |
| `never_both_pad_contact` | 从未出现双指垫接触，且从未释放 |
| `grasped_never_lifted` | 有双指垫接触，但抬升从未超过 20 mm |
| `lifted_then_lost_contact` | 抬升超过 20 mm，之后失去接触且未接触盘子 |
| `placed_but_misaligned` | 抬升足够但末态 XY 偏差大于 30 mm |
| `success` | 末态满足成功判据 |

夹爪时序是最直接的证据：成功回合的闭合持续 32–34 步，失败回合只有 11–20 步。

**干预窗口的形状**沿用 state 33 的 `[释放动作, 释放动作+19]`，这样有效闭合长度与成功情形一致，
而不去按模型调参。窗口表在 `glance_handover/intervention_windows.json`，逐 state 的单回合协议
在 `glance_handover/protocol_state{28,34,35,42}_color_strong.json`。

```bash
cd /root/autodl-tmp/wam_glance_project/GlanceWAM
LIBERO_CONFIG_PATH=/root/autodl-tmp/wam_glance_project/LIBERO/libero \
  /root/wam_sim_env/bin/python <repo>/day1/replay_failure_cases.py \
  --source <repo>/results/stage_v1/development \
  --out <repo>/results/failure_mechanism_28_34_35_42 \
  --case color_strong:28 --case color_strong:34 --case color_strong:35 \
  --case color_strong:42 --case color_strong:33 \
  --models base C P --physics-only
cd <repo> && python3 day1/analyze_failure_mechanism.py \
  --replay results/failure_mechanism_28_34_35_42 \
  --out results/failure_mechanism_28_34_35_42/mechanisms.json
```

归档的开发期运行只保存了动作与日志、**没有视频**，所以相机末帧比对在这些 state 上不可用；
`replay_failure_cases.py` 现在把视频比对改为可选，缺失时记录 `terminal_video_available: false`，
物理级校验（初态 sha、settle 后 sha、成功与否、首次成功步）始终执行。

## 在线夹爪规则

`diagnose_gripper_feedback.py` 的 `--min-closed-steps N`：夹爪第一次闭合后，累计闭合步数未达
`N` 之前抑制所有张爪命令。它只用夹爪命令历史，不读归档轨迹、不读对象状态，因而是可部署的。
`--min-lift-mm L` 曾作为逃生口试过（抬升到位就放行），但在 state 28 上失败——瞬时抬升会被
误判为搬运完成，已不推荐。

```bash
cd /root/autodl-tmp/wam_glance_project/GlanceWAM
bash /root/autodl-tmp/run_p2.sh A 1 3 5
bash /root/autodl-tmp/run_p2.sh B 2 4
bash /root/autodl-tmp/run_p2_33.sh
cd <repo> && .venv/bin/python day1/analyze_intervention_reps.py \
  --runs results/p2 --baseline results/flip20/flip_map.json \
  --out results/p2/intervention_reps.json
```

## 在线规则的泛化与鲁棒性

```bash
cd /root/autodl-tmp/wam_glance_project/GlanceWAM
bash /root/autodl-tmp/run_p3.sh 0 && bash /root/autodl-tmp/run_p3.sh 1   # 跨模型+阈值
bash /root/autodl-tmp/run_p3_audit.sh                                    # 20 态零扰动审计
cd <repo> && .venv/bin/python day1/analyze_intervention_reps.py \
  --runs results/p3 --baseline results/flip20/flip_map.json --out results/p3/intervention_reps.json
```

## 翻转率协议

`day1/analyze_flip_rates.py` 汇总重复运行的回合级结果：每个 (模型, 回合) 的成功次数、步数区间、
**不同动作轨迹的条数**（由每次运行写下的 `*_actions.npy` 哈希得到），以及与归档结果的一致率。
只有当一个回合在所有重复里结论一致时，它才支撑"模型特有"的说法；一旦翻转，
模型之间的差异就没有解释力了。

```bash
.venv/bin/python day1/analyze_flip_rates.py \
  --runs results/flip4 --archive results/stage_v1/development \
  --model base=base/color_strong --model C=C/color_strong \
  --model P=P/color_strong --model P33_initC=P33_initC/color_strong \
  --out results/flip4/flip_rates.json
```

封存初态的强制点在 `eval_protocol.validate_protocol`，`RESERVED_STATES` 声明 `(libero_goal, 8)` 保留 45–49。任务 8 实际有 50 个初态，所以 45–49 是可达的，仅把注记写进 JSON 不算强制；所有入口都经由该校验，五个归档协议复验全部通过。
