# GlanceWAM Light：机器人控制与数据管线验收

面向 GlanceWAM 动作模块后训练的 CPU 验证工程。在 **2GiB 内存、无 GPU** 的服务器上完成 LIBERO 控制接口检查、官方仿真示范数据验收、初始状态匹配回放和动作时序敏感性实验。

**当前成果是控制链路和数据管线验收；尚未运行 WAM/VLA 模型推理或 GPU 后训练，也不是实机实验。** `day1/` 中同时保留后续训练与策略评测代码，它们的 GPU 集成尚待验证。

## 已完成的实测

| 实验 | 结果 | 证据 |
|---|---|---|
| 单轴控制与固定动作 | 12 项方向检查通过；两次 150 步回放状态最大误差 0；原图/变暗/颜色变化对照 | [控制测试说明](day1/CPU控制测试说明.md)、[产物](results/cpu_control_diag/) |
| 官方仿真示范数据 | 抽取抽屉与抓取各 10 条轨迹，共 2321 帧；按轨迹划分训练 16 条 / 验证 4 条 | [抓取数据说明](day1/抓取示范验收与微调接入.md)、[划分](results/drawer_grasp_episode_split.json) |
| 动作接口与状态匹配 | 24 项目标缩放、14 项物理方向检查通过；匹配 LeRobot episode 0 与原始 HDF5 demo_10，112 步抓取放置成功 | [接口验收说明](day1/动作接口与示范回放验收.md)、[回放结果](results/matched_grasp_replay/) |
| 时序敏感性 | 同一条示范的 9 个条件均成功；夹爪延迟 200ms 后最大抬升下降约 9.9%，未找到失败边界 | [实验说明](day1/时序控制敏感性实验说明.md)、[完整结果](results/control_sensitivity/) |

### 时序实验

固定相同初始状态、20Hz 控制和 136 步时长，比较基准、整体动作延后 50/100/200ms、仅夹爪延后 50/100/200ms、每 2/4 步刷新指令。

| 条件 | 最终任务成功 | 最大抬升 mm |
|---|---|---:|
| 基准 | 是 | 125.851 |
| 夹爪延迟 50ms | 是 | 124.263 |
| 夹爪延迟 100ms | 是 | 123.044 |
| 夹爪延迟 200ms | 是 | 113.429 |

![时序实验结果](results/control_sensitivity/sensitivity_summary.png)

![夹爪指令与实际开度](results/control_sensitivity/gripper_timing.png)

[观看基准与夹爪延迟 200ms 并排视频](results/control_sensitivity/baseline_vs_perturbed.mp4)（左：基准；右：夹爪延迟；137 帧、20fps、6.85 秒）。GitHub 如不内嵌播放，可下载 MP4。

**结论边界：** 9 个条件来自同一条示范，不是 9 个独立任务，也不是策略成功率。延迟和频率均按仿真时间计算，CPU 运行慢于实时；整体动作延后只是固定动作流的起始时移；降低刷新频率时重复的是增量命令，不等同于 WAM 动作块执行。三种视觉条件属于同轨迹图像处理，未测视觉扰动下的策略鲁棒性。为降低内存，诊断资产只缩小纹理，物理文件不变；不用于官方视觉榜单比较。

## 阅读与复现

1. [部署与输入数据说明](docs/REPRODUCE.md)：环境、固定版本、数据来源、现有服务器命令与新机器前置条件。
2. [完整实施方案](day1/实施方案.md)：动作模块后训练与后续闭环评测路线，GPU 部分未实测。
3. [实测分析](results/control_sensitivity/analysis.md)：逐条件数值、限制与视频说明。

目录：`day1/` 为代码和研究说明，`results/` 为已完成实验的曲线、视频、JSON/CSV 和状态数组，`docs/` 为部署与来源记录。仓库不包含模型权重、完整示范数据、第三方仿真资产或 Python 环境。

## 来源

- [GlanceWAM](https://github.com/linhanwang/GlanceWAM)，commit `6a51bdc5139e72e8ad525779db0c989e1af5abab`。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- [GlanceWAM 数据与权重](https://huggingface.co/datasets/LinhanWang/GlanceWAM)，数据 revision `0b9208dff852914f94be879f702cd40118255bce`。
- [原始 LIBERO 示范](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)，revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。

本仓库是配套研究与验收工程，与上述官方项目无隶属关系。第三方代码、数据和衍生产物的使用应遵循各自上游许可；此处不重新授予第三方材料许可。
