# 真实示范数据管线验收

## 已实际完成

2026-10-15，在现有 2GiB CPU AutoDL 容器执行成功。没有加载世界模型、文本编码器、仿真环境，也没有进行训练。

这里的“真实示范”指官方发布的真实数据文件，来源为 **LIBERO 仿真人类示范**，不是现实机械臂采集，也不是随机合成动作。

| 项目 | 本次实测 |
| --- | --- |
| 任务 | open the middle drawer of the cabinet |
| 轨迹 | 10 条，1,359 行动作 |
| 训练轨迹 | 20、26、31、58、64、84、94、111；1,083 行 |
| 验证轨迹 | 42、90；276 行 |
| 交叉轨迹 / 交叉全局数据行 | 0 / 0 |
| 动作 | 7 维、全部有限值；20Hz 时间戳通过检查 |
| 样本 | 普通训练、普通验证、轨迹尾部各 1 个窗口 |
| 图像 | 16 张不同的原始视频帧，双相机；生成正常、暗光、颜色变化版本 |
| 进程峰值 RSS | 最终含输入 SHA 校验的一次运行：210.90234375 MiB |

每个版本的动作标签逐元素相等；尾部动作重复补齐、未来图像截断、任务索引和视频片段持续时间检查通过。

**发现：这 10 条示范夹爪动作全为 +1，标准差为 0。** 该子集适合先检查位姿控制训练链路；不能用它证明夹爪开合或抓取学习能力。后续加入放碗等任务，再统计夹爪分布。

## 数据来源及下载方式

- [官方数据仓库](https://huggingface.co/datasets/LinhanWang/GlanceWAM/tree/0b9208dff852914f94be879f702cd40118255bce/datasets/libero_goal_no_noops_1.0.0_lerobot)
- 数据版本：`0b9208dff852914f94be879f702cd40118255bce`
- 源码版本：`6a51bdc5139e72e8ad525779db0c989e1af5abab`
- 数据目录：`datasets/libero_goal_no_noops_1.0.0_lerobot`

动作表约 3.1MB；两路 AV1 视频约 330MB、各 52,042 帧、256×256、20fps。

服务器直连 Hugging Face 报错，hf-mirror 返回 403。实际做法：本地下载完整官方视频并对照官方 LFS SHA256 校验，用 OpenCV 抽取本次需要的 16 帧为无损 PNG；校验解码位置和 PTS；把原始 Parquet、元数据和图像缓存传入服务器。没有裁剪重编码视频，也没有重新生成动作。

主相机视频 SHA256：`477f644fa3d9103c4bb6ee2383155fce7bd8159c85fa2593b49ef4dfe39cc478`

腕部视频 SHA256：`b9e7c3e75f027348a3b3accb91066fb91ed79b557b2f27d1bf8caef7266213fd`

服务器再次核验原始动作表、元数据及每张 PNG 的 SHA256。`frame_receipt.json` 记录帧号、原视频哈希、解码 PTS 和 PNG 哈希。`download_receipt.json` 记录本地完整下载来源；其中列出视频，不表示服务器保存了完整视频。

当前服务器只具备本次样本需要的图像缓存，**不能拿该缓存直接持续训练全部帧**。GPU 恢复后复制完整视频，再由官方 loader 持续读取。

## 一条示范怎样变成训练样本

以训练轨迹 **episode=20** 为例，共 136 帧，局部帧号为 0…135。动作是控制命令，不应把这些数值直接解释为米或弧度；控制器还有缩放。源码字段 roll/pitch/yaw 对应这里的旋转命令分量，不能据字段名断言是欧拉角。

选 `t=10`，固定种子抽到 `g=14`：

```text
一条轨迹：图像序列 + 动作序列 + 语言任务
                     ↓ 在 t=10 切窗
图像槽 0：当前，第 10 帧
图像槽 1：世界模型未来监督，第 10+48=58 帧
图像槽 2：动作条件所用的 hindsight goal，第 10+14=24 帧

每个槽：主相机 224×224 | 腕部相机 224×224
image：3 个 RGB 图像，数组表示为 [3,224,448,3]，uint8

action：第 10…17 帧的 8 个动作 → [8,7]，float16
language："open the middle drawer of the cabinet"
goal_offset_rows：14
```

图像槽按角色排列，不是按时间升序排列。第 58 帧和第 24 帧作用不同；不能把它们都当作在线可观测输入。在训练中未来真值用于监督、目标真值用于 hindsight 条件，闭环部署按模型的目标生成逻辑运行，不能读取示范未来图像。

这条轨迹的主相机在共享视频内从 127.55 秒开始，所以局部第 10 帧对应 `128.05 秒 / 全局视频帧 2561`。**Parquet 全局行、轨迹局部帧、共享视频帧是三种索引**；`sample_trace.json` 逐一保留其映射。

前 6 维使用官方已发布的 `stats_gr00t.json` 中 action min/max：

```python
normalized = 2 * ((raw - minimum) / (maximum - minimum)) - 1
# maximum == minimum 时置 0；官方 min_max 分支不 clamp。
# 第 7 维 gripper 不执行这项归一化。
```

不从本次小子集重新估计归一化参数，也不分别拟合训练和验证归一化。使用的是预训练体系发布的全数据统计，不能宣称它是仅由本项目训练集拟合的统计。

尾部示例 `episode=20,t=133`：

```text
动作请求：133,134,135,136,137,138,139,140
实际读取：133,134,135,135,135,135,135,135
图像索引：133,135,135
请求 g=19，但实际目标只向后 2 帧；goal_offset_rows 仍记录 19。
```

这是当前 modality 元信息默认 `absolute=True` 导致的官方边界行为；不应根据物理动作是相对命令，就擅自改成补零。当前实现只是复现原始约定，没有额外添加尾部损失 mask。

每个 t 都可以生成一个动作窗口，因而这里有 1,359 个候选起点；本次只解码并可视化其中 3 个代表窗口，没有声称逐帧解码全部样本。

## 动作统计怎样看

`action_stats.csv` 按 7 维列出 min/max、均值、标准差、p01/p99、零值比例。统计范围是本次 10 条示范，包含训练和验证，只用于审计，不用于拟合训练归一化。

本次原始命令范围：

| 维度 | min | max |
| --- | ---: | ---: |
| x | -0.7366 | 0.7688 |
| y | -0.7366 | 0.9375 |
| z | -0.9375 | 0.5464 |
| roll | -0.1918 | 0.1146 |
| pitch | -0.2796 | 0.2936 |
| yaw | -0.2571 | 0.3504 |
| gripper | 1.0000 | 1.0000 |

## 在现有服务器复跑

项目绝对路径：`/root/autodl-tmp/wam_glance_project`

```bash
cd /root/autodl-tmp/wam_glance_project
bash day1/run_demo_audit_cpu.sh
```

脚本使用已存在的 `cpu_sim_env`，不安装包、不修改基础环境。核心依赖为 NumPy、Pillow、PyArrow、Matplotlib。当前缓存路径无需 Torch、CUDA、MuJoCo 或视频解码包。

输出目录：`/root/autodl-tmp/wam_glance_project/results/demo_data_audit`

```bash
cat /root/autodl-tmp/wam_glance_project/results/demo_data_audit/report.json
cat /root/autodl-tmp/wam_glance_project/results/demo_data_audit/sample_trace.json
```

联网环境重新下载完整视频：

```bash
cd /root/autodl-tmp/wam_glance_project
cpu_sim_env/bin/python day1/download_demo_data.py --root data/libero_goal
```

该命令不会使当前服务器网络故障自动消失。完整视频就位后，审计器可通过 imageio-ffmpeg 直接读取原视频；当前已有 PNG 时优先用缓存，若要验证完整视频解码，需另建一个不含 `frames` 的输入目录。

## 阅读和验收入口

- `audit_demo_data.py`：按轨迹读 Parquet → 校验索引 → 整轨迹划分 → 归一化 → 拼接图像 → 保存样本和统计。
- `prepare_demo_frame_cache.py`：断网服务器的少量图像缓存准备，当前固定复现这 10 条轨迹与 3 个窗口。
- `demo_download_manifest.json` / `download_demo_data.py`：固定版本下载和 LFS 哈希校验。
- `results/demo_data_audit/sample_windows.png`：3 行分别是训练、验证、尾部；3 列分别是当前、未来、目标。
- `appearance_comparison.png`：同一当前观测的正常、暗光和颜色变化。
- `split.json`：轨迹 ID 和全部对应全局数据行，可追踪重现。
- `train_clean.npz` 等：实际从数据产生的数组，`np.load(..., allow_pickle=False)` 即可检查。
- `SHA256SUMS.json`：输出文件校验清单。

## 本次验收的边界与下一步

通过的是原始示范数据契约和轻量参考预处理。PIL 双线性缩放与官方完整 loader 的像素级一致性尚未验证；未执行官方 TorchCodec loader、文本编码、VAE、世界模型或动作头前后向。轨迹划分仅用于本次适配验证，原模型可能在预训练时见过这些示范，不能声称未见任务泛化。

GPU 恢复后的下一步：完整视频就位，复用整轨迹划分，加载官方 loader，对照本报告检查局部帧号/动作/目标偏移；再跑 10 步动作头微调，验证冻结参数哈希、梯度归属、检查点重载和闭环动作输出。要训练夹爪能力则先加入有开合变化的示范任务。

源码依据：

- [LIBERO 动作顺序与归一化配置](https://github.com/linhanwang/GlanceWAM/blob/6a51bdc5139e72e8ad525779db0c989e1af5abab/examples/LIBERO/train_files/data_registry/data_config.py)
- [样本打包、目标偏移和边界补齐](https://github.com/linhanwang/GlanceWAM/blob/6a51bdc5139e72e8ad525779db0c989e1af5abab/glancewam/dataloader/gr00t_lerobot/datasets.py)
- [Normalizer 的 min_max 实现](https://github.com/linhanwang/GlanceWAM/blob/6a51bdc5139e72e8ad525779db0c989e1af5abab/glancewam/dataloader/gr00t_lerobot/transform/state_action.py)
