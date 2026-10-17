# A100 40GB 部署与真实数据验证记录

日期：2026-10-16。服务器工作目录：`/root/autodl-tmp/wam_glance_project`。

本次已实际完成预训练推理、动作模块三步前向/反向/AdamW 更新、冻结参数检查、完整模型导出、官方加载入口重载及重载后推理。尚未运行闭环控制评测，不能声称抓取成功率或鲁棒性提升。

## 部署结果

- GPU：NVIDIA A100-PCIE-40GB；容器内存上限 72GiB。
- Python 3.11.17；PyTorch 2.10.0+cu128；CUDA runtime 12.8；BF16 可用。
- GlanceWAM 固定提交 `6a51bdc5139e72e8ad525779db0c989e1af5abab`。
- Diffusers 固定提交 `4ca863323d550842e7d0122efd57d84d9b75d1cf`，版本 0.39.0.dev0。
- 项目源码：`/root/autodl-tmp/wam_glance_project/GlanceWAM`。
- 配套验收仓库：`/root/autodl-tmp/wam_glance_project/glance_wam_light`。
- 29 个模型和数据文件，共 9.085GiB；包括基础 DiT、VAE、调度器、官方 LIBERO 检查点、完整 goal 数据和官方 UMT5 缓存。所有文件校验大小；LFS 文件另核对上游 SHA256，其余文件记录实际 SHA256。未下载或加载 UMT5 编码器。
- 网络：PyPI/PyTorch 站点正常；Hugging Face 使用镜像但固定 revision；GitHub 受限时通过本地 Git bundle 部署固定提交，未替换为未知版本。
- 已验证 TorchCodec/PyAV FFmpeg6 随机访问视频路径；27 项 CPU 测试及 5 个子测试通过。

## 实测结果

| 阶段 | 结果 | 峰值分配显存 GiB | 峰值保留显存 GiB |
|---|---|---:|---:|
| 官方预训练模型推理 | 输出 `(1,8,7)` 动作与 `(1,16,1,28,56)` 目标 latent，均有限 | 3.451 | 3.479 |
| 三步动作模块更新 | 非零有限梯度，AdamW 实际更新 | 5.208 | 5.494 |
| 含训练后验证 | 优化器状态驻留时计算干净/扰动验证损失 | 5.760 | 5.904 |
| 完整导出重载后推理 | 官方加载入口与推理均通过 | 3.451 | 3.479 |

显存来自 PyTorch allocated/reserved 计数，不等于整卡 NVML 总占用。每步训练计时约 0.52/0.40/0.31 秒，包含真实样本读取、冻结特征提取、动作反向与优化器更新，不含参数哈希、验证和保存。预训练推理首次 13.34s，后两次 0.988/0.984s；每次都重新生成未来目标，不是缓存目标下的快速控制周期。

## 数据、冻结与损失

使用此前按轨迹验收的 10 条抓取示范：8 条训练、2 条验证，766 个训练帧；本次只取固定 2 个验证窗口做连通性检查。输入为完整官方视频经官方 loader 读取，不使用稀疏审阅 PNG 代替训练数据。

冻结 DiT、VAE、目标时间投影和 tap embedding；只更新 `action_model.*` 中参与无状态 LIBERO 训练的参数，未使用的 `state_encoder` 冻结。可训练参数 148,094,471。batch=1，梯度累积=1，AdamW，3 步，50% 图像扰动；使用检查点的 franka min/max 归一化，夹爪原始 0/1 训练标签不变。

动作损失为官方 flow matching。三步损失分别为 0.033564、0.001113、0.001394，对应不同样本与噪声，不作为收敛趋势。梯度范数均非零有限；冻结参数训练前后 SHA256 一致，动作参数发生变化。

干净验证损失 0.012070→0.012443；扰动验证损失 0.012988→0.013140。本次未观察到验证损失改善；3 步与 2 个验证窗口仅用于工程可运行性验证。`best.pt` 仍是 step 0，正式更新后的模型为 `last.pt`。

## 本次发现并修复的问题

首次训练在更新前被官方特征一致性检查拦下：自定义路径只为 DiT 开启 BF16 autocast，而用于比对的官方前向在包含 VAE 的完整上下文中启用 autocast。两者条件特征最大绝对差达到 0.5，未放宽原有 `atol=rtol=1e-3` 校验阈值。

为 `extract_frozen_features` 补上相同的 BF16 autocast 范围后，一致性检查通过：条件 `(1,3136,1536)`，动作目标 `(1,8,7)`。首次失败的日志和输出目录保留为 `action_finetune_3steps`，成功结果是 `action_finetune_amp_3steps`。

另为检查点配置准备脚本增加 `--data-mix libero_goal`，避免继承四套件配置而访问未下载数据。上游代码提交保持原样，修改仅在配套 `day1/` 文件。

## 服务器使用入口

```bash
source /root/autodl-tmp/wam_glance_project/deploy_env.sh
cd /root/autodl-tmp/wam_glance_project/GlanceWAM

# 查看实测指标
cat results/gpu_validation/action_finetune_amp_3steps/summary.json

# 用训练后的模型推理；建议每次更换输出目录
python day1/smoke_action_head.py --mode infer \
  --ckpt results/gpu_validation/action_finetune_amp_3steps/checkpoints/last.pt \
  --out results/my_inference_check
```

完整模型检查点：

```text
/root/autodl-tmp/wam_glance_project/GlanceWAM/results/gpu_validation/action_finetune_amp_3steps/checkpoints/last.pt
```

重跑三步训练请使用新的 `OUT_DIR`，沿用已验收 episode split。部署脚本位于工作目录，审阅副本在 `deployment/gpu/`。`run_gpu_validation.sh` 是首次尝试命令记录，原输出目录存在时不会覆盖训练；修正后的训练入口为 `validate_action_training.sh`。导出的 pt 为推理模型，不含 AdamW 状态，不能用于精确恢复优化器续训。

下一步可从原始检查点分别开展 clean/aug 的 200 步对照，再做固定初始状态的闭环评测。本次不自动扩大训练。GPU 进程已退出，模型与数据保留服务器。

全部详细 JSON、配置、数据划分、日志和检查点文件哈希见本目录；不在结果包中复制大体积模型权重。
