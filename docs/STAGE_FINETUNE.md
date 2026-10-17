# C/P：保守动作微调与阶段重采样

本轮只修改和审计代码，不启动真实训练。以下训练命令供后续明确执行；CPU合成测试不是机器人效果或A100新配置显存测量。

## 配方

两组都从原始`results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt`开始，更新交叉注意力`attn1`与`action_decoder`；冻结世界模型、VAE、文本、动作编码器、自注意力、FFN、时间/位置嵌入及未使用的状态编码器。冻结子模块保持eval模式，梯度仍可穿过冻结运算到达可训练交叉注意力。

| 项目 | C | P |
|---|---|---|
| 抽样 | 全轨迹均匀、有放回 | 每4次抽样：2次均匀、1次抓取、1次放置，随机排列 |
| 更新预算 | 100次更新，batch1、累积8 | 相同 |
| 学习率与调度 | 3e-6、warmup10、余弦降至0.1倍 | 相同 |
| 增强 | 关闭 | 关闭 |
| 种子 | 42 | 42 |
| 验证 | 0/25/50/75/100步 | 相同 |
| 保存 | 动作模块best/last，无优化器状态 | 相同 |

P每次更新的8个窗口按来源精确为4/2/2；均匀池本身也可能抽到关键阶段，不等于最终样本标签只有25%抓取。两阶段可以重叠，重叠窗口只计算一次联合动作损失。每轮抽样长度从766补齐到768，均为训练轨迹内的合法窗口；800个训练窗口预算不变。不同抽样方式会产生不同实际输入，这是对照的唯一配方差异。

## 数据与损失

沿用`results/grasp_episode_split.json`：训练`[0,43,47,54,68,69,80,87]`，验证`[52,72]`。两个命名空间必须分清：这些是LeRobot示范ID，不是LIBERO初始化状态ID39、32。

通过官方数据集的`get_trajectory_data`读取原始Parquet夹爪列；`1→0`闭爪、`0→1`释放，要求每条示范恰有一组配对变化。抓取窗口锚点范围`[close-8,close+24]`，放置范围`[release-24,release+8]`，两端包含、按该轨迹截断。只从训练entries构造抽样池；验证阶段标签只用于统计，不参与训练抽样。阶段依据开合事件，只是阶段代理，不声称拥有真实接触力或位姿标签。

原始8×7动作块及目标不变。前六维在官方装包前使用检查点`franka.action`上下界归一化，夹爪0/1保持不归一化，延续既有轨迹内未来帧/目标帧和末端填充。动作MSE仍直接调用官方`action_model(cond, targets, None)`：`x_tau=(1-tau)*noise+tau*actions`，速度目标`actions-noise`，七维共同参与原FM MSE。不添加BCE、夹持力损失、通道倍率或额外阶段损失倍率；不把39/32的失败动作作为正例。

固定验证窗口和未来目标采样种子，每次验证隔离并恢复torch RNG。正常与强偏色`RGB[0.7,1,1.3]`使用相同随机噪声，分别报告全局、抓取、放置FM均值及窗口数。强偏色仅记录，不参与本轮离线选模。

选模分数为`0.5*(grasp_loss/grasp_loss_step0 + place_loss/place_loss_step0)`。正常全局损失必须不超过step0的1.05倍，且分数严格低于当前best。step0分数为1，没有合格改进就保留原始模型。FM验证使用示范未来目标，不等同于生成目标条件下的闭环成功率。

## 服务器启动命令：尚未执行

代码部署位置：`/root/autodl-tmp/wam_glance_project/GlanceWAM/day1`。单进程python、单张A100，不用torchrun。

```bash
set -euo pipefail
source /root/autodl-tmp/wam_glance_project/deploy_env.sh
cd "$WAM_MODEL_ROOT"
unset AUDIT_ONLY
export CUDA_VISIBLE_DEVICES=0

# 以下两条才会开始真实训练；输出目录已有内容时拒绝覆盖。
bash day1/run_phase_finetune_a100_40g.sh C
bash day1/run_phase_finetune_a100_40g.sh P
```

默认输出为`results/stage_v1/C_seed42`和`P_seed42`。训练会先进行官方冻结特征一致性检查，并在首次更新检查梯度泄漏、动作参数变化与冻结哈希，结束时再次核对冻结哈希。`parameter_manifest.json`记录实际可训练与冻结参数名称、形状和数量；`metrics.jsonl`记录窗口ID、来源配额、联合损失、梯度范数、学习率、显存和耗时。

已完成的CPU真实数据审计：C/P的split哈希均为`d83ea4c02905c34900bff187696e76f3416026d5c43ef21ca8999ed1b2818252`，与上一轮一致；训练766窗口、固定验证26窗口；训练抓取池264、放置池258；验证两阶段各8窗口。可训练参数29,392,903，完整FP32动作state大小595,559,452字节（约596MB，不含序列化开销）。本轮A100实测C/P的峰值allocated均4.43GiB；训练进程分别348.99秒与363.09秒。

代码验收：10项CPU测试通过；另将真实发布底座的动作权重加载到CPU动作头，临时导出595,642,047字节的动作文件，严格重载后248个状态张量逐位相同，临时文件已删除。此过程没有模型forward/backward或优化器；测试中的小尺寸合成动作头单独核对了真实上游FM公式和冻结梯度，不构成真实数据训练。

## 只审计、不训练

```bash
source /root/autodl-tmp/wam_glance_project/deploy_env.sh
cd "$WAM_MODEL_ROOT"
CUDA_VISIBLE_DEVICES=-1 AUDIT_ONLY=1 OUT_DIR=results/stage_v1_audit/C_new \
  bash day1/run_phase_finetune_a100_40g.sh C
CUDA_VISIBLE_DEVICES=-1 AUDIT_ONLY=1 OUT_DIR=results/stage_v1_audit/P_new \
  bash day1/run_phase_finetune_a100_40g.sh P
```

该入口读取真实数据并生成split/phase manifest与抽样预览，在世界模型、优化器、forward/backward之前返回。额外的`audit_stage_recipe.py`只在CPU创建动作头结构统计参数、读取一条真实样本；不会加载世界模型或调用模型forward。审计输出和未来训练输出分开。

## 动作模块检查点加载与闭环入口

`best.action.pt`、`last.action.pt`包含完整动作模块state_dict，而不是整个世界模型，也不是只保存几层更新张量。记录原始底座的权重/config/归一化文件SHA256、冻结哈希、数据划分、阶段清单、选择分数与步数。缺少优化器状态，因此不支持精确恢复AdamW训练。

加载器检查动作文件SHA256、底座三项SHA256和动作键/形状；保留底座符号链接的配置视图。先加载原始机器人底座，再严格加载动作模块，最后按原部署方式转BF16。只对`.action.pt`使用此逻辑，旧完整检查点仍走原入口。换服务器时可以用`read_action_checkpoint`的base_override指定内容一致的新底座视图，但不能绕过哈希。

训练完成后可只做CPU完整性检查：

```bash
CUDA_VISIBLE_DEVICES=-1 python day1/action_checkpoint.py \
  --checkpoint results/stage_v1/P_seed42/checkpoints/best.action.pt
```

现有闭环runner已通过`paired_policy_server.py`接入新加载器。以下为独立重跑示例，输出目录必须为空。本轮已执行C/P的best闭环，复用同协议基线40回合，新增80回合，详细结果见[实验记录](../results/stage_v1/comparison/实验记录.md)：

```bash
python day1/run_robust_eval.py \
  --protocol protocols/grasp_color_holdout.json \
  --model base=results/day1_base/checkpoints/steps_15000_pytorch_model_ema.pt \
  --model C=results/stage_v1/C_seed42/checkpoints/best.action.pt \
  --model P=results/stage_v1/P_seed42/checkpoints/best.action.pt \
  --out results/stage_v1_development \
  --model-python "$WAM_MODEL_ROOT/.venv/bin/python" \
  --libero-python /root/wam_sim_env/bin/python \
  --libero-home /root/autodl-tmp/wam_glance_project/LIBERO --save-videos
```

状态25–44已用于诊断，只能称开发集。45–49继续预留作本轮最终小样本验收；不得因39、32单独恢复就声称鲁棒性提升。保留成功率、完成步数、脱持、松爪位姿与正常条件退化记录，不增加释放门控或恢复控制器混淆对照。

本轮实际产物位于服务器`GlanceWAM/results/stage_v1/`。C/P的best均选中第100步；best/last动作权重只保留服务器，仓库记录侧车JSON、SHA、大小与基座指纹。分阶段固定窗口验证为hindsight目标下的官方FM损失，闭环使用世界模型预测目标，损失降低不等价于闭环成功率上升。45–49本轮未使用。

本轮结论：C/P正常均20/20，偏色分别16/20与15/20，原始模型偏色15/20。P的阶段验证损失略低，但恢复状态26的同时新增33失败，未显示优于均匀采样C的闭环收益。此前39正常与32偏色在C/P均成功，不能将其单独归因于阶段采样。当前不据此追加训练或使用45–49调参。

后续仅回放33偏色的基线/C/P原动作，未训练：P的夹持接触在张爪前已不同于C，并在动作56提前持续张爪，最大抬升4.29mm，随后空手离开。定位为抓取建立与相位切换问题，不能归为盘上放置失败或断言是世界模型预测错误。2GB CPU关闭渲染后原始物理资产回放成功，峰值RSS约1.45GiB，详见[33号失败分析](../results/failure_replay_33/失败原因分析.md)。45–49仍未用于评测或调参。

单次夹爪反事实：仅P动作56–75设为闭合，其他动作逐项不变，干预前状态0–56逐位一致。状态56–76抬升4.29→32.51mm，固定状态60–75末端系漂移72.89→8.87mm；未出现双侧指垫同时接触，76恢复松爪时碗盘水平距离61.57mm，任务仍未成功。支持提前张爪对早期抬升失败有贡献，不能把固定延迟当作策略收益；本次无训练及45–49调参。见[反事实记录](../results/counterfactual_33_hold56_75/反事实回放记录.md)。
