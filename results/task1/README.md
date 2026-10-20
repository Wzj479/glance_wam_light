# 第二任务迁移检验（task 1 = put_the_bowl_on_the_stove）

目的：检验“过早松爪”机理与在线夹爪规则能否从 task 08 迁移到第二个抓取放置任务。

## 已就绪
- 协议：`protocols/protocol_task1_{a,b}.json`（init 0-4 / 5-9，color_strong，base）。
- 目标体已参数化：driver 新增 `--grasp-body`（默认 akita_black_bowl_1）与 `--target-body`（默认 plate_1）。
  task 1 的抓取物仍是 `akita_black_bowl_1`，放置目标是 `flat_stove_1`（据 bddl 确认），故运行时传 `--target-body flat_stove_1`。
- 官方成功由 LIBERO 的 goal 检查 `(On akita_black_bowl_1 flat_stove_1_cook_region)` 给出（env `done`），不依赖代理列。

## 未执行的原因（环境）
当前机器无 GPU（`/dev/nvidia*` 不存在，torch.cuda=False）且 cgroup 内存上限 2 GiB，
模型服务需加载 3 GB 权重 + 峰值 ~3.5 GiB 显存，无法启动（表现为 `Server startup failed` / 进程被 OOM Kill）。
需在 A100 机器上运行；命令见仓库发布说明或本目录上文脚本 `run_task1.sh`。

## 待跑后要看的
1. 失败回合的“首次闭爪步数”（成功应 ~32-34、失败 ~11-20）→ 是否同一机理；
2. 对该机理回合施加 `--min-closed-steps 30` 规则 → 救回率与副作用。
