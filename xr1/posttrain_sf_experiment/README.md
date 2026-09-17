# XR-1 Spatial Forcing post-training SF10

## 目的

比较两个从同一个 XR-1 base checkpoint 开始的 RoboCasa365 10-task post-training：

1. Official post-training baseline
2. Spatial Forcing post-training

共同起点：

/data-cfs/data3/models/Xiaomi-Robotics-1-5B/model_states.pt

这是 XR-1 base checkpoint，不是 RoboCasa365 finetune checkpoint。训练代码不会加载
/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365 的模型权重。

processor 子目录只保存 tokenizer、Qwen3-VL 配置和 preprocessing metadata，不包含模型权重。

## 数据

使用：

/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/manifests/robocasa365_sf10/train.jsonl

每个 task 按 episode ID 自然排序，最后两个完整 episode 作为 validation，其余作为 training。
没有 frame-level random split，train/val trajectory 完全不相交。入口会展开 episode 内 frame，
但 split 只继承 episode split。

共同设置：

- history = 4，interval = 2
- LEFT、RIGHT、WRIST 三视图，固定顺序
- Qwen official crop ratio = 0.95
- action horizon = 16
- official RoboCasa action normalization
- 每 GPU batch = 24，2 GPUs，global batch = 48
- gradient accumulation = 1
- max steps = 20000
- checkpoint 和 validation 每 1000 steps

## 两个实验的差异

Baseline：

- 只调用官方 XR-1 policy objective
- 包含 flow、frequency、choice、score loss
- 不构造 ESM、Spatial Forcing branch 或 alignment projector

Spatial Forcing：

- 使用同一个 XR-1 base checkpoint 和同一 policy data path
- 增加 frozen ESM teacher、alignment projector 和所需 LoRA 参数
- 总损失为 L_policy + alpha L_align，alpha = 0.5
- 默认从 Qwen layer 27 取 selected hidden
- student 只使用当前 frame 的 LEFT_t、RIGHT_t、WRIST_t
- Qwen 使用 official 0.95 crop
- ESM 使用 raw RGB 和原始 full-image preprocessing，不使用 0.95 crop
- ESM 37x37 full-image grid 经过 crop coordinate transform 后 bilinear sample 到 Qwen grid
- LEFT、RIGHT、WRIST 分别对齐，不做 view mean 或 global pooling
- ESM eval、no_grad、requires_grad=False，且不进入 optimizer

gradient audit 会检查 ESM grad 为 None、projector grad 大于 0、LoRA grad 大于 0。

## Logging 和 success evaluation

train loss、val loss 和官方 loss 项由 Lightning logger 记录。

offline loss 不能推断真实 task success。若已有外部 evaluator 输出，可在调用入口时传入
--success-eval-json；格式为：

    {"task_success": {"TaskName": 0.8, "...": 0.7}}

入口会在 validation boundary 记录 per-task success rate 和 mean success rate。未提供时会明确打印
success_evaluation not_available，不会自动启动 rollout evaluator。

日志目录：

posttrain_sf_experiment/logs/baseline/
posttrain_sf_experiment/logs/spatial_forcing/

checkpoint 目录：

posttrain_sf_experiment/checkpoints/baseline/
posttrain_sf_experiment/checkpoints/spatial_forcing/

## 启动

本次任务不启动训练。以后实际运行：

    bash scripts/run_baseline_2gpu.sh
    bash scripts/run_spatial_forcing_2gpu.sh

两个脚本都明确设置 CUDA_VISIBLE_DEVICES=0,1。

## 文件说明

- README.md：实验协议和差异说明
- train_posttrain_baseline.py：独立 official baseline 入口
- train_posttrain_spatial_forcing.py：独立 Spatial Forcing 入口
- configs/baseline.yaml：baseline 配置
- configs/spatial_forcing.yaml：Spatial Forcing 配置
- scripts/run_baseline_2gpu.sh：baseline 2-GPU launcher
- scripts/run_spatial_forcing_2gpu.sh：Spatial Forcing 2-GPU launcher
- processor/：仅 processor/tokenizer/runtime metadata snapshot
- runs/：resolved audits 和 runtime metadata
- logs/：TensorBoard/Lightning/console 日志
- checkpoints/：训练 checkpoint

## 独立性和旧实验

本目录不修改以下文件：

- ../train_xr1_baseline_official20.py
- ../spatial_forcing_xr1.py
- ../train_xr1_spatial_official20.py
- 其他已有实验文件和已有 runs

新入口复用 XR-1 仓库内的官方 mibot model、runner、collate 和 optimizer 接口，
但不 import 旧实验 entrypoint。新实验的 audit、logs 和 checkpoints 均写入
posttrain_sf_experiment，不污染旧实验目录。
