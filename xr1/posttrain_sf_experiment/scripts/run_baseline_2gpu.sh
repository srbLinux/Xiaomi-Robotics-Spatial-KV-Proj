#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT=/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial-KV-Proj/xr1/posttrain_sf_experiment
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export MAX_LENGTH=40000
export PYTHONPATH=$EXP_ROOT:/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial-KV-Proj/xr1

mkdir -p $EXP_ROOT/logs/baseline $EXP_ROOT/checkpoints/baseline $EXP_ROOT/runs/baseline
cd $EXP_ROOT
exec python train_posttrain_baseline.py \
  --manifest /data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/manifests/robocasa365_sf10/train.jsonl \
  --dataset-root /data-cfs/data3/datasets/robocasa365-datasets/pretrain \
  --model-ckpt /data-cfs/data3/models/Xiaomi-Robotics-1-5B/model_states.pt \
  --processor $EXP_ROOT/processor \
  --batch-size 24 \
  --devices 2 \
  --gradient-accumulation 1 \
  --max-steps 20000 \
  --workers 4 \
  --val-batches 16 \
  --output-dir $EXP_ROOT/runs/baseline \
  --checkpoint-dir $EXP_ROOT/checkpoints/baseline \
  --log-dir $EXP_ROOT/logs/baseline
