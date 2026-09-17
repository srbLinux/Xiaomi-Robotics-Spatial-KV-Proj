#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT=/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial-KV-Proj/xr1/posttrain_sf_experiment
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export MAX_LENGTH=40000
export PYTHONPATH=$EXP_ROOT:/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial-KV-Proj/xr1

mkdir -p $EXP_ROOT/logs/spatial_forcing $EXP_ROOT/checkpoints/spatial_forcing $EXP_ROOT/runs/spatial_forcing
cd $EXP_ROOT
exec python train_posttrain_spatial_forcing.py \
  --manifest /data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/manifests/robocasa365_sf10/train.jsonl \
  --dataset-root /data-cfs/data3/datasets/robocasa365-datasets/pretrain \
  --model-ckpt /data-cfs/data3/models/Xiaomi-Robotics-1-5B/model_states.pt \
  --processor $EXP_ROOT/processor \
  --esm-checkpoint /data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/runs/esm1b_robocasa365_geometry_fresh_400k_p4_retry2/checkpoint_014000.pt \
  --falcon-root /data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/vendor/FALCON \
  --experiment SF_qwen_L27 \
  --alignment-layer 27 \
  --alpha 0.5 \
  --batch-size 24 \
  --devices 2 \
  --gradient-accumulation 1 \
  --max-steps 20000 \
  --workers 4 \
  --val-batches 16 \
  --output-dir $EXP_ROOT/runs/spatial_forcing \
  --checkpoint-dir $EXP_ROOT/checkpoints/spatial_forcing \
  --log-dir $EXP_ROOT/logs/spatial_forcing
