#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${SPATIAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$PWD:/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/xr1:${PYTHONPATH:-}"
export STAGE1_OPTIMIZER=adamw

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PYTHON_BIN="${PYTHON_BIN:-/data-cfs/data3/shurenbin/envs/xr1-server/bin/python}"
ROOT="${SPATIAL_ROOT:-/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial-KV-Proj/xr1/runs/spatial_forcing_v2_4x_b32}"
MANIFEST="$ROOT/manifests/spatial6_stage1_train_val.jsonl"
DATASET="/data-cfs/data3/datasets/robocasa365-datasets"
MODEL_DIR="/data-cfs/data3/models/Xiaomi-Robotics-1-RoboCasa365"
MODEL_STATE="/data-cfs/data3/models/Xiaomi-Robotics-1-5B/model_states.pt"
ESM="/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/runs/esm1b_robocasa365_geometry_fresh_400k_p4_retry2/checkpoint_014000.pt"
FALCON="/data-cfs/data3/shurenbin/Xiaomi-Robotics-Spatial/train/esm_sink_token/vendor/FALCON"

if [[ ! -f "$MANIFEST" ]]; then
  "$PYTHON_BIN" build_spatial6_stage1_manifest.py --output-dir "$ROOT/manifests"
fi

COMMON=(--manifest "$MANIFEST" --dataset-root "$DATASET" --model-ckpt "$MODEL_STATE" --processor "$MODEL_DIR" --esm-checkpoint "$ESM" --falcon-root "$FALCON" --batch-size "${SPATIAL_BATCH_SIZE:-32}" --devices 2 --workers "${SPATIAL_WORKERS:-4}" --max-steps 20000 --target-gpu-gb 67 --limit-gpu-gb 70)

if [[ "${1:-}" == "calibrate" ]]; then
  exec "$PYTHON_BIN" -u spatial_forcing_xr1.py --experiment SF_qwen_L24 --calibration-only --calibration-steps "${CALIBRATION_STEPS:-8}" --output-dir "$ROOT/calibration_L24_bs${SPATIAL_BATCH_SIZE:-24}" "${COMMON[@]}"
fi

EXPERIMENTS=(SF_control SF_qwen_L24 SF_qwen_L27 SF_qwen_L32)
if [[ "$#" -gt 0 ]]; then
  EXPERIMENTS=("$@")
fi
if [[ "${#EXPERIMENTS[@]}" -ne 4 ]]; then
  echo "parallel formal launch requires exactly four experiments (one per 2-GPU pair)" >&2
  exit 2
fi

GPU_PAIRS_RAW="${SPATIAL_GPU_PAIRS:-0,1;2,3;4,5;6,7}"
IFS=';' read -r -a GPU_PAIRS <<< "$GPU_PAIRS_RAW"
if [[ "${#GPU_PAIRS[@]}" -ne 4 ]]; then
  echo "SPATIAL_GPU_PAIRS must contain four semicolon-separated GPU pairs" >&2
  exit 2
fi

PIDS=()
cleanup() {
  trap - INT TERM EXIT
  for PID in "${PIDS[@]:-}"; do
    kill "$PID" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for IDX in "${!EXPERIMENTS[@]}"; do
  EXP="${EXPERIMENTS[$IDX]}"
  PAIR="${GPU_PAIRS[$IDX]}"
  LOG="$ROOT/$EXP/launch.log"
  mkdir -p "$ROOT/$EXP"
  echo "launching $EXP on CUDA_VISIBLE_DEVICES=$PAIR; log=$LOG"
  CUDA_VISIBLE_DEVICES="$PAIR" "$PYTHON_BIN" -u spatial_forcing_xr1.py --experiment "$EXP" --output-dir "$ROOT/$EXP" "${COMMON[@]}" >"$LOG" 2>&1 &
  PIDS+=("$!")
done

status=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    status=1
  fi
done
trap - INT TERM
exit "$status"
