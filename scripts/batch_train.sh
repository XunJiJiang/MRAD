#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/var.sh"

which python

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [-b|--batch <batch_size>]"
      exit 1
      ;;
  esac
done

# 默认执行次数为1
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=1
fi

# 读取 $CHECKPOINT_DIR 中的最后一次的结果的索引
latest_checkpoint=""
latest_number=-1
shopt -s nullglob
for dir in ${CHECKPOINT_DIR}_*/; do
  if [ -d "$dir" ]; then
    dir_name=$(basename "$dir")
    dir_number=$(echo "$dir_name" | grep -oE '[0-9]+')
    if [ -z "$latest_checkpoint" ] || [ "$dir_number" -gt "$latest_number" ]; then
      latest_checkpoint="$dir"
      latest_number="$dir_number"
    fi
  fi
done

last_checkpoint_idx=$latest_number

echo "Last checkpoint index: $last_checkpoint_idx"

# 循环执行训练脚本
for i in $(seq 1 $BATCH_SIZE); do
  echo "Starting training iteration $i..."
  new_idx=$(($i + $last_checkpoint_idx))

  python $TRAIN_SCRIPT \
    --model_type "$MODEL_TYPE" \
    --dataset "$TRAIN_DATASET" \
    --data_path "$TRAIN_DATA_PATH" \
    --save_path "$CHECKPOINT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --device "$TRAIN_GPU"

  if [ "$new_idx" != -1 ]; then
    mkdir -p "${CHECKPOINT_DIR}_$new_idx"

    mv ${CHECKPOINT_DIR}/* "${CHECKPOINT_DIR}_$new_idx/"
    echo "Moved ${CHECKPOINT_DIR}/* to ${CHECKPOINT_DIR}_$new_idx"

    # 执行测试脚本
    python $TEST_SCRIPT \
      --model_type "mrad-clip" \
      --dataset "$TEST_DATASET" \
      --data_path "$TEST_DATA_PATH" \
      --cache_dir "$CACHE_DIR" \
      --save_path "${LOG_DIR}/results" \
      --checkpoint_path "${CHECKPOINT_DIR}_$new_idx/mrad_clip_final.pth" \
      --model_index "$new_idx" \
      --device "$TRAIN_GPU"

  else
    echo "No released/ directory found to rename."
  fi
done

echo "Batch training completed."
