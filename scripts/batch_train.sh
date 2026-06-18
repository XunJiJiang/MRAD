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

# 尝试的学习率 0.0004-0.0001 , 0.0005-0.0009
# 先使用 0.0004-0.0001 共训练 4 次, 然后使用 0.0005-0.0009 共训练 5 次, 最后使用 0.0001-0.0009 共训练 9 次
# 这里限制训练 9 次
# BATCH_SIZE=9

# 循环执行训练脚本
for i in $(seq 1 $BATCH_SIZE); do
  echo "Starting training iteration $i..."
  new_idx=$(($i + $last_checkpoint_idx))

  # 前4次使用 0.0004-0.0001 共训练 4 次, 然后使用 0.0005-0.0009 共训练 5 次
  # 判断是否在前4次
  # if [ $i -le 4 ]; then
  #   # 0.0004-0.0001 共训练 4 次
  #   lr=$(awk -v min=0.0001 -v max=0.0004 'BEGIN{srand(); print min+rand()*(max-min)}')
  # elif [ $i -le 9 ]; then
  #   # 0.0005-0.0009 共训练 5 次
  #   lr=$(awk -v min=0.0005 -v max=0.0009 'BEGIN{srand(); print min+rand()*(max-min)}')
  # else
  #   # 0.0001-0.0009 共训练 9 次
  #   lr=$(awk -v min=0.0001 -v max=0.0009 'BEGIN{srand(); print min+rand()*(max-min)}')
  # fi

  python $TRAIN_SCRIPT \
    --model_type "$MODEL_TYPE" \
    --dataset "$TRAIN_DATASET" \
    --data_path "$TRAIN_DATA_PATH" \
    --save_path "$CHECKPOINT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --device "$TRAIN_GPU"
    # --learning_rate "$lr"

  if [ "$new_idx" != -1 ]; then
    mkdir -p "${CHECKPOINT_DIR}_$new_idx"

    mv ${CHECKPOINT_DIR}/* "${CHECKPOINT_DIR}_$new_idx/"
    echo "Moved ${CHECKPOINT_DIR}/* to ${CHECKPOINT_DIR}_$new_idx"

    # 执行测试脚本
    python $TEST_SCRIPT \
      --model_type $MODEL_TYPE \
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
