#!/bin/bash

# 获取当前脚本所在目录
source "$(dirname "${BASH_SOURCE[0]}")/var.sh"

source .venv/bin/activate

which python

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --from is required and must have a value"
        echo "Usage: $0 --from <from_idx> --to <to_idx>"
        exit 1
      fi
      FROM_IDX="$2"
      shift 2
      ;;
    --to)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --to is required and must have a value"
        echo "Usage: $0 --from <from_idx> --to <to_idx>"
        exit 1
      fi
      TO_IDX="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 --from <from_idx> --to <to_idx>"
      exit 1
      ;;
  esac
done

if [[ -z "$FROM_IDX" || -z "$TO_IDX" ]]; then
  echo "Error: --from and --to are required"
  echo "Usage: $0 --from <from_idx> --to <to_idx>"
  exit 1
fi

# 检查 TRAIN_GPU 是否为空，如果是，则使用默认值
if [ -z "$TRAIN_GPU" ]; then
  TRAIN_GPU="cuda:1"
fi

# 循环执行测试脚本
for ((i=FROM_IDX; i<=TO_IDX; i++)); do
  echo "Running test for index: $i"
  python $TEST_SCRIPT \
      --model_type "mrad-clip" \
      --dataset "$TEST_DATASET" \
      --data_path "$TEST_DATA_PATH" \
      --cache_dir "$CACHE_DIR" \
      --save_path "${LOG_DIR}/results" \
      --device "$TRAIN_GPU" \
      --checkpoint_path "${CHECKPOINT_DIR}_$i/mrad_clip_final.pth" \
      --model_index "$i"
done

echo "Batch training completed."
