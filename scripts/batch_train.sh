#!/bin/bash
source "$(dirname "${BASH_SOURCE[0]}")/var.sh"

which python

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    # 是否使用多张卡同时进行两个训练任务
    -g|--gpus)
      TRAIN_GPU="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [-b|--batch <batch_size>] [-g|--gpus <cuda:0,cuda:1,...>]"
      exit 1
      ;;
  esac
done

# 默认执行次数为1
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=1
fi

if [ -z "$TRAIN_GPU" ]; then
  TRAIN_GPU="$_TRAIN_GPU"
fi

parse_gpu_list() {
  local raw_gpu_list="$1"
  local -n output_array="$2"
  local raw_gpu

  output_array=()
  IFS=',' read -ra raw_gpu_list_array <<< "$raw_gpu_list"
  for raw_gpu in "${raw_gpu_list_array[@]}"; do
    raw_gpu="${raw_gpu#${raw_gpu%%[![:space:]]*}}"
    raw_gpu="${raw_gpu%${raw_gpu##*[![:space:]]}}"
    if [ -n "$raw_gpu" ]; then
      output_array+=("$raw_gpu")
    fi
  done
}

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

# 多卡训练时，分配GPU资源
# 使用逗号分隔的GPU列表，例如 "cuda:0,cuda:1,cuda:2"
parse_gpu_list "$TRAIN_GPU" GPU_ARRAY

if [ ${#GPU_ARRAY[@]} -eq 0 ]; then
  echo "No valid GPU found in TRAIN_GPU: $TRAIN_GPU"
  exit 1
fi

run_job() {
  local iteration="$1"
  local new_idx="$2"
  local current_gpu="$3"
  local job_checkpoint_dir="${CHECKPOINT_DIR}_${new_idx}"
  local train_log_file="${CHECKPOINT_DIR}/train_${new_idx}.log"
  local test_log_file="${LOG_DIR}/test_${new_idx}.log"
  local result_dir="${LOG_DIR}/results_${new_idx}"

  mkdir -p "$job_checkpoint_dir" "$LOG_DIR" "$result_dir"

  echo "[Task ${iteration}] Starting training on ${current_gpu}"
  echo "[Task ${iteration}] Training log: $train_log_file"

  python "$TRAIN_SCRIPT" \
    --model_type "$MODEL_TYPE" \
    --dataset "$TRAIN_DATASET" \
    --data_path "$TRAIN_DATA_PATH" \
    --save_path "$job_checkpoint_dir" \
    --cache_dir "$CACHE_DIR" \
    --device "$current_gpu" \
    > "$train_log_file" 2>&1

  echo "[Task ${iteration}] Training completed"
  echo "[Task ${iteration}] Starting testing on ${current_gpu}"
  echo "[Task ${iteration}] Testing log: $test_log_file"

  python "$TEST_SCRIPT" \
    --model_type "mrad-clip" \
    --dataset "$TEST_DATASET" \
    --data_path "$TEST_DATA_PATH" \
    --cache_dir "$CACHE_DIR" \
    --save_path "$result_dir" \
    --checkpoint_path "$job_checkpoint_dir/mrad_clip_final.pth" \
    --model_index "$new_idx" \
    --device "$current_gpu" \
    > "$test_log_file" 2>&1

  echo "[Task ${iteration}] Testing completed"
}

# 每张 GPU 启动一个 worker，同一张卡上的任务会顺序执行，避免并发抢占
run_gpu_worker() {
  local gpu_index="$1"
  local current_gpu="$2"
  local gpu_count="$3"
  local iteration
  local new_idx

  for ((iteration = gpu_index + 1; iteration <= BATCH_SIZE; iteration += gpu_count)); do
    new_idx=$((iteration + last_checkpoint_idx))
    run_job "$iteration" "$new_idx" "$current_gpu"
  done
}

gpu_count=${#GPU_ARRAY[@]}
job_pids=()
for gpu_index in "${!GPU_ARRAY[@]}"; do
  current_gpu="${GPU_ARRAY[$gpu_index]}"
  run_gpu_worker "$gpu_index" "$current_gpu" "$gpu_count" &
  job_pids+=("$!")
done

for job_pid in "${job_pids[@]}"; do
  wait "$job_pid"
done

echo "All training and testing tasks are completed."
