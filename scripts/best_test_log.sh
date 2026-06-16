#!/bin/bash

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--log_path)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --log_path is required and must have a value"
        echo "Usage: $0 --log_path <log_path>"
        exit 1
      fi
      LOG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 --log_path <log_path>"
      exit 1
      ;;
  esac
done

if [[ -z "$LOG_PATH" ]]; then
  echo "Error: --log_path is required"
  echo "Usage: $0 --log_path <log_path>"
  exit 1
fi

# 读取日志文件，提取测试结果并找到最佳结果
BEST_RESULT=""

OLD_IFS="$IFS"  #保存当前shell默认的分割符，一会要恢复回去
IFS=","                  #将shell的分割符号改为，“”

# 读取 log_path/log-*.csv 的最后一行
shopt -s nullglob
for file in ${LOG_PATH}/log-*.csv; do
  if [[ -f "$file" ]]; then
    LAST_LINE=$(tail -n 1 "$file" | tr -d '\r')
    # 共5项, objects,pixel_auroc,pixel_aupro,image_auroc,image_ap
    array=($LAST_LINE)     #分割符是“，”，"hello,shell,split,test" 赋值给array 就成了数组赋值

    MEAN_SCORE=$(echo "0" | bc)
    for i in "${array[@]}"; do
      if [[ "$i" == "mean" ]]; then
        continue
      fi
      echo "Current score: $i"
      MEAN_SCORE=$(echo "$MEAN_SCORE + $i" | bc)
    done
    MEAN_SCORE=$(echo "scale=8; $MEAN_SCORE / 4" | bc)

    echo "File: $file, Mean Score: $MEAN_SCORE"

    if [[ -z "$BEST_RESULT" || $(echo "$MEAN_SCORE > $BEST_RESULT" | bc) -eq 1 ]]; then
      BEST_RESULT="$MEAN_SCORE"
      BEST_FILE="$file"
    fi
  fi
done

IFS="$OLD_IFS"  #恢复shell默认分割符配置


if [[ -n "$BEST_FILE" ]]; then
  echo "Best result found in file: $BEST_FILE with mean score: $BEST_RESULT"
else
  echo "No log files found in $LOG_PATH"
fi
