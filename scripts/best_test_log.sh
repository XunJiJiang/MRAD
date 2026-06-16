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

# 低于 92 的数量
LOWER_THAN_92_COUNT=0
# 92 - 92.1 的数量
BETWEEN_92_AND_921_COUNT=0
# 92.1 - 92.2 的数量
BETWEEN_921_AND_922_COUNT=0
# 92.2 - 92.3 的数量
BETWEEN_922_AND_923_COUNT=0
# 92.3 - 92.4 的数量
BETWEEN_923_AND_924_COUNT=0
# 92.4 - 92.5 的数量
BETWEEN_924_AND_925_COUNT=0
# 92.5 - 92.6 的数量
BETWEEN_925_AND_926_COUNT=0
# 92.6 - 92.7 的数量
BETWEEN_926_AND_927_COUNT=0
# 92.7 - 92.8 的数量
BETWEEN_927_AND_928_COUNT=0
# 92.8 - 92.9 的数量
BETWEEN_928_AND_929_COUNT=0
# 92.9 - 93 的数量
BETWEEN_929_AND_930_COUNT=0
# 高于 93 的数量
HIGHER_THAN_93_COUNT=0

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

    if (( $(echo "$MEAN_SCORE < 92" | bc -l) )); then
      ((LOWER_THAN_92_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92 && $MEAN_SCORE < 92.1" | bc -l) )); then
      ((BETWEEN_92_AND_921_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.1 && $MEAN_SCORE < 92.2" | bc -l) )); then
      ((BETWEEN_921_AND_922_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.2 && $MEAN_SCORE < 92.3" | bc -l) )); then
      ((BETWEEN_922_AND_923_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.3 && $MEAN_SCORE < 92.4" | bc -l) )); then
      ((BETWEEN_923_AND_924_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.4 && $MEAN_SCORE < 92.5" | bc -l) )); then
      ((BETWEEN_924_AND_925_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.5 && $MEAN_SCORE < 92.6" | bc -l) )); then
      ((BETWEEN_925_AND_926_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.6 && $MEAN_SCORE < 92.7" | bc -l) )); then
      ((BETWEEN_926_AND_927_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.7 && $MEAN_SCORE < 92.8" | bc -l) )); then
      ((BETWEEN_927_AND_928_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.8 && $MEAN_SCORE < 92.9" | bc -l) )); then
      ((BETWEEN_928_AND_929_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.9 && $MEAN_SCORE < 93" | bc -l) )); then
      ((BETWEEN_929_AND_930_COUNT++))
    elif (( $(echo "$MEAN_SCORE >= 92.8 && $MEAN_SCORE < 0" | bc -l) )); then
      echo "Error: Invalid score"
    else
      ((HIGHER_THAN_93_COUNT++))
    fi

    if [[ -z "$BEST_RESULT" || $(echo "$MEAN_SCORE > $BEST_RESULT" | bc) -eq 1 ]]; then
      BEST_RESULT="$MEAN_SCORE"
      BEST_FILE="$file"
    fi
  fi
done

IFS="$OLD_IFS"  #恢复shell默认分割符配置


if [[ -n "$BEST_FILE" ]]; then
  echo "< 92.0      : $LOWER_THAN_92_COUNT"
  echo "92.0 =~ 92.1: $BETWEEN_92_AND_921_COUNT"
  echo "92.1 =~ 92.2: $BETWEEN_921_AND_922_COUNT"
  echo "92.2 =~ 92.3: $BETWEEN_922_AND_923_COUNT"
  echo "92.3 =~ 92.4: $BETWEEN_923_AND_924_COUNT"
  echo "92.4 =~ 92.5: $BETWEEN_924_AND_925_COUNT"
  echo "92.5 =~ 92.6: $BETWEEN_925_AND_926_COUNT"
  echo "92.6 =~ 92.7: $BETWEEN_926_AND_927_COUNT"
  echo "92.7 =~ 92.8: $BETWEEN_927_AND_928_COUNT"
  echo "92.8 =~ 92.9: $BETWEEN_928_AND_929_COUNT"
  echo "92.9 =~ 93.0: $BETWEEN_929_AND_930_COUNT"
  echo "> 93.0      : $HIGHER_THAN_93_COUNT"
  echo "Best result found in file: $BEST_FILE with mean score: $BEST_RESULT"
else
  echo "No log files found in $LOG_PATH"
fi
