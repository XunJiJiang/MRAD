

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${PROJECT_DIR}/.venv/bin/activate"

MODEL_TYPE="mrad-clip"

# Path configuration
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/released_${MODEL_TYPE}-Memory-Compression-Prototype-Selection"
LOG_DIR="${PROJECT_DIR}/logs/released_${MODEL_TYPE}-Memory-Compression-Prototype-Selection"
CACHE_DIR="${PROJECT_DIR}/cache/Memory-Compression-Prototype-Selection"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"
TEST_SCRIPT="${PROJECT_DIR}/test.py"

# Dataset configuration (users should modify these paths)
TRAIN_DATASET="visa"
TRAIN_DATA_PATH="/home/ts-cjh/Data/MRAD/data/spot-diff/data"
TEST_DATASET="mvtec"
TEST_DATA_PATH="/home/ts-cjh/Data/MRAD/data/mvtec_anomaly_detection"

# Training GPU
_TRAIN_GPU="cuda:1"

# Memory bank compression configuration
# 压缩方法: none(不压缩), kmeans, greedy, herding
COMPRESS_METHOD="kmeans"
# 每类原型数量 (如 500 表示正常500+异常500=1000总计)
N_PROTOTYPES=500
