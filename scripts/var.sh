

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${PROJECT_DIR}/.venv/bin/activate"

MODEL_TYPE="mrad-clip"

# Path configuration
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/released_${MODEL_TYPE}"
LOG_DIR="${PROJECT_DIR}/logs/released_${MODEL_TYPE}"
CACHE_DIR="${PROJECT_DIR}/cache"
TRAIN_SCRIPT="${PROJECT_DIR}/train.py"
TEST_SCRIPT="${PROJECT_DIR}/test.py"

# Dataset configuration (users should modify these paths)
TRAIN_DATASET="visa"
TRAIN_DATA_PATH="/home/ts-cjh/Data/MRAD/data/spot-diff/data"
TEST_DATASET="mvtec"
TEST_DATA_PATH="/home/ts-cjh/Data/MRAD/data/mvtec_anomaly_detection"

# Training GPU
_TRAIN_GPU="cuda:1"
