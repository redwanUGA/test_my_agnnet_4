#!/usr/bin/env bash
# Robust runner: venv, portable logging, and ensure PyG wheels match Torch.

set -euo pipefail

# If we are executing locally (default), delegate to the Vast.ai helper which
# rents a remote GPU, runs this script there, and tears the instance down. The
# remote invocation sets RUNNING_IN_VAST=1 to skip this block.
if [ "${RUNNING_IN_VAST:-0}" != "1" ]; then
  python vast_gpu_runner.py "$@"
  exit $?
fi

# -----------------------
# 0) Activate existing virtual environment
# -----------------------
if [ ! -d ".venv" ]; then
  echo "[ERROR] .venv not found. Please create it before running this script."
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# -----------------------
# Logging (portable; no process substitution)
# -----------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_experiments_$(date +'%Y%m%d_%H%M%S').txt"
LOG_PIPE="$LOG_DIR/.log_pipe_$$"

echo "Saving logs to $LOG_FILE"

mkfifo "$LOG_PIPE"
tee -a "$LOG_FILE" < "$LOG_PIPE" &
TEE_PID=$!

exec >"$LOG_PIPE" 2>&1

cleanup() {
  exec 1>&- 2>&-
  if kill -0 "$TEE_PID" 2>/dev/null; then
    wait "$TEE_PID" || true
  fi
  [ -p "$LOG_PIPE" ] && rm -f "$LOG_PIPE"
}
trap cleanup EXIT

# -----------------------
# 1) Download datasets if missing
# -----------------------
if [ ! -d "simple_data" ]; then
  echo "[INFO] Downloading datasets to simple_data/ …"
  # gdown is already in requirements; falls back to your Google Drive folder
  gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder
fi

# -----------------------
# 2) Experiment settings
# -----------------------
SEARCH_EPOCHS="${SEARCH_EPOCHS:-1}"
EPOCHS="${EPOCHS:-2}"
SAVE_DIR="saved_models"
mkdir -p "$SAVE_DIR"

models=(BaselineGCN GraphSAGE GAT TGAT TGN AGNNet)
datasets=("OGB-Arxiv" "Reddit" "TGB-Wiki" "MOOC")

# -----------------------
# 3) Run hyperparameter search (if needed) and training
# -----------------------
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    model_file="$SAVE_DIR/${model}_${dataset}.pt"
    config_file="$SAVE_DIR/${model}_${dataset}_params.json"

    if [ ! -f "$model_file" ] || [ ! -f "$config_file" ]; then
      echo "[$(date +'%F %T')] Hyperparameter search for model=$model dataset=$dataset"
      python hyperparameter_search.py \
        --model "$model" \
        --dataset "$dataset" \
        --epochs "$SEARCH_EPOCHS" \
        --save-dir "$SAVE_DIR"
    else
      echo "[$(date +'%F %T')] Using existing model/config for model=$model dataset=$dataset"
    fi

    echo "[$(date +'%F %T')] Training model=$model dataset=$dataset"
    python main.py \
      --model "$model" \
      --dataset "$dataset" \
      --epochs "$EPOCHS" \
      --load-model "$model_file" \
      --config "$config_file"
  done
done

echo "[DONE] All experiments completed."
