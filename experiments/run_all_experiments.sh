#!/usr/bin/env bash
# Robust runner: portable logging and ensure PyG wheels match Torch.

set -euo pipefail

# Local-only execution: remote orchestration removed

# -----------------------
# Logging (portable; no process substitution)
# -----------------------
LOG_DIR="../results/logs"
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
if [ ! -d "../simple_data" ]; then
  echo "[INFO] Downloading datasets to ../simple_data/ …"
  # gdown is already in requirements; falls back to your Google Drive folder
  (cd .. && gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder)
  mkdir -p ../simple_data
fi

# -----------------------
# 2) Experiment settings
# -----------------------
# Default epochs (ensure at least 20)
SEARCH_EPOCHS="${SEARCH_EPOCHS:-20}"
EPOCHS="${EPOCHS:-20}"
SAVE_DIR="../results/saved_models"
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
      python3 ../backend/hyperparameter_search.py \
        --model "$model" \
        --dataset "$dataset" \
        --epochs "$SEARCH_EPOCHS" \
        --save-dir "$SAVE_DIR"
    else
      echo "[$(date +'%F %T')] Using existing model/config for model=$model dataset=$dataset"
    fi

    echo "[$(date +'%F %T')] Training model=$model dataset=$dataset"
    python ../backend/main.py \
      --model "$model" \
      --dataset "$dataset" \
      --epochs "$EPOCHS" \
      --load-model "$model_file" \
      --config "$config_file"
  done
done

echo "[DONE] All experiments completed."
