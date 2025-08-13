#!/usr/bin/env bash
# Robust runner: portable logging and ensure PyG wheels match Torch.

set -euo pipefail

# -----------------------
# Remote configuration (edit placeholders or set env vars)
# -----------------------
REMOTE_HOST="${REMOTE_HOST:-YOUR.SERVER.IP}"
REMOTE_USER="${REMOTE_USER:-youruser}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-~/agnnet_remote}"

# If we are executing locally (default), perform remote orchestration. The remote
# invocation sets RUN_REMOTE=1 to skip this block and run the actual experiments.
if [ "${RUN_REMOTE:-0}" != "1" ]; then
  if [ "$REMOTE_HOST" = "YOUR.SERVER.IP" ]; then
    echo "Please set REMOTE_HOST/REMOTE_USER (or edit placeholders in run_all_experiments.sh)."
    exit 1
  fi
  repo_dir="$(pwd)"
  remote="${REMOTE_USER}@${REMOTE_HOST}"
  remote_path="${REMOTE_DIR}"

  echo "[LOCAL] Preparing remote directory at ${remote}:${remote_path}"
  ssh -p "$REMOTE_PORT" "$remote" "mkdir -p '${remote_path}' && [ -n \"${remote_path}\" ] && [ \"${remote_path}\" != \"/\" ] && rm -rf \"${remote_path}\"/*"

  echo "[LOCAL] Copying repository to remote..."
  scp -P "$REMOTE_PORT" -r "$repo_dir"/* "$remote":"$remote_path"

  echo "[LOCAL] Running experiments on remote..."
  remote_cmd="cd '${remote_path}' && python3 -m pip install -r requirements.txt && RUN_REMOTE=1 bash run_all_experiments.sh"
  ssh -p "$REMOTE_PORT" "$remote" "$remote_cmd"

  echo "[LOCAL] Fetching results back to local machine..."
  mkdir -p logs saved_models
  scp -P "$REMOTE_PORT" -r "$remote":"${remote_path}/logs" .
  scp -P "$REMOTE_PORT" -r "$remote":"${remote_path}/saved_models" .

  echo "[LOCAL] Done."
  exit 0
fi

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
