#!/usr/bin/env bash
# Install dependencies, download datasets, and run all model-dataset experiments.
# Portable logging (no process substitution), safe for /bin/bash on systems without >() support.

set -euo pipefail

# -----------------------
# Logging (portable way)
# -----------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_experiments_$(date +'%Y%m%d_%H%M%S').txt"
LOG_PIPE="$LOG_DIR/.log_pipe_$$"

echo "Saving logs to $LOG_FILE"

# Create a FIFO for tee-based logging without process substitution
mkfifo "$LOG_PIPE"
# Start tee in the background to write to file and stdout
tee -a "$LOG_FILE" < "$LOG_PIPE" &
TEE_PID=$!

# Redirect all stdout/stderr through the pipe
exec >"$LOG_PIPE" 2>&1

# Ensure cleanup of FIFO and tee on exit
cleanup() {
  # Close our stdout/stderr so tee can finish
  exec 1>&- 2>&-
  # Give tee a moment to flush, then kill if still around
  if kill -0 "$TEE_PID" 2>/dev/null; then
    wait "$TEE_PID" || true
  fi
  # Remove the pipe
  [ -p "$LOG_PIPE" ] && rm -f "$LOG_PIPE"
}
trap cleanup EXIT

# -----------------------
# 1. Install Python requirements
# -----------------------
pip install -r requirements.txt

# -----------------------
# 2. Download datasets if missing
# -----------------------
if [ ! -d "simple_data" ]; then
  echo "Downloading datasets..."
  gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder
fi

# -----------------------
# 3. Experiment settings
# -----------------------
SEARCH_EPOCHS=1
EPOCHS=2
SAVE_DIR="saved_models"
mkdir -p "$SAVE_DIR"

# Declare arrays (require bash; this script uses /usr/bin/env bash shebang)
models=(BaselineGCN GraphSAGE GAT TGAT TGN AGNNet)
datasets=("OGB-Arxiv" "Reddit" "TGB-Wiki" "MOOC")

# -----------------------
# 4. Run hyperparameter search (if needed) and training
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

echo "All experiments completed."
