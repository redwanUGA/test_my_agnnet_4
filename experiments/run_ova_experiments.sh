#!/usr/bin/env bash
# Run One-vs-All (OVA) experiments with per-class SMOTE across models and datasets.
# Uses main.py --ova-smote which averages per-class metrics.

set -euo pipefail

# -----------------------
# Path anchoring (resolve to project root)
# -----------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# -----------------------
# Logging (portable; no process substitution assumptions beyond mkfifo)
# -----------------------
LOG_DIR="$PROJECT_ROOT/results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_ova_experiments_$(date +'%Y%m%d_%H%M%S').txt"
LOG_PIPE="$LOG_DIR/.log_pipe_ova_$$"

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
if [ ! -d "$PROJECT_ROOT/simple_data" ]; then
  echo "[INFO] Downloading datasets to $PROJECT_ROOT/simple_data/ …"
  (cd "$PROJECT_ROOT" && gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder)
fi

# -----------------------
# 2) Experiment settings
# -----------------------
EPOCHS="${EPOCHS:-20}"  # you can override: EPOCHS=50 ./run_ova_experiments.sh

models=(BaselineGCN GraphSAGE GAT TGAT TGN AGNNet)
datasets=("OGB-Arxiv" "Reddit" "TGB-Wiki" "MOOC")

# -----------------------
# 3) Run OVA-SMOTE training for each model/dataset
# -----------------------
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "[$(date +'%F %T')] OVA-SMOTE: model=$model dataset=$dataset"
    python3 main.py \
      --model "$model" \
      --dataset "$dataset" \
      --epochs "$EPOCHS" \
      --ova-smote
  done
done

echo "[DONE] OVA experiments completed."
