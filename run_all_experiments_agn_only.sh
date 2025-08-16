#!/usr/bin/env bash
# Runs ONLY AGNNet across datasets with recommended modifications and k/tau sweeps.
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +'%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/run_all_experiments_agn_only_${TS}.txt"
LOG_PIPE="$LOG_DIR/.log_pipe_agn_$$"

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

# Ensure datasets
if [ ! -d "simple_data" ]; then
  echo "[INFO] Downloading datasets to simple_data/ …"
  gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder
fi

EPOCHS=${EPOCHS:-50}
SAVE_DIR="saved_models_agn"
mkdir -p "$SAVE_DIR"

# Grids
DATASETS=("OGB-Arxiv" "Reddit" "TGB-Wiki" "MOOC")
KS=(2 4 8 16 32)
TAUS=(0.7 0.9 1.2)

for ds in "${DATASETS[@]}"; do
  for k in "${KS[@]}"; do
    for tau in "${TAUS[@]}"; do
      echo "[$(date +'%F %T')] AGNNet ds=$ds k=$k tau=$tau"
      python3 main.py \
        --model AGNNet \
        --dataset "$ds" \
        --epochs "$EPOCHS" \
        --hidden-channels 128 \
        --num-layers 3 \
        --dropout 0.25 \
        --tau "$tau" \
        --k "$k" \
        --k-anneal \
        --k-min 2 \
        --k-max "$k" \
        --soft-topk \
        --ffn-expansion 2.0 \
        --optimizer adamw \
        --lr 0.003 \
        --weight-decay 0.0005 \
        --lr-schedule cosine \
        --warmup-epochs 500 \
        --label-smoothing 0.05
    done
  done
  # Ablate predictive-subgraph module once per dataset (sanity check)
  echo "[$(date +'%F %T')] AGNNet ds=$ds (ablation: disable predictive subgraph)"
  python3 main.py \
    --model AGNNet \
    --dataset "$ds" \
    --epochs "$EPOCHS" \
    --hidden-channels 128 \
    --num-layers 3 \
    --dropout 0.25 \
    --tau 0.9 \
    --k 8 \
    --k-anneal \
    --k-min 2 \
    --k-max 8 \
    --soft-topk \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr 0.003 \
    --weight-decay 0.0005 \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05 \
    --disable-pred-subgraph

done

echo "[DONE] AGNNet-only experiments completed."
