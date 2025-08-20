#!/usr/bin/env bash
set -euo pipefail

# AGNNet Ablation Study Runner (Unix)
# This script runs a compact set of ablations isolating key components of AGNNet.
# Results/logs are saved under logs/ with a timestamp; each run prints config.

LOG_DIR="../results/logs"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_ablation_agnnet_${TS}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

# Ensure datasets exist (uses Google Drive folder from README)
if [ ! -d ../simple_data ]; then
  echo "[INFO] Downloading datasets to ../simple_data/ …"
  (cd .. && gdown "https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing" --folder)
fi

# Default hyperparameters for the ablation baseline
EPOCHS=${EPOCHS:-50}
HIDDEN=128
LAYERS=3
DROPOUT=0.25
LR=0.003
WD=0.0005
TAU=0.9
K=8

# Datasets to evaluate. Reddit is sampled, others are full-batch.
DATASETS=(OGB-Arxiv Reddit TGB-Wiki MOOC)

for D in "${DATASETS[@]}"; do
  echo "$(date) ===== Dataset: $D ====="

  # 1) Baseline
  echo "$(date) AGNNet/base ds=$D"
  python ../backend/main.py \
    --model AGNNet \
    --dataset "$D" \
    --epochs "$EPOCHS" \
    --hidden-channels "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --tau "$TAU" \
    --k "$K" \
    --k-anneal \
    --k-min 2 \
    --k-max "$K" \
    --soft-topk \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay "$WD" \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05 \
    --edge-threshold 0.0

  # 2) No predictive subgraph
  echo "$(date) AGNNet/ablate:no_pred_subgraph ds=$D"
  python ../backend/main.py \
    --model AGNNet \
    --dataset "$D" \
    --epochs "$EPOCHS" \
    --hidden-channels "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --tau "$TAU" \
    --k "$K" \
    --k-anneal \
    --k-min 2 \
    --k-max "$K" \
    --soft-topk \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay "$WD" \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05 \
    --edge-threshold 0.0 \
    --disable-pred-subgraph

  # 3) No k annealing (fixed k)
  echo "$(date) AGNNet/ablate:no_k_anneal ds=$D"
  python ../backend/main.py \
    --model AGNNet \
    --dataset "$D" \
    --epochs "$EPOCHS" \
    --hidden-channels "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --tau "$TAU" \
    --k "$K" \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay "$WD" \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05 \
    --edge-threshold 0.0

  # 4) No soft top-k (hard cap only)
  echo "$(date) AGNNet/ablate:no_soft_topk ds=$D"
  python ../backend/main.py \
    --model AGNNet \
    --dataset "$D" \
    --epochs "$EPOCHS" \
    --hidden-channels "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --tau "$TAU" \
    --k "$K" \
    --k-anneal \
    --k-min 2 \
    --k-max "$K" \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay "$WD" \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05

  # 5) No self loops
  echo "$(date) AGNNet/ablate:no_self_loops ds=$D"
  python ../backend/main.py \
    --model AGNNet \
    --dataset "$D" \
    --epochs "$EPOCHS" \
    --hidden-channels "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --tau "$TAU" \
    --k "$K" \
    --k-anneal \
    --k-min 2 \
    --k-max "$K" \
    --soft-topk \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay "$WD" \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05 \
    --no-self-loops

  # 6) No edge threshold (explicit)
  echo "$(date) AGNNet/ablate:no_edge_threshold ds=$D"
  python ../backend/main.py \
    --model AGNNet \
    --dataset "$D" \
    --epochs "$EPOCHS" \
    --hidden-channels "$HIDDEN" \
    --num-layers "$LAYERS" \
    --dropout "$DROPOUT" \
    --tau "$TAU" \
    --k "$K" \
    --k-anneal \
    --k-min 2 \
    --k-max "$K" \
    --soft-topk \
    --ffn-expansion 2.0 \
    --optimizer adamw \
    --lr "$LR" \
    --weight-decay "$WD" \
    --lr-schedule cosine \
    --warmup-epochs 500 \
    --label-smoothing 0.05 \
    --edge-threshold 0.0

echo "[DONE] Ablation experiments completed."
