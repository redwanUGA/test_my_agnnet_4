#!/usr/bin/env bash
# Usage: run_param_scaling_ova.sh <MODEL> <DATASET> <EPOCHS> <HIDDEN> <NUM_LAYERS> <HEADS>
set -euo pipefail

if [ $# -lt 6 ]; then
  echo "Usage: $(basename "$0") MODEL DATASET EPOCHS HIDDEN NUM_LAYERS HEADS"
  exit 2
fi

MODEL="$1"
DATASET="$2"
EPOCHS="$3"
HIDDEN="$4"
NUM_LAYERS="$5"
HEADS="$6"

# Anchor paths
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Ensure datasets present (optional - prints note if missing)
if [ ! -d "$PROJECT_ROOT/simple_data" ]; then
  echo "[INFO] Datasets folder not found at $PROJECT_ROOT/simple_data. If needed, run the dataset download step from README."
fi

python3 "$PROJECT_ROOT/backend/main.py" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --epochs "$EPOCHS" \
  --hidden-channels "$HIDDEN" \
  --num-layers "$NUM_LAYERS" \
  --heads "$HEADS" \
  --ova-smote
