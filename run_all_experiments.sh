#!/usr/bin/env bash
# Robust runner: venv, portable logging, and ensure PyG wheels match Torch.

set -euo pipefail

# -----------------------
# 0) Virtualenv (isolate from system Python)
# -----------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging build

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
# 1) Install Python requirements
# -----------------------
REQ_TMP=".tmp_requirements_no_extensions.txt"
# Remove PyG extension packages (installed later with the correct wheel index)
# in a case-insensitive manner
awk 'BEGIN{IGNORECASE=1} /^[[:space:]]*(torch-scatter|torch-sparse)(\b|[=<> ]).*/ {next} {print}' requirements.txt > "$REQ_TMP"

echo "[INFO] Installing core requirements…"
pip install -r "$REQ_TMP"

# -----------------------
# 1b) Ensure PyG binary wheels match your Torch (prevents building torch-scatter/torch-sparse from source)
# -----------------------
python - <<'PY'
import re, torch, sys, subprocess
ver = torch.__version__  # e.g., "2.3.1+cu121" or "2.3.1"
m = re.match(r'^(\d+\.\d+)', ver)
if not m:
    print(f"[WARN] Could not parse torch version from {ver}", flush=True)
    sys.exit(0)

major_minor = m.group(1)
# Determine CUDA tag understood by PyG wheel index
cuda = torch.version.cuda
if cuda is None:
    tag = f"torch-{major_minor}+cpu"
else:
    # cuda like "12.1" -> cu121
    cu = "cu" + "".join(cuda.split("."))
    tag = f"torch-{major_minor}+{cu}"

index = f"https://data.pyg.org/whl/{tag}.html"
pkgs = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
cmd = [sys.executable, "-m", "pip", "install", "-U"] + pkgs + ["-f", index]
print("[INFO] Installing PyG extensions from:", index, flush=True)
print("[INFO] Command:", " ".join(cmd), flush=True)
subprocess.check_call(cmd)
PY

# -----------------------
# 2) Download datasets if missing
# -----------------------
if [ ! -d "simple_data" ]; then
  echo "[INFO] Downloading datasets to simple_data/ …"
  # gdown is already in requirements; falls back to your Google Drive folder
  gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder
fi

# -----------------------
# 3) Experiment settings
# -----------------------
SEARCH_EPOCHS="${SEARCH_EPOCHS:-1}"
EPOCHS="${EPOCHS:-2}"
SAVE_DIR="saved_models"
mkdir -p "$SAVE_DIR"

models=(BaselineGCN GraphSAGE GAT TGAT TGN AGNNet)
datasets=("OGB-Arxiv" "Reddit" "TGB-Wiki" "MOOC")

# -----------------------
# 4) Run hyperparameter search (if needed) and training
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
