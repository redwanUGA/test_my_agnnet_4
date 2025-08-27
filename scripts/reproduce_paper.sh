#!/usr/bin/env bash
set -euo pipefail

# One-command entrypoint to reproduce paper artifacts on Unix.
# Steps: (a) ensure data present; (b) run experiments; (c) write logs to results/; (d) produce results/summary.json.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs results/configs
TS=$(date +"%Y%m%d_%H%M%S")
LOG="results/logs/reproduce_${TS}.txt"

# Ensure datasets exist (simple_data folder). If missing, attempt to download via gdown.
if [ ! -d simple_data ]; then
  echo "[reproduce] simple_data/ not found. Attempting download using gdown..." | tee -a "$LOG"
  echo "Please see DOWNLOAD_INSTRUCTIONS.md for details." | tee -a "$LOG"
  python -m pip install -q gdown >> "$LOG" 2>&1 || true
  python -m gdown 'https://drive.google.com/drive/folders/1iZE_Cg5wAk_94Uk1DgNrOLiqp4F6cbfZ?usp=sharing' --folder --output simple_data >> "$LOG" 2>&1 || true
fi

# Run experiments (adjust as needed to match Figures/Tables mapping in README)
if [ -f experiments/run_all_experiments_agn_only.sh ]; then
  bash experiments/run_all_experiments_agn_only.sh >> "$LOG" 2>&1
elif [ -f experiments/run_all_experiments.sh ]; then
  bash experiments/run_all_experiments.sh >> "$LOG" 2>&1
else
  echo "No comprehensive experiments shell script found. Running a default baseline sanity." | tee -a "$LOG"
  python backend/main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 5 >> "$LOG" 2>&1
fi

# Produce summary.json from logs
python scripts/make_summary.py results/logs results/summary.json >> "$LOG" 2>&1

echo "Reproduction complete. See $LOG and results/summary.json"
