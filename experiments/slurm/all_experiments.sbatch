#!/bin/bash
#SBATCH --job-name=gnn_all
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --error=results/logs/slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Uncomment and set your account if required by your cluster
# #SBATCH --account=<YOUR_ACCOUNT>

# Usage notes:
# - Submit from project root: sbatch experiments/slurm/all_experiments.sbatch
# - Customize partition/account/resources as needed for your cluster.
# - Logs will be written under results/logs/.

set -euo pipefail

PROJECT_ROOT="$(pwd)"
echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"

# Activate environment (choose one approach)
if command -v module >/dev/null 2>&1; then
  module purge || true
  # Example modules; customize for your site
  module load cuda/12.1 || true
  module load anaconda || module load miniconda || true
fi

# Conda activation (edit env name/path accordingly)
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate gnn-env || echo "[WARN] Could not activate conda env 'gnn-env'. Ensure dependencies are installed."
fi

# Ensure logs dir exists
mkdir -p "$PROJECT_ROOT/results/logs"

# Optional: print GPU info
nvidia-smi || true

# Epoch overrides via environment
export SEARCH_EPOCHS=${SEARCH_EPOCHS:-20}
export EPOCHS=${EPOCHS:-50}

# Run the experiment driver
bash "$PROJECT_ROOT/experiments/run_all_experiments.sh"