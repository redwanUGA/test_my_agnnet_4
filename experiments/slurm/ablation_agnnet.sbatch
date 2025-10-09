#!/bin/bash
#SBATCH --job-name=agn_ablate
#SBATCH --output=results/logs/slurm-%x-%j.out
#SBATCH --error=results/logs/slurm-%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
# #SBATCH --account=<YOUR_ACCOUNT>

set -euo pipefail
PROJECT_ROOT="$(pwd)"

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load cuda/12.1 || true
  module load anaconda || module load miniconda || true
fi

if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate gnn-env || true
fi

mkdir -p "$PROJECT_ROOT/results/logs"

bash "$PROJECT_ROOT/experiments/run_ablation_agnnet.sh"