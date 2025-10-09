#!/usr/bin/env bash
# Convenience helper to submit a suite of SLURM jobs.
# Submit from the project root: bash experiments/submit_all_slurm.sh
# Customize PARTITION/ACCOUNT below as needed.

set -euo pipefail

PARTITION=${PARTITION:-gpu}
ACCOUNT_FLAG=${ACCOUNT:+--account=$ACCOUNT}

submit() {
  local sbatch_file=$1
  echo "[SUBMIT] sbatch --partition=$PARTITION ${ACCOUNT_FLAG:-} $sbatch_file"
  sbatch --partition="$PARTITION" ${ACCOUNT_FLAG:-} "$sbatch_file"
}

# Create logs dir
mkdir -p results/logs

submit experiments/slurm/all_experiments.sbatch
submit experiments/slurm/agn_only.sbatch
submit experiments/slurm/ova_experiments.sbatch
submit experiments/slurm/ablation_agnnet.sbatch
submit experiments/slurm/param_scaling_ova.sbatch

echo "[DONE] Submitted all jobs. Use 'squeue -u $USER' to monitor. Logs in results/logs/."}