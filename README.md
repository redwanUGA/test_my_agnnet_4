# Graph Neural Network Experiments

This repository contains a small collection of scripts for experimenting with various graph neural network (GNN) architectures. The code supports loading several datasets and running pre-configured training sessions.

## Contents
- `data_loader.py` – Utilities for downloading and loading graph datasets. Supports **OGB-Arxiv**, **Reddit**, **TGB-Wiki**, and **MOOC**.
- `models.py` – Implementation of baseline GCN and GraphSAGE models along with research prototypes such as `AGNNet`.
- `train.py` – Reusable routines for full-batch and sampled mini-batch training.
- `main.py` – Command line interface to run a single experiment.
- `DOWNLOAD_INSTRUCTIONS.md` – Steps for obtaining large datasets from the provided Google Drive folder.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
  Note: Some PyTorch Geometric extensions (e.g., torch-scatter, torch-sparse) may require CUDA-specific wheels. If needed, install them separately to match your Torch/CUDA version. When these extensions are missing, a slower fallback loader implemented in pure Python may be used.
2. Download the datasets from the provided Google Drive folder by following [DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md).
   The archives will create a `simple_data/` directory containing `.pt` files.
3. Run an experiment:
   ```bash
   python main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20
   ```
   Additional hyperparameters such as learning rate, dropout and hidden size can
   be supplied via command line flags (see `python main.py --help`). Defaults
   for these optional values are printed as part of the configuration output.

4. Run all predefined experiments and capture logs:
  ```bash
  bash run_all_experiments.sh
  ```
  Or on Windows PowerShell/CMD:
  ```bat
  run_all_experiments.bat
  ```
  By default, these scripts orchestrate a remote run over SSH. Set the placeholders at the top of the scripts or export environment variables:
  - REMOTE_HOST: server IP or hostname
  - REMOTE_USER: SSH username
  - REMOTE_PORT: optional, default 22
  - REMOTE_DIR: optional, remote working directory (default ~/agnnet_remote)

  Example (bash):
  ```bash
  export REMOTE_HOST=203.0.113.10
  export REMOTE_USER=ubuntu
  export REMOTE_PORT=22
  bash run_all_experiments.sh
  ```
  Example (Windows PowerShell):
  ```powershell
  $env:REMOTE_HOST="203.0.113.10"
  $env:REMOTE_USER="ubuntu"
  $env:REMOTE_PORT="22"
  .\run_all_experiments.bat
  ```
  After completion, logs/ and saved_models/ are copied back to your local machine.
  Requirements: OpenSSH client (ssh, scp) must be available on your system.

  Output is streamed to the console and a timestamped log is written to the
  `logs/` directory.

### Valid command line values
Models:
`BaselineGCN`, `GraphSAGE`, `TGAT`, `TGN`, `AGNNet`

Datasets:
`OGB-Arxiv`, `Reddit`, `TGB-Wiki`, `MOOC`

Default hyperparameters:
- `lr`: `0.01`
- `hidden-channels`: `64`
- `dropout`: `0.5`
- `weight-decay`: `5e-4`
- `num-layers`: `2`

## Notes
- The datasets are large and therefore not stored in this repository.
- After downloading they reside in `simple_data/` and are loaded directly
  from those `.pt` files.
- Use the command line flags in `main.py` to adjust hyperparameters or integrate new models.

