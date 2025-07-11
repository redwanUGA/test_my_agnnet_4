# Graph Neural Network Experiments

This repository contains a small collection of scripts for experimenting with various graph neural network (GNN) architectures. The code supports loading several datasets and running pre-configured training sessions.

## Contents
- `data_loader.py` – Utilities for downloading and loading graph datasets. Supports **OGB-Arxiv**, **Reddit**, **TGB-Wiki**, and **MOOC**.
- `models.py` – Implementation of baseline GCN and GraphSAGE models along with research prototypes such as `AGNNet`.
- `train.py` – Reusable routines for full-batch and sampled mini-batch training.
- `main.py` – Entry point that iterates through a set of experiment configurations.
- `DOWNLOAD_INSTRUCTIONS.md` – Steps for obtaining large datasets from the provided Google Drive folders.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt  # if provided
   ```
   PyTorch and PyTorch Geometric must be installed with versions that match your system and CUDA setup.
2. Download the datasets following [DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md).
3. Run the experiments:
   ```bash
   python main.py
   ```

The default experiments will run a few epochs for each model/dataset pair and print accuracy statistics.

## Notes
- The datasets are large and therefore not stored in this repository.
- Modify the experiments listed in `main.py` to adjust hyperparameters or add new models.
