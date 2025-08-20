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

4. Run all predefined experiments locally and capture logs:
  ```bash
  bash run_all_experiments.sh
  ```
  Or on Windows PowerShell/CMD:
  ```bat
  run_all_experiments.bat
  ```
  These scripts now run everything locally. No remote connections (SSH/SCP) are used.

  After completion, logs/ and saved_models/ will be present in your working directory.

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

## One-vs-All SMOTE Average Accuracy
You can run experiments that compute the average accuracy over n binary one-vs-all classifiers (n = number of classes). For each class, the training nodes are oversampled using SMOTE in a binary setting, a binary (2-class) model is trained, and the per-class validation/test accuracy is recorded and averaged.

Example:
```bash
python main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20 --ova-smote
```
Notes:
- For sampled datasets like Reddit, only validation accuracy is averaged (test accuracy is not computed in the sampled path, consistent with the existing pipeline).
- For TGN or very large graphs the per-class SMOTE step is skipped to avoid memory blow-ups.

## Parameter-Scaling OVA Experiments (CSV Output)
Generate seven variants per model by scaling trainable parameter counts from ~10M to ~100M, run OVA-SMOTE accuracy for each dataset/model variant, and save results to CSV.

Usage:
```bash
python run_param_scaling_ova.py --epochs 5 --output-csv results_param_scaling.csv \
  --datasets OGB-Arxiv Reddit TGB-Wiki MOOC \
  --models BaselineGCN GraphSAGE GAT TGAT AGNNet
```
Notes:
- The CSV will contain: dataset, model, param_count, average_ova_accuracy (validation average).
- TGN is skipped here because its parameter count is dominated by node-dependent memory state, making counts incomparable.
- You can adjust epochs to trade off runtime vs accuracy. The script searches simple grids over hidden size and number of layers to approximate target parameter counts.

## Ablation Study: AGNNet
This repo includes a ready-to-run, reproducible ablation study for AGNNet that isolates the contribution of its main components. The ablation compares a baseline AGNNet configuration against targeted removals/toggles across all supported datasets.

Key components under study:
- Predictive subgraph selection (enable/disable with `--disable-pred-subgraph`).
- k-annealing over epochs (enable with `--k-anneal` vs fixed k).
- Soft top-k attention cap (toggle `--soft-topk`).
- Forced self-loops for stability (`--no-self-loops` to ablate).
- Edge thresholding in attention pre-filtering (`--edge-threshold`).

Baseline configuration used by the scripts:
- Hidden size: 128, Layers: 3, Dropout: 0.25
- Optimizer: AdamW (lr=0.003, weight_decay=5e-4)
- LR schedule: cosine with warmup=500 epochs
- Label smoothing: 0.05
- AGN params: tau=0.9, k=8, k-anneal enabled (k-min=2, k-max=8), soft-topk enabled, edge-threshold=0.0

Ablation variants run:
1) Baseline (all components enabled)
2) No predictive subgraph selection (`--disable-pred-subgraph`)
3) No k-annealing (fixed k)
4) No soft top-k (hard cap only)
5) No self-loops (`--no-self-loops`)
6) No edge threshold (explicitly set `--edge-threshold 0.0`)

How to run (Windows PowerShell/CMD):
```bat
run_ablation_agnnet.bat
```
On Unix-like systems:
```bash
bash run_ablation_agnnet.sh
```

Notes:
- Both scripts run the ablations for: OGB-Arxiv, Reddit, TGB-Wiki, MOOC.
- Logs are saved under `logs/` with a timestamped filename. Console output includes epoch-wise metrics and selected configuration for each run.
- You can override the number of epochs by setting `EPOCHS` in your environment, e.g., `set EPOCHS=20` on Windows or `export EPOCHS=20` on Unix.
- The scripts will attempt to download datasets to `simple_data/` if not found (using the same Google Drive folder referenced by DOWNLOAD_INSTRUCTIONS.md).

## Notes
- The datasets are large and therefore not stored in this repository.
- After downloading they reside in `simple_data/` and are loaded directly
  from those `.pt` files.
- Use the command line flags in `main.py` to adjust hyperparameters or integrate new models.

