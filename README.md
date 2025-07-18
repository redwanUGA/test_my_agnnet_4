# Graph Neural Network Experiments

This repository contains a small collection of scripts for experimenting with various graph neural network (GNN) architectures. The code supports loading several datasets and running pre-configured training sessions.

## Contents
- `data_loader.py` – Utilities for downloading and loading graph datasets. Supports **OGB-Arxiv**, **Reddit**, **TGB-Wiki**, and **MOOC**.
- `models.py` – Implementation of baseline GCN and GraphSAGE models along with research prototypes such as `AGNNet`.
- `train.py` – Reusable routines for full-batch and sampled mini-batch training.
- `main.py` – Command line interface to run a single experiment.
- `DOWNLOAD_INSTRUCTIONS.md` – Steps for obtaining large datasets from the provided Google Drive folders.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt  # if provided
   ```
   PyTorch and PyTorch Geometric must be installed with versions that match your system and CUDA setup.
2. Download the datasets following [DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md).
3. Run an experiment:
   ```bash
   python main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20
   ```
   Additional hyperparameters such as learning rate, dropout and hidden size can
   be supplied via command line flags (see `python main.py --help`). Defaults
   for these optional values are printed as part of the configuration output.

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
- Use the command line flags in `main.py` to adjust hyperparameters or integrate new models.

## Web Interface
An interactive UI is provided using Flask. The pages are stored in the
`templates/` directory and use Bootstrap for styling.
Launch it with:
```bash
python app.py
```
Open `http://localhost:5000` in your browser to download datasets or start
experiments. Logs refresh automatically.
