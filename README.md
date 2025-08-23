# Graph Neural Network Experiments

This repository contains a small, reproducible suite for experimenting with several graph neural network (GNN) architectures across multiple datasets. It provides:
- A unified CLI to run any model/dataset pair;
- Utilities for loading large preprocessed datasets;
- End-to-end scripts for all-experiments, OVA-SMOTE benchmarking, AGNNet ablations, and parameter-scaling sweeps.


## Getting Started
1) Create environment and install dependencies
- Unix/macOS:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Windows (PowerShell):
  ```powershell
  py -m venv .venv; .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
Note: Some PyTorch Geometric (PyG) extensions (e.g., torch-scatter, torch-sparse) may require CUDA-specific wheels. If needed, install wheels matching your Torch/CUDA version. When these extensions are missing, the code falls back to a pure-Python sampler (slower but functional).

2) Download datasets
- Follow [DOWNLOAD_INSTRUCTIONS.md](DOWNLOAD_INSTRUCTIONS.md). The download creates a `simple_data/` directory with `.pt` files the loaders expect.

3) Quick sanity run (from project root)
- Bash:
  ```bash
  python backend/main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 5
  ```
- Windows (PowerShell/CMD):
  ```bat
  python backend\main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 5
  ```


## What’s Inside (1‑line map of folders and files)
- README.md – This guide with how-tos and commands.
- DOWNLOAD_INSTRUCTIONS.md – How to obtain the preprocessed datasets.
- requirements.txt – Python dependencies (PyTorch, PyG, gdown, etc.).
- simple_data\ – Folder created after dataset download; contains .pt files (not in repo).
- results\ – Output artifacts (logs, CSVs, saved models) created by scripts.
  - results\logs\ – Timestamped logs captured by batch/shell scripts.

Backend (training code and utilities)
- backend\main.py – Single-run CLI entry point (choose model/dataset/epochs, etc.).
- backend\models.py – All model definitions: BaselineGCN, GraphSAGE, GAT, TGAT, TGN, AGNNet.
- backend\models_agn_net_only.py – Deprecated shim re-exporting AGNNet from models.py.
- backend\data_loader.py – Load `.pt` datasets; optional SMOTE; fallback partitioning.
- backend\train.py – Training/eval loops for full-batch and sampled NeighborLoader; schedulers.
- backend\ova_smote.py – One-vs-All with per-class SMOTE; reports averaged accuracy.
- backend\hyperparameter_search.py – Local hyperparameter search with caching of best config.
- backend\simple_sampler.py – Pure-CPU NeighborLoader fallback when PyG extensions are missing.
- backend\count_saved_model_params.py – Count parameters in saved `.pt` checkpoints (optionally verbose).
- backend\verify_mooc_models.py – Small MOOC-only smoke test to verify model forward/training.
- backend\verify_tgn_shape.py – TGN forward-shape check on both subgraph and full-graph paths.

Experiments (ready-to-run scripts)
- experiments\run_all_experiments.bat/.sh – Run HP search (if needed) then train all models on all datasets.
- experiments\run_all_experiments_agn_only.bat/.sh – Run only AGNNet with small sweeps and an ablation.
- experiments\run_ova_experiments.bat/.sh – Run OVA-SMOTE average-accuracy across models/datasets.
- experiments\run_ablation_agnnet.bat/.sh – Reproducible AGNNet ablation suite.
- experiments\run_param_scaling_ova.py – Orchestrates parameter-scaling OVA runs and writes a CSV.
- experiments\run_param_scaling_ova.bat/.sh – Low-level runners invoked by the Python driver.


## Supported models and datasets
- Models: BaselineGCN, GraphSAGE, GAT, TGAT, TGN, AGNNet
- Datasets: OGB-Arxiv, Reddit, TGB-Wiki, MOOC

All CLI arguments for backend\main.py (single-run trainer):
- --model (required; choices: BaselineGCN, GraphSAGE, GAT, TGAT, TGN, AGNNet): Model architecture to use.
- --dataset (required; choices: OGB-Arxiv, Reddit, TGB-Wiki, MOOC): Dataset to train on.
- --epochs (int, required): Number of training epochs.
- --lr (float, default 0.01): Learning rate.
- --hidden-channels (int, default 64): Hidden dimension size.
- --dropout (float, default 0.5): Dropout probability.
- --weight-decay (float, default 5e-4): Weight decay.
- --num-layers (int, default 2): Number of GNN layers.
- --aggr (str, default "mean"): Aggregator for GraphSAGE.
- --heads (int, default 2): Number of attention heads for GAT/TGAT.
- --time-dim (int, default 32): Temporal dimension for TGAT.
- --mem (int, default 100): Memory dimension for TGN.
- --encoder (int, default 64): Encoder dimension for TGN.
- --tau (float, default 0.9): Temperature/threshold tau for AGNNet.
- --k (int, default 8): Neighborhood/Top-k cap for AGNNet.
- --k-anneal (flag): Enable annealing k over epochs for AGNNet.
- --k-min (int, default 2): Minimum k for annealing.
- --k-max (int, default None): Maximum k for annealing (defaults to --k if omitted).
- --ffn-expansion (float, default 2.0): FFN expansion factor in AGN blocks.
- --soft-topk (flag): Use soft attention with top-k cap.
- --edge-threshold (float, default 0.0): Keep edges with attention logits above this threshold before top-k cap.
- --disable-pred-subgraph (flag): Ablate predictive-subgraph selection (use full graph).
- --no-self-loops (flag): Disable forced self-loops in AGNNet subgraph.
- --optimizer (str, default "adamw"; choices: adam, adamw): Optimizer choice.
- --lr-schedule (str, default "cosine"; choices: none, cosine): LR scheduler.
- --warmup-epochs (int, default 500): Warmup steps for cosine schedule (epochs).
- --label-smoothing (float, default 0.0): Label smoothing for CrossEntropyLoss.
- --load-model (path, default None): Path to model checkpoint.
- --config (path, default None): JSON file with hyperparameters (overrides matching CLI values).
- --num-parts (int, default 4): Number of partitions to use on OOM fallback.
- --ova-smote (flag): Run One-vs-All experiments with per-class SMOTE; prints OVA_AVG_ACCURACY.

Other CLI commands and their arguments:
- backend\hyperparameter_search.py
  - --model (required)
  - --dataset (required)
  - --epochs (int, default 20)
  - --save-dir (path, default "saved_models")
- backend\count_saved_model_params.py
  - --dir (path, default "saved_models"): Directory containing saved .pt files
  - --verbose (flag): Print layer-wise parameter counts
- experiments\run_param_scaling_ova.py
  - --output-csv (path, default results/param_scaling_ova_results.csv)
  - --epochs (int, default 5): Epochs per OVA run (per class)
  - --datasets (list, default: OGB-Arxiv Reddit TGB-Wiki MOOC)
  - --models (list, default: BaselineGCN GraphSAGE GAT TGAT AGNNet; TGN skipped inside script)


## How to run any architecture on any dataset from the command line
All commands are from the project root.

Template
- Bash:
  ```bash
  python backend/main.py --model <MODEL> --dataset <DATASET> --epochs <E>
  ```
- Windows (PowerShell/CMD):
  ```bat
  python backend\main.py --model <MODEL> --dataset <DATASET> --epochs <E>
  ```

Examples per architecture and dataset (Bash)
- BaselineGCN
  - `python backend/main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20`
  - `python backend/main.py --model BaselineGCN --dataset Reddit --epochs 20`
  - `python backend/main.py --model BaselineGCN --dataset TGB-Wiki --epochs 20`
  - `python backend/main.py --model BaselineGCN --dataset MOOC --epochs 20`
- GraphSAGE
  - `python backend/main.py --model GraphSAGE --dataset OGB-Arxiv --epochs 20`
  - `python backend/main.py --model GraphSAGE --dataset Reddit --epochs 20`
  - `python backend/main.py --model GraphSAGE --dataset TGB-Wiki --epochs 20`
  - `python backend/main.py --model GraphSAGE --dataset MOOC --epochs 20`
- GAT
  - `python backend/main.py --model GAT --dataset OGB-Arxiv --epochs 20`
  - `python backend/main.py --model GAT --dataset Reddit --epochs 20`
  - `python backend/main.py --model GAT --dataset TGB-Wiki --epochs 20`
  - `python backend/main.py --model GAT --dataset MOOC --epochs 20`
- TGAT
  - `python backend/main.py --model TGAT --dataset OGB-Arxiv --epochs 20`
  - `python backend/main.py --model TGAT --dataset Reddit --epochs 20`
  - `python backend/main.py --model TGAT --dataset TGB-Wiki --epochs 20`
  - `python backend/main.py --model TGAT --dataset MOOC --epochs 20`
- TGN
  - `python backend/main.py --model TGN --dataset OGB-Arxiv --epochs 20`
  - `python backend/main.py --model TGN --dataset Reddit --epochs 20`
  - `python backend/main.py --model TGN --dataset TGB-Wiki --epochs 20`
  - `python backend/main.py --model TGN --dataset MOOC --epochs 20`
- AGNNet
  - `python backend/main.py --model AGNNet --dataset OGB-Arxiv --epochs 20`
  - `python backend/main.py --model AGNNet --dataset Reddit --epochs 20`
  - `python backend/main.py --model AGNNet --dataset TGB-Wiki --epochs 20`
  - `python backend/main.py --model AGNNet --dataset MOOC --epochs 20`

Examples per architecture and dataset (Windows)
- BaselineGCN
  - `python backend\main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20`
  - `python backend\main.py --model BaselineGCN --dataset Reddit --epochs 20`
  - `python backend\main.py --model BaselineGCN --dataset TGB-Wiki --epochs 20`
  - `python backend\main.py --model BaselineGCN --dataset MOOC --epochs 20`
- GraphSAGE
  - `python backend\main.py --model GraphSAGE --dataset OGB-Arxiv --epochs 20`
  - `python backend\main.py --model GraphSAGE --dataset Reddit --epochs 20`
  - `python backend\main.py --model GraphSAGE --dataset TGB-Wiki --epochs 20`
  - `python backend\main.py --model GraphSAGE --dataset MOOC --epochs 20`
- GAT
  - `python backend\main.py --model GAT --dataset OGB-Arxiv --epochs 20`
  - `python backend\main.py --model GAT --dataset Reddit --epochs 20`
  - `python backend\main.py --model GAT --dataset TGB-Wiki --epochs 20`
  - `python backend\main.py --model GAT --dataset MOOC --epochs 20`
- TGAT
  - `python backend\main.py --model TGAT --dataset OGB-Arxiv --epochs 20`
  - `python backend\main.py --model TGAT --dataset Reddit --epochs 20`
  - `python backend\main.py --model TGAT --dataset TGB-Wiki --epochs 20`
  - `python backend\main.py --model TGAT --dataset MOOC --epochs 20`
- TGN
  - `python backend\main.py --model TGN --dataset OGB-Arxiv --epochs 20`
  - `python backend\main.py --model TGN --dataset Reddit --epochs 20`
  - `python backend\main.py --model TGN --dataset TGB-Wiki --epochs 20`
  - `python backend\main.py --model TGN --dataset MOOC --epochs 20`
- AGNNet
  - `python backend\main.py --model AGNNet --dataset OGB-Arxiv --epochs 20`
  - `python backend\main.py --model AGNNet --dataset Reddit --epochs 20`
  - `python backend\main.py --model AGNNet --dataset TGB-Wiki --epochs 20`
  - `python backend\main.py --model AGNNet --dataset MOOC --epochs 20`

Tip: You can add hyperparameters to any command, e.g., `--hidden-channels 128 --num-layers 3 --dropout 0.25`.


## One‑vs‑All (OVA) with SMOTE
The OVA path trains a binary classifier per class (with SMOTE oversampling on the train nodes) and prints the averaged accuracy.

- Bash:
  ```bash
  python backend/main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20 --ova-smote
  ```
- Windows:
  ```bat
  python backend\main.py --model BaselineGCN --dataset OGB-Arxiv --epochs 20 --ova-smote
  ```
Notes:
- For Reddit (sampled training), the averaged validation accuracy is reported; test average may be skipped.
- For TGN or very large graphs, SMOTE is skipped to avoid memory blow-ups.


## Run all predefined experiments
- Bash:
  ```bash
  bash experiments/run_all_experiments.sh
  ```
- Windows (PowerShell/CMD):
  ```bat
  experiments\run_all_experiments.bat
  ```
Behavior:
- Ensures datasets exist; runs hyperparameter search if a model/dataset pair lacks a cached config; then trains.
- Saves logs under `results/logs/` (bash) or similar transcript; Windows batch also records to `results\logs` from its working dir.
- Saves best checkpoints and configs under `results/saved_models/`.


## AGNNet ablation study
- Bash: `bash experiments/run_ablation_agnnet.sh`
- Windows: `experiments\run_ablation_agnnet.bat`
What it runs:
- Baseline AGNNet run, then 5 ablations toggling predictive-subgraph, k-anneal, soft-topk, self-loops, edge-threshold across all datasets.
Override epochs: `export EPOCHS=20` (bash) or `set EPOCHS=20` (Windows).


## Parameter‑scaling OVA sweep (CSV output)
Use the Python driver (it writes the CSV) and let it call the OS-specific runners.

- Quick sanity (one dataset/model):
  - Bash:
    ```bash
    python experiments/run_param_scaling_ova.py --epochs 3 --output-csv results/param_scaling_ova_results.csv \
      --datasets OGB-Arxiv \
      --models BaselineGCN
    ```
  - Windows:
    ```bat
    python experiments\run_param_scaling_ova.py --epochs 3 --output-csv results\param_scaling_ova_results.csv ^
      --datasets OGB-Arxiv ^
      --models BaselineGCN
    ```
- Full sweep (all datasets/models except TGN in scaling logic):
  - Bash:
    ```bash
    python experiments/run_param_scaling_ova.py --epochs 5 --output-csv results/param_scaling_ova_results.csv \
      --datasets OGB-Arxiv Reddit TGB-Wiki MOOC \
      --models BaselineGCN GraphSAGE GAT TGAT AGNNet
    ```
  - Windows:
    ```bat
    python experiments\run_param_scaling_ova.py --epochs 5 --output-csv results\param_scaling_ova_results.csv ^
      --datasets OGB-Arxiv Reddit TGB-Wiki MOOC ^
      --models BaselineGCN GraphSAGE GAT TGAT AGNNet
    ```
Output:
- CSV columns: `dataset, model, param_count, average_ova_accuracy` (validation average).
- Logs written under `results/logs/`.


## OVA across all models/datasets (quick driver)
- Bash: `bash experiments/run_ova_experiments.sh`
- Windows: `experiments\run_ova_experiments.bat`


## Hyperparameter search (ad-hoc)
To run a small local search for a specific model/dataset (caches best config and checkpoint under `results/saved_models/`):

- Bash:
  ```bash
  python backend/hyperparameter_search.py --model GraphSAGE --dataset OGB-Arxiv --epochs 20 --save-dir results/saved_models
  ```
- Windows:
  ```bat
  python backend\hyperparameter_search.py --model GraphSAGE --dataset OGB-Arxiv --epochs 20 --save-dir results\saved_models
  ```


## Outputs and logs
- Checkpoints/configs: `results/saved_models/` (created automatically by the runners).
- Logs: typically under `results/logs/` (shell) or `logs/` (some .bat scripts). Search both if unsure.
- CSVs: placed in `results/` (e.g., `param_scaling_ova_results.csv`).


## Troubleshooting
- Missing PyG extensions: The code will fall back to `backend/simple_sampler.py`, which is slower; consider installing proper wheels if you see very slow sampled runs.
- CUDA memory: For very large graphs or TGN, SMOTE and/or full-batch training may be skipped; use the sampled path or reduce hidden size/layers.
- Dataset path: Ensure `simple_data/` exists at the project root with the expected `.pt` files.


## Contributing / Extending
- Add a new model in `backend/models.py` and wire it in `backend/main.py` CLI choices.
- Add new datasets by extending `backend/data_loader.py` to return `(data, feat_dim, num_classes)`.
- Use `backend/count_saved_model_params.py --dir results/saved_models` to audit parameter counts of saved checkpoints.




## Non-admin CUDA setup (no system CUDA required)
If you need a GPU-enabled setup without installing a system-wide CUDA toolkit or admin rights, follow the step-by-step guide in [CUDA_non_admin_fix.md](CUDA_non_admin_fix.md). It installs PyTorch 2.4.0 with bundled CUDA 12.1 and matching PyG wheels.
