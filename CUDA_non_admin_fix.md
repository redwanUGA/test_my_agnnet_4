# Non-Admin CUDA Setup (Bundled CUDA 12.1)

This guide shows how to run PyTorch and PyTorch Geometric (PyG) with GPU support without installing system-wide CUDA or requiring administrator rights. It uses the official PyTorch wheels that bundle CUDA 12.1 and the matching PyG wheels.

If you cannot install a system CUDA toolkit (or don't have admin rights), follow the steps below in a fresh virtual environment.

---

## 1) Create a new virtual environment in your home directory

Linux/macOS:
```bash
python3 -m venv ~/.venvs/agn
source ~/.venvs/agn/bin/activate
python -m pip install -U pip
```

Windows (PowerShell):
```powershell
py -3 -m venv $env:USERPROFILE\.venvs\agn
& "$env:USERPROFILE\.venvs\agn\Scripts\Activate.ps1"
python -m pip install -U pip
```

---

## 2) Install PyTorch with bundled CUDA 12.1 (no system CUDA needed)

Inside the activated venv (single line works on all shells):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

Note: These wheels bundle CUDA 12.1. You do not need to install a separate CUDA toolkit. You still need a compatible NVIDIA driver on the machine.

---

## 3) Install PyG (torch-geometric) + optional accelerators matching torch/cu121

Inside the activated venv:
```bash
pip install torch_geometric
pip install -f https://data.pyg.org/whl/torch-2.4.0+cu121.html pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
```

---

## 4) Sanity check

Run the snippet below to verify versions and CUDA availability:

Linux/macOS:
```bash
python - <<'PY'
import torch, torch_geometric as tg
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("pyg:", tg.__version__)
PY
```

Windows (PowerShell):
```powershell
python -c "import torch, torch_geometric as tg; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available()); print('pyg:', tg.__version__)"
```

You should see your torch version (e.g., 2.4.0) and `cuda_available: True` on a machine with a compatible NVIDIA driver.

---

## 5) Run your script inside this venv

Example for this repository:
- Linux/macOS:
  ```bash
  python backend/hyperparameter_search.py
  ```
- Windows (PowerShell/CMD):
  ```powershell
  python backend\hyperparameter_search.py
  ```

You can also use the provided experiment drivers, e.g.:
- Windows:
  ```powershell
  python experiments\run_param_scaling_ova.py --epochs 3 --output-csv results\param_scaling_ova_results.csv --datasets OGB-Arxiv --models BaselineGCN
  ```
- Linux/macOS:
  ```bash
  python experiments/run_param_scaling_ova.py --epochs 3 --output-csv results/param_scaling_ova_results.csv --datasets OGB-Arxiv --models BaselineGCN
  ```

---

## Notes / Troubleshooting
- No admin rights are required to install the above wheels in a virtual environment.
- Ensure the PyG wheel index matches your Torch/CUDA combo (here: torch 2.4.0 + cu121).
- If `torch.cuda.is_available()` is False, verify that the machine has a compatible NVIDIA driver. A separate CUDA toolkit installation is not needed, but a driver is.
- If you encounter package conflicts, create a fresh venv and re-run the steps above.
