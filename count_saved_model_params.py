import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import torch


def load_state_dict(file_path: Path) -> Dict[str, torch.Tensor]:
    """
    Loads a PyTorch state_dict from the given .pt file, mapping to CPU.
    Returns an empty dict if loading fails.
    """
    try:
        # weights_only was introduced in newer torch; be compatible if absent
        try:
            state = torch.load(str(file_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(file_path), map_location="cpu")
        if isinstance(state, dict):
            # Some checkpoints wrap state_dict under keys like 'state_dict'
            if all(isinstance(v, torch.Tensor) for v in state.values()):
                return state  # looks like a plain state_dict
            if "state_dict" in state and isinstance(state["state_dict"], dict):
                return state["state_dict"]
        # Fallback: not a dict of tensors
        return {}
    except Exception as e:
        print(f"[WARN] Failed to load '{file_path.name}': {e}")
        return {}


def count_params_from_state_dict(sd: Dict[str, Any]) -> int:
    total = 0
    for k, v in sd.items():
        try:
            if isinstance(v, torch.Tensor):
                total += v.numel()
        except Exception:
            # skip non-tensor or malformed entries
            continue
    return int(total)


def maybe_load_params_json(pt_path: Path) -> Dict[str, Any]:
    params_path = pt_path.with_name(pt_path.stem + "_params.json")
    if params_path.exists():
        try:
            with open(params_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def humanize(n: int) -> str:
    # Show both raw and millions
    if n >= 1_000_000:
        return f"{n} ({n/1_000_000:.3f}M)"
    if n >= 1_000:
        return f"{n} ({n/1_000:.3f}K)"
    return str(n)


def main():
    parser = argparse.ArgumentParser(description="Count parameters of saved model .pt files.")
    parser.add_argument("--dir", default="saved_models", help="Directory containing saved model .pt files")
    parser.add_argument("--verbose", action="store_true", help="Print layer-wise parameter counts")
    args = parser.parse_args()

    save_dir = Path(args.dir)
    if not save_dir.exists() or not save_dir.is_dir():
        print(f"[ERROR] Directory '{save_dir}' does not exist or is not a directory.")
        sys.exit(1)

    pt_files = sorted([p for p in save_dir.iterdir() if p.suffix.lower() == ".pt" and p.is_file()])
    if not pt_files:
        print(f"No .pt files found in '{save_dir}'.")
        sys.exit(0)

    print(f"Scanning {len(pt_files)} model file(s) in: {save_dir}\n")

    grand_total = 0
    for pt in pt_files:
        sd = load_state_dict(pt)
        if not sd:
            print(f"- {pt.name}: could not read state dict (skipped)")
            continue
        total = count_params_from_state_dict(sd)
        grand_total += total

        params_meta = maybe_load_params_json(pt)
        meta_str = ""
        if params_meta:
            # show a concise subset if present
            keys_of_interest = [
                "hidden_channels", "num_layers", "heads", "dropout", "time_dim", "mem", "aggr", "tau", "k"
            ]
            subset = {k: params_meta[k] for k in keys_of_interest if k in params_meta}
            if subset:
                meta_str = f" | params: {subset}"

        print(f"- {pt.name}: {humanize(total)} parameters{meta_str}")

        if args.verbose:
            # Layer-wise counts
            for k, v in sd.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k:50s} {v.shape!s:>20s} -> {v.numel()}" )

    print("\nDone.")


if __name__ == "__main__":
    main()
