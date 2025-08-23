import argparse
import csv
import os
import sys
import subprocess
import re
from types import SimpleNamespace
from typing import Dict, List, Tuple

# Ensure project root is on sys.path so 'backend' can be imported when running this script directly
_CURRENT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from backend import data_loader
from backend import models


SUPPORTED_MODELS = [
    "BaselineGCN",
    "GraphSAGE",
    "GAT",
    "TGAT",
    # TGN intentionally excluded from param-scaling due to memory-state size scaling with num_nodes
    "AGNNet",
]

SUPPORTED_DATASETS = ["OGB-Arxiv", "Reddit", "TGB-Wiki", "MOOC"]


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str,
                feat_dim: int,
                out_channels: int,
                cfg: Dict) -> torch.nn.Module:
    name = model_name.lower()
    if name == "baselinegcn":
        return models.BaselineGCN(feat_dim, cfg.get("hidden_channels", 64), out_channels, cfg.get("dropout", 0.5))
    elif name == "graphsage":
        return models.GraphSAGE(
            feat_dim,
            cfg.get("hidden_channels", 64),
            out_channels,
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.5),
            aggr=cfg.get("aggr", "mean"),
        )
    elif name == "gat":
        return models.GAT(
            feat_dim,
            cfg.get("hidden_channels", 64),
            out_channels,
            heads=cfg.get("heads", 2),
            dropout=cfg.get("dropout", 0.5),
        )
    elif name == "tgat":
        return models.TGAT(
            feat_dim,
            cfg.get("hidden_channels", 64),
            out_channels,
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.5),
            heads=cfg.get("heads", 2),
            time_dim=cfg.get("time_dim", 32),
        )
    elif name == "agnnet":
        return models.AGNNet(
            feat_dim,
            cfg.get("hidden_channels", 64),
            out_channels,
            tau=cfg.get("tau", 0.9),
            k=cfg.get("k", 8),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.25),
            ffn_expansion=cfg.get("ffn_expansion", 2.0),
            soft_topk=cfg.get("soft_topk", True),
            edge_threshold=cfg.get("edge_threshold", 0.0),
            disable_pred_subgraph=cfg.get("disable_pred_subgraph", False),
            add_self_loops=cfg.get("add_self_loops", True),
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported in param-scaling experiments")


def enumerate_configs(model_name: str) -> List[Dict]:
    # Provide a small grid of configs for parameter scaling
    hs_candidates = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    layer_candidates = [2, 3, 4, 5, 6]
    heads_candidates = [2, 4] if model_name in ("GAT", "TGAT") else [2]
    base = dict(dropout=0.5, aggr="mean", heads=2, time_dim=32, tau=0.9, k=8, ffn_expansion=2.0)

    cfgs: List[Dict] = []
    for hs in hs_candidates:
        for nl in layer_candidates:
            for hd in heads_candidates:
                cfg = dict(base)
                cfg.update(hidden_channels=hs, num_layers=nl, heads=hd)
                cfgs.append(cfg)
    return cfgs


def pick_variants_for_targets(model_name: str,
                              feat_dim: int,
                              out_channels: int,
                              targets: List[int]) -> List[Tuple[Dict, int]]:
    # Build many configs, measure params, pick closest increasing sequence matching targets
    device = torch.device("cpu")
    cfgs = enumerate_configs(model_name)
    entries: List[Tuple[Dict, int]] = []
    for cfg in cfgs:
        try:
            m = build_model(model_name, feat_dim, out_channels, cfg)
            m.to(device)
            n = count_trainable_params(m)
            entries.append((cfg, n))
            del m
        except Exception:
            continue
    # sort by param count ascending
    entries.sort(key=lambda x: x[1])

    picked: List[Tuple[Dict, int]] = []
    used = set()
    prev_n = -1
    for t in targets:
        # find candidate with min |n - t| while ensuring strictly increasing over previous
        best = None
        best_err = None
        for cfg, n in entries:
            if n <= prev_n:
                continue
            err = abs(n - t)
            if best is None or err < best_err:
                best = (cfg, n)
                best_err = err
        if best is None:
            # fallback: take the next bigger one not used
            for cfg, n in entries:
                if n > prev_n:
                    best = (cfg, n)
                    break
        if best is None:
            break
        cfg, n = best
        # avoid duplicates with same param count if possible
        if n in used:
            continue
        used.add(n)
        picked.append((cfg, n))
        prev_n = n
        if len(picked) >= len(targets):
            break
    return picked


def build_args_for_run(model_name: str, dataset: str, epochs: int, cfg: Dict) -> SimpleNamespace:
    # Mirror main.py parser defaults where relevant
    return SimpleNamespace(
        model=model_name,
        dataset=dataset,
        epochs=epochs,
        lr=0.01,
        hidden_channels=cfg.get("hidden_channels", 64),
        dropout=cfg.get("dropout", 0.5),
        weight_decay=5e-4,
        num_layers=cfg.get("num_layers", 2),
        aggr=cfg.get("aggr", "mean"),
        heads=cfg.get("heads", 2),
        time_dim=cfg.get("time_dim", 32),
        mem=100,
        encoder=64,
        tau=cfg.get("tau", 0.9),
        k=cfg.get("k", 8),
        k_anneal=False,
        k_min=2,
        k_max=None,
        ffn_expansion=cfg.get("ffn_expansion", 2.0),
        soft_topk=True,
        edge_threshold=0.0,
        disable_pred_subgraph=False,
        no_self_loops=False,
        optimizer="adamw",
        lr_schedule="cosine",
        warmup_epochs=0,
        label_smoothing=0.0,
        load_model=None,
        config=None,
        num_parts=4,
        ova_smote=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run param-scaling OVA experiments and write CSV results.")
    parser.add_argument("--output-csv", default=os.path.join("results", "param_scaling_ova_results.csv"), help="Output CSV path")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per OVA run (per class)")
    parser.add_argument("--datasets", nargs="*", default=SUPPORTED_DATASETS, help="Datasets to include")
    parser.add_argument("--models", nargs="*", default=SUPPORTED_MODELS, help="Models to include (TGN excluded)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 7 evenly spaced targets from ~10M to ~100M
    targets = [int(x * 1_000_000) for x in [10, 25, 40, 55, 70, 85, 100]]

    # Prepare CSV
    header = ["dataset", "model", "param_count", "average_ova_accuracy"]
    csv_path = args.output_csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for dataset in args.datasets:
        if dataset not in SUPPORTED_DATASETS:
            print(f"[WARN] Skipping unknown dataset: {dataset}")
            continue
        # Load once for feature dim; OVA runner will re-load internally
        try:
            data, feat_dim, num_classes = data_loader.load_dataset(name=dataset, root="simple_data")
        except Exception as e:
            print(f"[ERROR] Failed to load dataset {dataset}: {e}")
            continue

        for model_name in args.models:
            if model_name == "TGN":
                print("[INFO] Skipping TGN for param-scaling due to memory-state dominated parameter count.")
                continue
            print(f"\n=== Selecting variants for {model_name} on {dataset} ===")
            variants = pick_variants_for_targets(model_name, feat_dim, out_channels=2, targets=targets)
            if not variants or len(variants) < 1:
                print(f"[WARN] No viable variants found for {model_name} on {dataset}.")
                continue

            for idx, (cfg, param_count) in enumerate(variants, start=1):
                print(f"\n--- Running OVA {model_name} v{idx} on {dataset} | params≈{param_count:,} ---")
                run_args = build_args_for_run(model_name, dataset, epochs=args.epochs, cfg=cfg)
                # Delegate actual training/eval to platform-specific scripts (.bat/.sh)
                script_base = os.path.join(_CURRENT_DIR, "run_param_scaling_ova")
                is_windows = os.name == "nt"
                script_path = script_base + (".bat" if is_windows else ".sh")
                if not os.path.exists(script_path):
                    print(f"[ERROR] Missing script: {script_path}")
                    avg_val = None
                else:
                    cmd = []
                    if is_windows:
                        cmd = [script_path,
                               model_name,
                               dataset,
                               str(args.epochs),
                               str(run_args.hidden_channels),
                               str(run_args.num_layers),
                               str(run_args.heads)]
                    else:
                        cmd = ["bash", script_path,
                               model_name,
                               dataset,
                               str(args.epochs),
                               str(run_args.hidden_channels),
                               str(run_args.num_layers),
                               str(run_args.heads)]
                    try:
                        completed = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
                        out = (completed.stdout or "") + "\n" + (completed.stderr or "")
                        m = re.search(r"OVA_AVG_ACCURACY=([0-9]*\.?[0-9]+)", out)
                        avg_val = float(m.group(1)) if m else None
                        if completed.returncode != 0:
                            print(f"[ERROR] Script failed (code {completed.returncode}) for {model_name} v{idx} on {dataset}\n--- OUTPUT ---\n{out}\n--- END OUTPUT ---")
                    except Exception as e:
                        print(f"[ERROR] Failed to run script for {model_name} v{idx} on {dataset}: {e}")
                        avg_val = None

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        dataset,
                        model_name,
                        param_count,
                        (f"{avg_val:.6f}" if isinstance(avg_val, float) and avg_val is not None else "")
                    ])

    print(f"\nAll done. Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
