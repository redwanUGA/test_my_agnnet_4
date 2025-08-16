import argparse
import gc
import json
import os
import torch
from torch_geometric.loader import NeighborLoader

import data_loader
import models
import train


def parse_args():
    """Parse command line arguments for a single training run."""
    parser = argparse.ArgumentParser(
        description="Train a GNN model on a selected dataset")
    parser.add_argument(
        "--model",
        required=True,
        choices=["BaselineGCN", "GraphSAGE", "GAT", "TGAT", "TGN", "AGNNet"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["OGB-Arxiv", "Reddit", "TGB-Wiki", "MOOC"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden dimension size",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument(
        "--aggr", type=str, default="mean", help="Aggregator for GraphSAGE")
    parser.add_argument(
        "--heads", type=int, default=2, help="Number of attention heads for GAT/TGAT")
    parser.add_argument(
        "--time-dim", type=int, default=32, help="Temporal dimension for TGAT")
    parser.add_argument(
        "--mem", type=int, default=100, help="Memory dimension for TGN")
    parser.add_argument(
        "--encoder", type=int, default=64, help="Encoder dimension for TGN")
    parser.add_argument(
        "--tau", type=float, default=0.9, help="Threshold tau for AGNNet")
    parser.add_argument(
        "--k", type=int, default=2, help="k-hop value for AGNNet")
    parser.add_argument(
        "--load-model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--config", type=str, default=None, help="JSON file with hyperparameters")
    parser.add_argument(
        "--num-parts", type=int, default=4, help="Number of partitions to use on OOM fallback")
    return parser.parse_args()


def main():
    """Run a single training experiment based on command line arguments."""
    args = parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initial setup on device: {device}")
    print(f"Configuration: {args}")

    # --- Data Loading ---
    data, feat_dim, num_classes = data_loader.load_dataset(name=args.dataset, root="simple_data")
    # Avoid SMOTE for TGN or large graphs to prevent memory blow-up (e.g., TGN memory scales with num_nodes)
    if args.model.lower() != "tgn" and getattr(data, 'num_nodes', 0) <= 200_000:
        data = data_loader.apply_smote(data)
    else:
        print("Skipping SMOTE for this configuration to avoid memory blow-up.")
    # Keep full graph on GPU unless sampled (e.g., Reddit) to avoid OOM in NeighborLoader init
    is_sampled_dataset = args.dataset == "Reddit"
    if not is_sampled_dataset:
        data = data.to(device)

    # --- Model Initialization ---
    model_name = args.model.lower()
    if model_name == "baselinegcn":
        model = models.BaselineGCN(feat_dim, args.hidden_channels, num_classes, args.dropout)
    elif model_name == "graphsage":
        model = models.GraphSAGE(
            feat_dim,
            args.hidden_channels,
            num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            aggr=args.aggr,
        )
    elif model_name == "gat":
        model = models.GAT(
            feat_dim,
            args.hidden_channels,
            num_classes,
            heads=args.heads,
            dropout=args.dropout,
        )
    elif model_name == "tgat":
        model = models.TGAT(
            feat_dim,
            args.hidden_channels,
            num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=args.heads,
            time_dim=args.time_dim,
        )
    elif model_name == "tgn":
        model = models.TGN(
            data.num_nodes,
            args.mem,
            1,
            num_classes,
            heads=args.heads,
        )
    elif model_name == "agnnet":
        model = models.AGNNet(
            feat_dim,
            args.hidden_channels,
            num_classes,
            tau=args.tau,
            k=args.k,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Model '{args.model}' not implemented")

    if args.load_model and os.path.exists(args.load_model):
        state_dict = torch.load(args.load_model, map_location=device, weights_only=True)
        # Safely load checkpoint: skip node-dependent memory tensors if shape mismatches
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state:
                tgt = model_state[k]
                if v.shape == tgt.shape:
                    filtered[k] = v
                else:
                    # For TGN, memory.* tensors depend on num_nodes; skip mismatched ones
                    if k.startswith("memory."):
                        skipped.append((k, tuple(v.shape), tuple(tgt.shape)))
                        continue
                    else:
                        # Keep other mismatched keys out to avoid RuntimeError
                        skipped.append((k, tuple(v.shape), tuple(tgt.shape)))
                        continue
            # Ignore keys not present in current model
        if skipped:
            print("Warning: Skipping mismatched keys during checkpoint load:")
            for k, src_shape, tgt_shape in skipped:
                print(f"  - {k}: checkpoint {src_shape} -> model {tgt_shape}")
        # Load with strict=False to tolerate missing keys
        model.load_state_dict(filtered, strict=False)

    model = model.to(device)

    # Train on a single device without DataParallel to ensure reproducible
    # behavior when running individual experiments from the shell script.
    print(f"\nModel Initialized: {args.model}")

    # --- Dataloader Setup ---
    is_sampled = args.dataset == "Reddit"
    train_loader = val_loader = test_loader = None

    if is_sampled:
        # Ensure that required PyG sampling dependencies are available
        deps_available = False
        try:
            import torch_sparse  # noqa: F401
            deps_available = True
        except ImportError:
            try:
                import pyg_lib  # noqa: F401
                deps_available = True
            except ImportError:
                print(
                    "Optional PyG extensions not available. Proceeding with NeighborLoader; performance may be reduced."
                )

        if deps_available:
            print("Using NeighborSampler for mini-batch training.")
            neighbor_sizes = [15] * args.num_layers
            batch_size = 512
            args.accum_steps = 2  # accumulate gradients to mimic a larger batch
            num_workers = 0 if os.name == 'nt' else 4
            pin_memory = torch.cuda.is_available()

            def build_neighbor_loaders(ns, bs):
                tl = NeighborLoader(
                    data,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=data.train_mask,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                vl = NeighborLoader(
                    data,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=data.val_mask,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                te = NeighborLoader(
                    data,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=data.test_mask,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                return tl, vl, te

            # GPU-aware attempts based on available memory
            free_gb = None
            if torch.cuda.is_available():
                try:
                    free_bytes = torch.cuda.mem_get_info()[0]
                    free_gb = free_bytes / (1024**3)
                except Exception:
                    free_gb = None

            if free_gb is not None and free_gb < 2.0:
                neighbor_sizes = [5] * args.num_layers
                batch_size = 128
            elif free_gb is not None and free_gb < 4.0:
                neighbor_sizes = [10] * args.num_layers
                batch_size = 256
            else:
                neighbor_sizes = [15] * args.num_layers
                batch_size = 512

            attempts = [
                (neighbor_sizes, batch_size),
                ([10] * args.num_layers, 256),
                ([5] * args.num_layers, 128),
                ([3] * args.num_layers, 64),
                ([2] * args.num_layers, 32),
            ]
            last_err = None
            for ns, bs in attempts:
                try:
                    train_loader, val_loader, test_loader = build_neighbor_loaders(ns, bs)
                    last_err = None
                    break
                except torch.cuda.OutOfMemoryError as e:
                    last_err = e
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            if last_err is not None:
                raise last_err
        else:
            print("Proceeding without optional PyG extensions; using NeighborLoader with conservative settings.")
            num_workers = 0 if os.name == 'nt' else 4
            pin_memory = torch.cuda.is_available()
            def build_neighbor_loaders(ns, bs):
                tl = NeighborLoader(
                    data,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=data.train_mask,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                vl = NeighborLoader(
                    data,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=data.val_mask,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                te = NeighborLoader(
                    data,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=data.test_mask,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                return tl, vl, te
            # GPU-aware attempts based on available memory
            free_gb = None
            if torch.cuda.is_available():
                try:
                    free_bytes = torch.cuda.mem_get_info()[0]
                    free_gb = free_bytes / (1024**3)
                except Exception:
                    free_gb = None
            if free_gb is not None and free_gb < 2.0:
                neighbor_sizes = [5] * args.num_layers
                batch_size = 128
            elif free_gb is not None and free_gb < 4.0:
                neighbor_sizes = [10] * args.num_layers
                batch_size = 256
            else:
                neighbor_sizes = [15] * args.num_layers
                batch_size = 512
            attempts = [
                (neighbor_sizes, batch_size),
                ([10] * args.num_layers, 256),
                ([5] * args.num_layers, 128),
                ([3] * args.num_layers, 64),
                ([2] * args.num_layers, 32),
            ]
            last_err = None
            for ns, bs in attempts:
                try:
                    train_loader, val_loader, test_loader = build_neighbor_loaders(ns, bs)
                    last_err = None
                    break
                except torch.cuda.OutOfMemoryError as e:
                    last_err = e
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            if last_err is not None:
                raise last_err

    if not is_sampled:
        train_loader = val_loader = test_loader = data

    # --- Training ---
    try:
        train.run_training_session(
            model,
            data,
            train_loader,
            val_loader,
            test_loader,
            is_sampled,
            device,
            args,
        )
    except torch.cuda.OutOfMemoryError as oom:
        print("\nCUDA OOM detected during training. Triggering partitioned fallback...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        parts = data_loader.partition_graph(data, getattr(args, 'num_parts', 4))
        print(f"Partitioned dataset into {len(parts)} parts. Training each part independently and averaging metrics...")

        part_val_accs = []
        part_test_accs = []

        def build_and_train_on_part(part_data, part_idx):
            # Recreate model for this partition (notably for TGN: num_nodes changes)
            model_name = args.model.lower()
            if model_name == "baselinegcn":
                part_model = models.BaselineGCN(feat_dim, args.hidden_channels, num_classes, args.dropout)
            elif model_name == "graphsage":
                part_model = models.GraphSAGE(
                    feat_dim,
                    args.hidden_channels,
                    num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    aggr=args.aggr,
                )
            elif model_name == "gat":
                part_model = models.GAT(
                    feat_dim,
                    args.hidden_channels,
                    num_classes,
                    heads=args.heads,
                    dropout=args.dropout,
                )
            elif model_name == "tgat":
                part_model = models.TGAT(
                    feat_dim,
                    args.hidden_channels,
                    num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    heads=args.heads,
                    time_dim=args.time_dim,
                )
            elif model_name == "tgn":
                part_model = models.TGN(
                    part_data.num_nodes,
                    args.mem,
                    1,
                    num_classes,
                    heads=args.heads,
                )
            elif model_name == "agnnet":
                part_model = models.AGNNet(
                    feat_dim,
                    args.hidden_channels,
                    num_classes,
                    tau=args.tau,
                    k=args.k,
                    dropout=args.dropout,
                )
            else:
                raise ValueError(f"Model '{args.model}' not implemented")

            # Load checkpoint if provided (skip mismatched memory tensors as above)
            if args.load_model and os.path.exists(args.load_model):
                state_dict = torch.load(args.load_model, map_location=device, weights_only=True)
                model_state = part_model.state_dict()
                filtered = {}
                for k, v in state_dict.items():
                    if k in model_state and v.shape == model_state[k].shape:
                        filtered[k] = v
                part_model.load_state_dict(filtered, strict=False)

            part_model = part_model.to(device)

            # Build loaders for this partition using the same logic
            part_is_sampled = (args.dataset == "Reddit")
            if part_is_sampled:
                neighbor_sizes = [15] * args.num_layers
                batch_size = 512
                args.accum_steps = 2
                # Build NeighborLoaders with GPU-aware settings
                num_workers = 0 if os.name == 'nt' else 4
                pin_memory = torch.cuda.is_available()
                def build_part_neighbor_loaders(ns, bs):
                    tl = NeighborLoader(
                        part_data,
                        num_neighbors=ns,
                        batch_size=bs,
                        input_nodes=part_data.train_mask,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=(num_workers > 0),
                    )
                    vl = NeighborLoader(
                        part_data,
                        num_neighbors=ns,
                        batch_size=bs,
                        input_nodes=part_data.val_mask,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=(num_workers > 0),
                    )
                    te = NeighborLoader(
                        part_data,
                        num_neighbors=ns,
                        batch_size=bs,
                        input_nodes=part_data.test_mask,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=(num_workers > 0),
                    )
                    return tl, vl, te
                # Determine initial sizes based on free GPU memory
                free_gb = None
                if torch.cuda.is_available():
                    try:
                        free_bytes = torch.cuda.mem_get_info()[0]
                        free_gb = free_bytes / (1024**3)
                    except Exception:
                        free_gb = None
                if free_gb is not None and free_gb < 2.0:
                    neighbor_sizes = [5] * args.num_layers
                    batch_size = 128
                elif free_gb is not None and free_gb < 4.0:
                    neighbor_sizes = [10] * args.num_layers
                    batch_size = 256
                else:
                    neighbor_sizes = [15] * args.num_layers
                    batch_size = 512
                attempts = [
                    (neighbor_sizes, batch_size),
                    ([10] * args.num_layers, 256),
                    ([5] * args.num_layers, 128),
                    ([3] * args.num_layers, 64),
                    ([2] * args.num_layers, 32),
                ]
                last_err = None
                for ns, bs in attempts:
                    try:
                        part_train_loader, part_val_loader, part_test_loader = build_part_neighbor_loaders(ns, bs)
                        last_err = None
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        last_err = e
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                if last_err is not None:
                    raise last_err
            else:
                part_train_loader = part_val_loader = part_test_loader = part_data

            print(f"\n--- Training on partition {part_idx+1}/{len(parts)} (nodes={part_data.num_nodes}, edges={part_data.edge_index.size(1)}) ---")
            try:
                best_val, best_test, _ = train.run_training_session(
                    part_model,
                    part_data,
                    part_train_loader,
                    part_val_loader,
                    part_test_loader,
                    part_is_sampled,
                    device,
                    args,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"OOM on partition {part_idx+1}. Consider increasing --num-parts.")
                return None, None
            finally:
                # Cleanup per-partition
                del part_model, part_train_loader, part_val_loader, part_test_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return best_val, best_test

        for i, part in enumerate(parts):
            part = part.to(device)
            val_acc, test_acc = build_and_train_on_part(part, i)
            if val_acc is not None:
                part_val_accs.append(val_acc)
                # Only include test acc for non-sampled datasets where it's computed
                if args.dataset != "Reddit" and test_acc is not None:
                    part_test_accs.append(test_acc)

        if part_val_accs:
            avg_val = sum(part_val_accs) / len(part_val_accs)
            print(f"\nAveraged Val Acc across {len(part_val_accs)} partitions: {avg_val:.4f}")
        if part_test_accs:
            avg_test = sum(part_test_accs) / len(part_test_accs)
            print(f"Averaged Test Acc across {len(part_test_accs)} partitions: {avg_test:.4f}")
        else:
            if args.dataset == "Reddit":
                print("Test accuracy is not computed in sampled training path; only validation accuracy is averaged.")
    
    # --- Cleanup ---
    print("Cleaning up memory...")
    del model, data, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
