import os
import torch
from torch_geometric.loader import NeighborLoader

import data_loader
import models
import train


def run_ova_smote_experiments(args, device):
    """
    Run One-vs-All experiments with per-class SMOTE. For each class c in {0..K-1}:
      - Create binary labels y_bin = 1 if y==c else 0
      - Apply SMOTE to training nodes
      - Build a 2-class version of the selected model
      - Train using the existing pipeline and record best validation (and test where applicable) accuracy
    Finally, report the average across classes.
    """
    import gc
    from copy import deepcopy  # kept for parity if needed later
    from torch_geometric.data import Data

    # Load base dataset once (without global SMOTE)
    data, feat_dim, num_classes = data_loader.load_dataset(name=args.dataset, root="simple_data")
    is_sampled_dataset = args.dataset == "Reddit"

    val_accs = []
    test_accs = []

    for cls in range(num_classes):
        print(f"\n=== OVA-SMOTE: Class {cls+1}/{num_classes} as positive ===")
        # Build a shallow copy of data to alter labels without touching the original
        d = Data(
            x=data.x.clone(),  # clone to avoid in-place side effects when SMOTE appends
            y=(data.y == cls).long(),
            edge_index=data.edge_index.clone(),
        )
        # Preserve optional attributes
        for opt in ["edge_attr", "train_mask", "val_mask", "test_mask", "n_id", "edge_time"]:
            if hasattr(data, opt):
                setattr(d, opt, getattr(data, opt).clone() if isinstance(getattr(data, opt), torch.Tensor) else getattr(data, opt))
        d.num_nodes = data.num_nodes

        # Apply SMOTE to the binary problem (on training nodes only)
        if args.model.lower() != "tgn" and getattr(d, 'num_nodes', 0) <= 200_000:
            d = data_loader.apply_smote(d)
        else:
            print("Skipping SMOTE for this configuration to avoid memory blow-up.")

        # Send to device if full-batch; NeighborLoader for Reddit will handle .to in batches
        if not is_sampled_dataset:
            d = d.to(device)

        # --- Build a 2-class model for this binary task ---
        model_name = args.model.lower()
        if model_name == "baselinegcn":
            model = models.BaselineGCN(feat_dim, args.hidden_channels, 2, args.dropout)
        elif model_name == "graphsage":
            model = models.GraphSAGE(
                feat_dim,
                args.hidden_channels,
                2,
                num_layers=args.num_layers,
                dropout=args.dropout,
                aggr=args.aggr,
            )
        elif model_name == "gat":
            model = models.GAT(
                feat_dim,
                args.hidden_channels,
                2,
                heads=args.heads,
                dropout=args.dropout,
            )
        elif model_name == "tgat":
            model = models.TGAT(
                feat_dim,
                args.hidden_channels,
                2,
                num_layers=args.num_layers,
                dropout=args.dropout,
                heads=args.heads,
                time_dim=args.time_dim,
            )
        elif model_name == "tgn":
            model = models.TGN(
                d.num_nodes,
                args.mem,
                1,
                2,
                heads=args.heads,
            )
        elif model_name == "agnnet":
            model = models.AGNNet(
                feat_dim,
                args.hidden_channels,
                2,
                tau=args.tau,
                k=args.k,
                num_layers=args.num_layers,
                dropout=args.dropout,
                ffn_expansion=args.ffn_expansion,
                soft_topk=args.soft_topk,
                edge_threshold=args.edge_threshold,
                disable_pred_subgraph=args.disable_pred_subgraph,
                add_self_loops=(not args.no_self_loops),
            )
        else:
            raise ValueError(f"Model '{args.model}' not implemented")
        model = model.to(device)

        # --- Dataloader Setup for this class ---
        train_loader = val_loader = test_loader = None
        if is_sampled_dataset:
            # Similar NeighborLoader setup as the main path (conservative defaults)
            neighbor_sizes = [15] * args.num_layers
            batch_size = 512
            args.accum_steps = 2
            num_workers = 0 if os.name == 'nt' else 4
            pin_memory = torch.cuda.is_available()

            def build_neighbor_loaders(ns, bs):
                tl = NeighborLoader(
                    d,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=d.train_mask,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                vl = NeighborLoader(
                    d,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=d.val_mask,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                te = NeighborLoader(
                    d,
                    num_neighbors=ns,
                    batch_size=bs,
                    input_nodes=d.test_mask,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=(num_workers > 0),
                )
                return tl, vl, te

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
            train_loader = val_loader = test_loader = d

        # --- Train ---
        try:
            best_val_acc, best_test_acc, _ = train.run_training_session(
                model,
                d,
                train_loader,
                val_loader,
                test_loader,
                is_sampled_dataset,
                device,
                args,
            )
        finally:
            # Cleanup per-class to reduce memory usage
            del model, train_loader, val_loader, test_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_accs.append(best_val_acc)
        # test acc is only meaningful for non-sampled flows
        if not is_sampled_dataset:
            test_accs.append(best_test_acc)

    # Report averages
    avg_val = None
    avg_test = None
    if val_accs:
        avg_val = sum(val_accs) / len(val_accs)
        print(f"\nOVA-SMOTE Average Validation Accuracy across {len(val_accs)} classes: {avg_val:.4f}")
    if test_accs:
        avg_test = sum(test_accs) / len(test_accs)
        print(f"OVA-SMOTE Average Test Accuracy across {len(test_accs)} classes: {avg_test:.4f}")
    else:
        if is_sampled_dataset:
            print("Test accuracy is not computed in sampled training path; only validation accuracy is averaged.")
    return avg_val
