import argparse
import json
import os
from pathlib import Path
import itertools
import random
import torch

import data_loader
import models
import train
from torch_geometric.loader import NeighborLoader


def get_search_space():
    return {
        "BaselineGCN": {
            "OGB-Arxiv": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64, 128],
                "dropout": [0.5, 0.6],
            },
            "Reddit": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64],
                "dropout": [0.5, 0.6],
            },
            "TGB-Wiki": {
                "lr": [0.01],
                "hidden_channels": [32],
                "dropout": [0.4, 0.6],
            },
            "MOOC": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64],
                "dropout": [0.6],
            },
        },
        "GAT": {
            "OGB-Arxiv": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64, 128],
                "heads": [2, 4],
                "dropout": [0.5, 0.6],
            },
            "Reddit": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64],
                "heads": [2, 4],
                "dropout": [0.6],
            },
            "TGB-Wiki": {
                "lr": [0.01],
                "hidden_channels": [32],
                "heads": [2, 4],
                "dropout": [0.5, 0.6],
            },
            "MOOC": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64],
                "heads": [2],
                "dropout": [0.6],
            },
        },
        "GraphSAGE": {
            "OGB-Arxiv": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64, 128],
                "aggr": ["mean", "max"],
            },
            "Reddit": {
                "lr": [0.005, 0.01],
                "hidden_channels": [64],
                "aggr": ["mean"],
            },
            "TGB-Wiki": {
                "lr": [0.01],
                "hidden_channels": [32],
                "aggr": ["max"],
            },
            "MOOC": {
                "lr": [0.005],
                "hidden_channels": [64],
                "aggr": ["mean"],
            },
        },
        "TGN": {
            "OGB-Arxiv": {
                "lr": [0.001],
                "mem": [200, 300],
                "encoder": [64, 128],
            },
            "Reddit": {
                "lr": [0.001],
                "mem": [100, 150],
                "encoder": [64],
            },
            "TGB-Wiki": {
                "lr": [0.001],
                "mem": [300],
                "encoder": [128],
            },
            "MOOC": {
                "lr": [0.001],
                "mem": [150],
                "encoder": [64, 128],
            },
        },
        "TGAT": {
            "OGB-Arxiv": {
                "lr": [0.001],
                "heads": [2, 4],
                "time_dim": [32, 64],
            },
            "Reddit": {
                "lr": [0.001],
                "heads": [2],
                "time_dim": [32, 64],
            },
            "TGB-Wiki": {
                "lr": [0.001],
                "heads": [2, 4],
                "time_dim": [64],
            },
            "MOOC": {
                "lr": [0.001],
                "heads": [2],
                "time_dim": [64],
            },
        },
        "AGNNet": {
            "OGB-Arxiv": {
                "lr": [0.01],
                "hidden_channels": [64],
                "tau": [0.85, 0.9, 0.95],
                "k": [1, 2, 3],
            },
            "Reddit": {
                "lr": [0.01],
                "hidden_channels": [64],
                "tau": [0.85, 0.9],
                "k": [1, 2],
            },
            "TGB-Wiki": {
                "lr": [0.01],
                "hidden_channels": [64],
                "tau": [0.9, 0.95],
                "k": [2],
            },
            "MOOC": {
                "lr": [0.01],
                "hidden_channels": [64],
                "tau": [0.85, 0.9],
                "k": [1],
            },
        },
    }


def create_model(model_name, feat_dim, num_classes, args):
    name = model_name.lower()
    if name == "baselinegcn":
        return models.BaselineGCN(feat_dim, args.hidden_channels, num_classes, args.dropout)
    if name == "gat":
        return models.GAT(feat_dim, args.hidden_channels, num_classes, heads=args.heads, dropout=args.dropout)
    if name == "graphsage":
        return models.GraphSAGE(feat_dim, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout, aggr=args.aggr)
    if name == "tgat":
        return models.TGAT(feat_dim, args.hidden_channels, num_classes, heads=args.heads, num_layers=args.num_layers, dropout=args.dropout, time_dim=args.time_dim)
    if name == "tgn":
        # Use the correct number of nodes for TGN memory initialization
        if not hasattr(args, "num_nodes"):
            raise ValueError("args.num_nodes must be set for TGN model creation")
        return models.TGN(args.num_nodes, args.mem, 1, num_classes, heads=args.heads)
    if name == "agnnet":
        return models.AGNNet(feat_dim, args.hidden_channels, num_classes, tau=args.tau, k=args.k, dropout=args.dropout)
    raise ValueError(f"Unknown model {model_name}")


def run_search(model_name, dataset, epochs=2, save_dir="saved_models"):
    space = get_search_space()[model_name][dataset]
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    # AGNNet-specific speed caps: limit trials and optionally epochs
    agn_fast = model_name.lower() == "agnnet"
    if agn_fast:
        try:
            agn_max_trials = int(os.environ.get("AGN_MAX_TRIALS", "3"))
        except Exception:
            agn_max_trials = 3
        if len(combos) > agn_max_trials:
            seed = int(os.environ.get("AGN_HPO_SEED", "0"))
            rnd = random.Random(seed)
            combos = rnd.sample(combos, agn_max_trials)
        try:
            agn_hpo_epochs = int(os.environ.get("AGN_HPO_EPOCHS", "20"))
        except Exception:
            agn_hpo_epochs = 20
        agn_hpo_epochs = max(1, min(agn_hpo_epochs, epochs))
        # Limits for Reddit partitions and NeighborLoader attempts
        try:
            agn_partitions = int(os.environ.get("AGN_PARTITIONS", "4"))
        except Exception:
            agn_partitions = 4
        try:
            agn_neighbor_attempts = int(os.environ.get("AGN_NEIGHBOR_ATTEMPTS", "2"))
        except Exception:
            agn_neighbor_attempts = 2
    else:
        agn_hpo_epochs = epochs
        agn_partitions = None
        agn_neighbor_attempts = None

    best_acc = -1
    best_params = None
    best_state = None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset once per search run to avoid redundant I/O and preprocessing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, feat_dim, num_classes = data_loader.load_dataset(name=dataset, root="simple_data")
    # Mirror main.py: skip SMOTE for TGAT on OGB-Arxiv and for TGN or very large graphs
    if (model_name.lower() != "tgn"
        and not (model_name.lower() == "tgat" and dataset == "OGB-Arxiv")
        and getattr(data, 'num_nodes', 0) <= 200_000):
        data = data_loader.apply_smote(data)
    else:
        print("Skipping SMOTE for this configuration to avoid memory blow-up.")
    # Use sampled NeighborLoader for Reddit and for TGAT on OGB-Arxiv to avoid OOM
    is_sampled_dataset = (dataset == "Reddit") or (dataset == "OGB-Arxiv" and model_name.lower() == "tgat")
    if not is_sampled_dataset:
        data = data.to(device)

    # Precompute Reddit partitions once to avoid recomputation per trial
    precomputed_parts = None
    precomputed_num_parts = None
    if dataset == "Reddit":
        precomputed_num_parts = int(os.environ.get("PARTITIONS", "32"))
        if agn_fast:
            try:
                precomputed_num_parts = max(1, min(precomputed_num_parts, agn_partitions))
            except Exception:
                precomputed_num_parts = max(1, precomputed_num_parts)
        precomputed_parts = data_loader.partition_graph(data, precomputed_num_parts)

    for combo in combos:
        params = dict(zip(keys, combo))
        args = argparse.Namespace(**params)
        # default values
        args.model = model_name
        args.dataset = dataset
        args.epochs = agn_hpo_epochs
        args.weight_decay = 5e-4
        args.num_layers = 2
        defaults = {
            "hidden_channels": 64,
            "dropout": 0.5,
            "heads": 2,
            "time_dim": 32,
            "mem": 100,
            "encoder": 64,
            "aggr": "mean",
            "tau": 0.9,
            "k": 2,
        }
        for field, val in defaults.items():
            if not hasattr(args, field):
                setattr(args, field, val)
        # Enable early stopping during AGNNet HPO to speed up trials
        if agn_fast:
            try:
                args.early_stop_patience = int(os.environ.get("AGN_EARLY_STOP", "2"))
            except Exception:
                args.early_stop_patience = 2

        # Reddit: run hyperparameter search separately for each partition and report/save per-part best
        if dataset == "Reddit":
            # Use precomputed partitions for efficiency
            num_parts = precomputed_num_parts if precomputed_num_parts is not None else int(os.environ.get("PARTITIONS", "32"))
            parts = precomputed_parts if precomputed_parts is not None else data_loader.partition_graph(data, num_parts)
            last_model_path = None
            last_param_path = None
            part_best_vals = []
            for p_idx, part in enumerate(parts, start=1):
                print(f"\n=== Hyperparameter search on Reddit partition {p_idx}/{len(parts)}: nodes={part.num_nodes}, edges={part.edge_index.size(1)} ===")
                best_acc_part = -1.0
                best_params_part = None
                best_state_part = None

                # GPU-aware NeighborLoader settings
                num_workers = 0 if os.name == 'nt' else 4
                pin_memory = torch.cuda.is_available()

                def build_neighbor_loaders_part(ns, bs):
                    tl = NeighborLoader(part, num_neighbors=ns, batch_size=bs, input_nodes=part.train_mask, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
                    vl = NeighborLoader(part, num_neighbors=ns, batch_size=bs, input_nodes=part.val_mask, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
                    te = NeighborLoader(part, num_neighbors=ns, batch_size=bs, input_nodes=part.test_mask, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
                    return tl, vl, te

                def neighbor_attempts():
                    neighbor_sizes = [15] * args.num_layers
                    batch_size = 512
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
                    return [
                        (neighbor_sizes, batch_size),
                        ([10] * args.num_layers, 256),
                        ([5] * args.num_layers, 128),
                        ([3] * args.num_layers, 64),
                        ([2] * args.num_layers, 32),
                    ]

                # Build NeighborLoaders once per partition and reuse across all hyperparameter combos
                attempts = neighbor_attempts()
                if agn_fast and agn_neighbor_attempts is not None:
                    attempts = attempts[:max(1, int(agn_neighbor_attempts))]
                last_err = None
                for ns, bs in attempts:
                    try:
                        train_loader, val_loader, test_loader = build_neighbor_loaders_part(ns, bs)
                        last_err = None
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        last_err = e
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                if last_err is not None:
                    raise last_err

                for combo in combos:
                    params = dict(zip(keys, combo))
                    part_args = argparse.Namespace(**params)
                    # copy defaults and common fields
                    part_args.model = model_name
                    part_args.dataset = dataset
                    part_args.epochs = agn_hpo_epochs
                    part_args.weight_decay = 5e-4
                    part_args.num_layers = getattr(args, 'num_layers', 2)
                    for field, val in defaults.items():
                        if not hasattr(part_args, field):
                            setattr(part_args, field, val)
                    # Provide global num_nodes for TGN
                    setattr(part_args, "num_nodes", data.num_nodes)
                    # Early stopping for AGNNet trials
                    if agn_fast:
                        try:
                            part_args.early_stop_patience = int(os.environ.get("AGN_EARLY_STOP", "2"))
                        except Exception:
                            part_args.early_stop_patience = 2

                    model = create_model(model_name, feat_dim, num_classes, part_args).to(device)

                    # Build loaders with attempts
                    attempts = neighbor_attempts()
                    if agn_fast and agn_neighbor_attempts is not None:
                        attempts = attempts[:max(1, int(agn_neighbor_attempts))]
                    last_err = None
                    for ns, bs in attempts:
                        try:
                            train_loader, val_loader, test_loader = build_neighbor_loaders_part(ns, bs)
                            last_err = None
                            break
                        except torch.cuda.OutOfMemoryError as e:
                            last_err = e
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                    if last_err is not None:
                        raise last_err

                    val_acc, _, model = train.run_training_session(
                        model,
                        part,
                        train_loader,
                        val_loader,
                        test_loader,
                        True,
                        device,
                        part_args,
                    )
                    if val_acc > best_acc_part:
                        best_acc_part = val_acc
                        best_params_part = params
                        try:
                            best_state_part = model.state_dict()
                        except Exception:
                            best_state_part = None

                    # cleanup per combo
                    del model, train_loader, val_loader, test_loader
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Save best per partition
                model_path = Path(save_dir) / f"{model_name}_{dataset}_part{p_idx}.pt"
                param_path = Path(save_dir) / f"{model_name}_{dataset}_part{p_idx}_params.json"
                if best_state_part is not None:
                    torch.save(best_state_part, model_path)
                with open(param_path, "w") as f:
                    json.dump(best_params_part, f)
                print(f"[Partition {p_idx}] Best Val Acc: {best_acc_part:.4f}. Saved model to {model_path}, params to {param_path}")
                part_best_vals.append(best_acc_part)
                last_model_path, last_param_path = model_path, param_path

            if part_best_vals:
                avg_best = float(sum(part_best_vals) / len(part_best_vals))
                print(f"\nAverage of best validation accuracies across {len(part_best_vals)} Reddit partitions: {avg_best:.4f}")
            return last_model_path, last_param_path

        # Determine if we should partition to avoid OOM
        use_partitions = (model_name.lower() == "tgn" and dataset == "OGB-Arxiv")
        num_parts = int(os.environ.get("PARTITIONS", "4")) if use_partitions else 1

        if use_partitions and num_parts > 1:
            # Partition the graph and average metrics over partitions
            parts = data_loader.partition_graph(data, num_parts)
            val_accs = []
            test_accs = []
            last_state = None
            for i, part in enumerate(parts, start=1):
                # Create a fresh model but size TGN memory to the GLOBAL graph
                # TGN uses global node IDs via part.n_id; memory must match the global num_nodes
                part_args = argparse.Namespace(**vars(args))
                setattr(part_args, "num_nodes", data.num_nodes)
                model = create_model(model_name, feat_dim, num_classes, part_args).to(device)

                is_sampled = False  # partitions run full-batch on their subgraph
                train_loader = val_loader = test_loader = part
                print(f"\n--- Training partition {i}/{len(parts)}: nodes={part.num_nodes}, edges={part.edge_index.size(1)} ---")
                val_acc, test_acc, model = train.run_training_session(
                    model,
                    part,
                    train_loader,
                    val_loader,
                    test_loader,
                    is_sampled,
                    device,
                    part_args,
                )
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                # Save last trained model's state before cleanup
                try:
                    last_state = model.state_dict()
                except Exception:
                    last_state = None

                # Proactively clear CUDA cache between partitions
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_val = float(sum(val_accs) / len(val_accs)) if val_accs else -1
            avg_test = float(sum(test_accs) / len(test_accs)) if test_accs else -1
            print(f"\nAveraged over {len(val_accs)} partitions -> Val Acc: {avg_val:.4f}, Test Acc: {avg_test:.4f}")

            # For search, select by averaged validation accuracy; keep last model state as placeholder
            val_acc = avg_val
            best_candidate_state = last_state if len(parts) > 0 else None
        else:
            # Non-partitioned path (original behavior)
            # Provide num_nodes to args for correct TGN initialization
            setattr(args, "num_nodes", data.num_nodes)
            model = create_model(model_name, feat_dim, num_classes, args).to(device)

            is_sampled = is_sampled_dataset
            if is_sampled:
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
                            "Optional PyG extensions not available. Proceeding with NeighborLoader on CPU graph; performance may be reduced."
                        )

                neighbor_sizes = [15] * args.num_layers
                batch_size = 512
                num_workers = 0 if os.name == 'nt' else 4
                pin_memory = torch.cuda.is_available()

                def build_neighbor_loaders(ns, bs, use_neighbor=True):
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
                if agn_fast and agn_neighbor_attempts is not None:
                    attempts = attempts[:max(1, int(agn_neighbor_attempts))]
                
                last_err = None
                for ns, bs in attempts:
                    try:
                        train_loader, val_loader, test_loader = build_neighbor_loaders(ns, bs, use_neighbor=True)
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
                train_loader = val_loader = test_loader = data
            val_acc, _, model = train.run_training_session(
                model,
                data,
                train_loader,
                val_loader,
                test_loader,
                is_sampled,
                device,
                args,
            )
            best_candidate_state = model.state_dict()

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            best_state = best_candidate_state

    model_path = Path(save_dir) / f"{model_name}_{dataset}.pt"
    param_path = Path(save_dir) / f"{model_name}_{dataset}_params.json"
    torch.save(best_state, model_path)
    with open(param_path, "w") as f:
        json.dump(best_params, f)
    print(f"Saved best model to {model_path}")
    print(f"Saved params to {param_path}")
    return model_path, param_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-dir", default="saved_models")
    args = parser.parse_args()
    run_search(args.model, args.dataset, epochs=args.epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
