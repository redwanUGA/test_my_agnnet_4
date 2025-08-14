import argparse
import json
import os
from pathlib import Path
import itertools
import torch

import data_loader
import models
import train
from torch_geometric.loader import NeighborLoader
from simple_sampler import SimpleNeighborLoader


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
    best_acc = -1
    best_params = None
    best_state = None

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for combo in combos:
        params = dict(zip(keys, combo))
        args = argparse.Namespace(**params)
        # default values
        args.model = model_name
        args.dataset = dataset
        args.epochs = epochs
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data, feat_dim, num_classes = data_loader.load_dataset(name=dataset, root="simple_data")
        data = data_loader.apply_smote(data)
        data = data.to(device)

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
                # Create a fresh model sized to the partition
                part_args = argparse.Namespace(**vars(args))
                setattr(part_args, "num_nodes", part.num_nodes)
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

            is_sampled = dataset == "Reddit"
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
                            "Optional PyG extensions not available. Using SimpleNeighborLoader fallback."
                        )

                neighbor_sizes = [15] * args.num_layers
                batch_size = 512

                if deps_available:
                    train_loader = NeighborLoader(
                        data,
                        num_neighbors=neighbor_sizes,
                        batch_size=batch_size,
                        input_nodes=data.train_mask,
                        shuffle=True,
                        num_workers=4,
                    )
                    val_loader = NeighborLoader(
                        data,
                        num_neighbors=neighbor_sizes,
                        batch_size=batch_size,
                        input_nodes=data.val_mask,
                        num_workers=4,
                    )
                    test_loader = NeighborLoader(
                        data,
                        num_neighbors=neighbor_sizes,
                        batch_size=batch_size,
                        input_nodes=data.test_mask,
                        num_workers=4,
                    )
                else:
                    train_loader = SimpleNeighborLoader(
                        data,
                        num_neighbors=neighbor_sizes,
                        batch_size=batch_size,
                        input_nodes=data.train_mask,
                    )
                    val_loader = SimpleNeighborLoader(
                        data,
                        num_neighbors=neighbor_sizes,
                        batch_size=batch_size,
                        input_nodes=data.val_mask,
                        shuffle=False,
                    )
                    test_loader = SimpleNeighborLoader(
                        data,
                        num_neighbors=neighbor_sizes,
                        batch_size=batch_size,
                        input_nodes=data.test_mask,
                        shuffle=False,
                    )
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save-dir", default="saved_models")
    args = parser.parse_args()
    run_search(args.model, args.dataset, epochs=args.epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
