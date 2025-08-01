import argparse
import gc
import torch
from torch_geometric.loader import NeighborLoader
from simple_sampler import SimpleNeighborLoader

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
        choices=["BaselineGCN", "GraphSAGE", "TGAT", "TGN", "AGNNet"],
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
    return parser.parse_args()


def main():
    """Run a single training experiment based on command line arguments."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initial setup on device: {device}")
    print(f"Configuration: {args}")

    # --- Data Loading ---
    data, feat_dim, num_classes = data_loader.load_dataset(name=args.dataset, root="simple_data")
    data = data_loader.apply_smote(data)
    data = data.to(device)

    # --- Model Initialization ---
    model_name = args.model.lower()
    if model_name == "baselinegcn":
        model = models.BaselineGCN(feat_dim, args.hidden_channels, num_classes, args.dropout)
    elif model_name == "graphsage":
        model = models.GraphSAGE(feat_dim, args.hidden_channels, num_classes, args.num_layers, args.dropout)
    elif model_name == "tgat":
        model = models.TGAT(
            feat_dim,
            args.hidden_channels,
            num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif model_name == "tgn":
        model = models.TGN(data.num_nodes, args.hidden_channels, 1, num_classes)
    elif model_name == "agnnet":
        model = models.AGNNet(feat_dim, args.hidden_channels, num_classes, dropout=args.dropout)
    else:
        raise ValueError(f"Model '{args.model}' not implemented")

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
                    "Optional PyG extensions not available. "
                    "Using slower SimpleNeighborLoader fallback."
                )

        if deps_available:
            print("Using NeighborSampler for mini-batch training.")
            neighbor_sizes = [15] * args.num_layers
            batch_size = 512
            args.accum_steps = 2  # accumulate gradients to mimic a larger batch

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
            neighbor_sizes = [15] * args.num_layers
            batch_size = 512
            args.accum_steps = 2
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

    if not is_sampled:
        train_loader = val_loader = test_loader = data

    # --- Training ---
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

    # --- Cleanup ---
    print("Cleaning up memory...")
    del model, data, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
