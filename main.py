import torch
from torch_geometric.loader import NeighborLoader
import gc

import data_loader
import models
import train


def main():
    """
    Main function to run a series of predefined GNN experiments.
    It iterates through a list of configurations, training each specified model
    on each specified dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initial setup on device: {device}")

    # Dynamically build experiments so all models train on all datasets.
    datasets = ['OGB-Arxiv', 'TGB-Wiki', 'MOOC', 'Reddit']
    model_defaults = {
        'BaselineGCN': {'epochs': 20, 'lr': 0.01, 'hidden_channels': 64,
                        'dropout': 0.5, 'weight_decay': 5e-4, 'num_layers': 2},
        'GraphSAGE':   {'epochs': 20, 'lr': 0.01, 'hidden_channels': 64,
                        'dropout': 0.5, 'weight_decay': 5e-4, 'num_layers': 2},
        'TGAT':        {'epochs': 5,  'lr': 0.01, 'hidden_channels': 64,
                        'dropout': 0.3, 'weight_decay': 5e-4, 'num_layers': 2},
        'TGN':         {'epochs': 5,  'lr': 0.01, 'hidden_channels': 64,
                        'dropout': 0.3, 'weight_decay': 5e-4, 'num_layers': 2},
        'AGNNet':      {'epochs': 5,  'lr': 0.01, 'hidden_channels': 64,
                        'dropout': 0.3, 'weight_decay': 5e-4, 'num_layers': 3}
    }

    experiments = []
    for model_name, defaults in model_defaults.items():
        for dataset in datasets:
            exp = {'model': model_name, 'dataset': dataset}
            exp.update(defaults)
            experiments.append(exp)

    for config in experiments:
        print("\n" + "=" * 60)
        print(f"STARTING EXPERIMENT: Model={config['model']}, Dataset={config['dataset']}")
        print(f"Configuration: {config}")
        print("=" * 60)

        # Simple class to mimic argparse.Namespace for compatibility with the train function
        class ConfigNamespace:
            def __init__(self, adict):
                self.__dict__.update(adict)

        args = ConfigNamespace(config)

        # --- Data Loading ---
        data, feat_dim, num_classes = data_loader.load_dataset(name=args.dataset, root='data')
        data = data.to(device)

        # --- Model Initialization ---
        model_name = args.model.lower()
        if model_name == 'baselinegcn':
            model = models.BaselineGCN(feat_dim, args.hidden_channels, num_classes, args.dropout)
        elif model_name == 'graphsage':
            model = models.GraphSAGE(feat_dim, args.hidden_channels, num_classes, args.num_layers, args.dropout)
        elif model_name == 'tgat':
            model = models.TGAT(feat_dim, args.hidden_channels, num_classes,
                                 num_layers=args.num_layers, dropout=args.dropout)
        elif model_name == 'tgn':
            model = models.TGN(data.num_nodes, args.hidden_channels, 1, num_classes)
        elif model_name == 'agnnet':
            model = models.AGNNet(feat_dim, args.hidden_channels, num_classes, dropout=args.dropout)
        else:
            print(f"Warning: Model '{args.model}' not implemented in this script. Skipping.")
            continue

        model = model.to(device)

        # If multiple GPUs are available, optionally parallelize the model
        if torch.cuda.device_count() > 1:
            try:
                from torch_geometric.nn import DataParallel as PyGDataParallel
                model = PyGDataParallel(model)
                print(f"Model parallelized over {torch.cuda.device_count()} GPUs using PyG DataParallel")
            except Exception:
                # Fall back to vanilla DataParallel if PyG's version is unavailable
                model = torch.nn.DataParallel(model)
                print(f"Model parallelized over {torch.cuda.device_count()} GPUs using torch DataParallel")
        print(f"\nModel Initialized: {args.model}")

        # --- Dataloader Setup ---
        is_sampled = (args.dataset == 'Reddit')
        train_loader, val_loader, test_loader = None, None, None

        if is_sampled:
            # Ensure that required PyG sampling dependencies are available
            try:
                import torch_sparse  # noqa: F401
            except ImportError:
                try:
                    import pyg_lib  # noqa: F401
                except ImportError:
                    msg = (
                        "NeighborLoader requires either 'torch_sparse' or 'pyg_lib'. "
                        "Install one of these packages to run the Reddit experiment."
                    )
                    print(msg)
                    raise ImportError(msg)

            print("Using NeighborSampler for mini-batch training.")
            # Limit the number of neighbors to avoid huge subgraphs on Reddit
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
            getattr(args, 'accum_steps', 1),
        )

        # --- Cleanup ---
        print("Cleaning up memory for next run...")
        del model, data, train_loader, val_loader, test_loader, args
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
