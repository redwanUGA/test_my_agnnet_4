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

    # A list of dictionaries, where each dictionary defines an experiment.
    # Models that work well with the unified training script are included.
    experiments = [
        # --- BaselineGCN Experiments ---
        {'model': 'BaselineGCN', 'dataset': 'OGB-Arxiv', 'epochs': 20, 'lr': 0.01, 'hidden_channels': 64,
         'dropout': 0.5, 'weight_decay': 5e-4, 'num_layers': 2},
        {'model': 'BaselineGCN', 'dataset': 'TGB-Wiki', 'epochs': 20, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.5,
         'weight_decay': 5e-4, 'num_layers': 2},
        {'model': 'BaselineGCN', 'dataset': 'MOOC', 'epochs': 20, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.5,
         'weight_decay': 5e-4, 'num_layers': 2},
        # {'model': 'BaselineGCN', 'dataset': 'Reddit', 'epochs': 10, 'lr': 0.01, 'hidden_channels': 128, 'dropout': 0.5,
        # 'weight_decay': 5e-4, 'num_layers': 2},

        # --- GraphSAGE Experiments ---
        {'model': 'GraphSAGE', 'dataset': 'OGB-Arxiv', 'epochs': 20, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.5,
         'weight_decay': 5e-4, 'num_layers': 2},
        {'model': 'GraphSAGE', 'dataset': 'TGB-Wiki', 'epochs': 20, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.5,
         'weight_decay': 5e-4, 'num_layers': 2},
        {'model': 'GraphSAGE', 'dataset': 'MOOC', 'epochs': 20, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.5,
         'weight_decay': 5e-4, 'num_layers': 2},
        # {'model': 'GraphSAGE', 'dataset': 'Reddit', 'epochs': 10, 'lr': 0.01, 'hidden_channels': 128, 'dropout': 0.5,
        # 'weight_decay': 5e-4, 'num_layers': 2},

        # --- AGNNet Experiments ---
        {'model': 'AGNNet', 'dataset': 'OGB-Arxiv', 'epochs': 5, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.3,
         'weight_decay': 5e-4, 'num_layers': 3},
        {'model': 'AGNNet', 'dataset': 'TGB-Wiki', 'epochs': 5, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.3,
         'weight_decay': 5e-4, 'num_layers': 3},
        {'model': 'AGNNet', 'dataset': 'MOOC', 'epochs': 5, 'lr': 0.01, 'hidden_channels': 64, 'dropout': 0.3,
         'weight_decay': 5e-4, 'num_layers': 3} #,
        # {'model': 'AGNNet', 'dataset': 'Reddit', 'epochs': 10, 'lr': 0.01, 'hidden_channels': 128, 'dropout': 0.3,
        # 'weight_decay': 5e-4, 'num_layers': 3},
    ]

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
        elif model_name == 'agnnet':
            model = models.AGNNet(feat_dim, args.hidden_channels, num_classes, dropout=args.dropout)
        else:
            print(f"Warning: Model '{args.model}' not implemented in this script. Skipping.")
            continue

        model = model.to(device)
        print(f"\nModel Initialized: {args.model}")

        # --- Dataloader Setup ---
        is_sampled = (args.dataset == 'Reddit')
        train_loader, val_loader, test_loader = None, None, None

        if is_sampled:
            print("Using NeighborSampler for mini-batch training.")
            train_loader = NeighborLoader(data, num_neighbors=[-1] * args.num_layers, batch_size=1024,
                                          input_nodes=data.train_mask, shuffle=True, num_workers=4)
            val_loader = NeighborLoader(data, num_neighbors=[-1] * args.num_layers, batch_size=1024,
                                        input_nodes=data.val_mask, num_workers=4)
            test_loader = NeighborLoader(data, num_neighbors=[-1] * args.num_layers, batch_size=1024,
                                         input_nodes=data.test_mask, num_workers=4)
        else:
            train_loader = val_loader = test_loader = data

        # --- Training ---
        train.run_training_session(
            model, data, train_loader, val_loader, test_loader,
            is_sampled, device, args
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