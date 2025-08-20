import torch
import data_loader
import models
import train


def run_epochs(model, data, num_epochs=2):
    """Train and evaluate the model for a small number of epochs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, num_epochs + 1):
        loss = train.train_epoch_full(model, data, optimizer)
        train_acc, val_acc, test_acc = train.evaluate_full(model, data)
        print(
            f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
            f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}"
        )


def main():
    device = torch.device('cpu')
    data, feat_dim, num_classes = data_loader.load_dataset('MOOC', root='simple_data')
    data = data.to(device)

    models_to_test = {
        'BaselineGCN': lambda: models.BaselineGCN(feat_dim, 16, num_classes),
        'GraphSAGE': lambda: models.GraphSAGE(feat_dim, 16, num_classes),
        'TGN': lambda: models.TGN(data.num_nodes, 16, 1, num_classes),
        'TGAT': lambda: models.TGAT(feat_dim, 16, num_classes),
        'AGNNet': lambda: models.AGNNet(feat_dim, 16, num_classes),
    }

    for name, builder in models_to_test.items():
        print(f"\nTesting {name}...")
        model = builder().to(device)
        run_epochs(model, data)


if __name__ == "__main__":
    main()
