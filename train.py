import torch
import torch.nn.functional as F
from tqdm import tqdm
# Ensure TGN is imported from models for isinstance checks
from models import AGNNet, TGN


def train_epoch_full(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    if isinstance(model, AGNNet):
        out = model(data.x, data.edge_index, edge_weight=None)
    else:
        out = model(data)

    labels = data.y[data.train_mask].view(-1)
    logits = out[data.train_mask]

    # 🔍 Check for invalid label indices
    if labels.max() >= logits.size(1) or labels.min() < 0:
        print(f"❌ Invalid label values detected. Label range: [{labels.min()} - {labels.max()}], Expected: [0 - {logits.size(1) - 1}]")
        print(f"Logits shape: {logits.shape}")
        raise ValueError("CrossEntropy target contains invalid class index.")

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()



@torch.no_grad()
def evaluate_full(model, data):
    model.eval()

    if isinstance(model, AGNNet):
        out = model(data.x, data.edge_index, edge_weight=None)
    else:
        out = model(data)

    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask].view(-1)).sum()
        acc = correct.item() / mask.sum().item()
        accs.append(acc)
    return accs  # returns [train_acc, val_acc, test_acc]



def train_epoch_sampled(model, loader, optimizer):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(next(model.parameters()).device)

        if isinstance(model, AGNNet):
            out = model(
                batch.x,
                batch.edge_index,
                edge_weight=getattr(batch, 'edge_weight', None),
                batch=getattr(batch, 'batch', None)
            )
        else:
            out = model(batch)

        labels = batch.y[batch.train_mask].view(-1)

        # Conditional logits indexing based on model type
        if isinstance(model, (AGNNet, TGN)):
            # These models output predictions for all nodes in the graph
            # We need to use batch.n_id to select predictions relevant to the current batch
            logits = out[batch.n_id][batch.train_mask]
        else:
            # Models like BaselineGCN, GraphSAGE, TGAT output predictions
            # only for the nodes in the current batch
            logits = out[batch.train_mask]


        # 🔍 Check for invalid label indices
        if labels.max() >= logits.size(1) or labels.min() < 0:
            print(f"❌ Invalid label values detected. Label range: [{labels.min()} - {labels.max()}], Expected: [0 - {logits.size(1) - 1}]")
            print(f"Logits shape: {logits.shape}")
            raise ValueError("CrossEntropy target contains invalid class index.")

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)




@torch.no_grad()
def evaluate_sampled(model, loader):
    model.eval()
    total_correct = [0, 0, 0]
    total = [0, 0, 0]
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)

        if isinstance(model, AGNNet):
            out = model(
                batch.x,
                batch.edge_index,
                edge_weight=getattr(batch, 'edge_weight', None),
                batch=getattr(batch, 'batch', None)
            )
        else:
            out = model(batch)

        pred = out.argmax(dim=-1)

        # Adjust pred indexing based on model type for evaluation
        if isinstance(model, (AGNNet, TGN)):
            pred_for_batch = pred[batch.n_id]
        else:
            pred_for_batch = pred

        for i, mask in enumerate([batch.train_mask, batch.val_mask, batch.test_mask]):
            total_correct[i] += (pred_for_batch[mask] == batch.y[mask].view(-1)).sum().item()
            total[i] += mask.sum().item()

    accs = [c / t if t > 0 else 0 for c, t in zip(total_correct, total)]
    return accs



def run_training_session(
        model, data, train_loader, val_loader, test_loader,
        is_sampled, device, args
):
    """
    Orchestrates the full training and evaluation process.

    Args:
        model (torch.nn.Module): The GNN model to train.
        data (Data): The full dataset graph (used for full-batch).
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        test_loader (DataLoader): Loader for testing data (for full-batch, this is `data`).
        is_sampled (bool): Flag indicating if mini-batching is used.
        device (torch.device): The device to run on.
        args (argparse.Namespace): Command-line arguments.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0
    best_test_acc = 0

    print(f"\n--- Training {args.model} on {args.dataset} for {args.epochs} epochs ---")

    for epoch in range(1, args.epochs + 1):
        if is_sampled:
            # Removed 'accum_steps' from the call, as train_epoch_sampled doesn't take it
            loss = train_epoch_sampled(model, train_loader, optimizer)
            # Unpack the list returned by evaluate_sampled to get val_acc
            _, val_acc, _ = evaluate_sampled(model, val_loader)
            test_acc = -1  # Skip test eval for now
        else:
            loss = train_epoch_full(model, data, optimizer)
            train_acc, val_acc, test_acc = evaluate_full(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not is_sampled:
                best_test_acc = test_acc
            # Optionally: torch.save(model.state_dict(), f'{args.model}_{args.dataset}_best.pt')

        test_acc_str = f"{test_acc:.4f}" if test_acc != -1 else "N/A"
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc_str}")

    print("\n--- Training Complete ---")
    if is_sampled:
        print(f"-> Best Val Acc for {args.dataset}: {best_val_acc:.4f}")
        print("   (To get final test accuracy, load best model and run on a test loader)")
    else:
        print(f"-> Best Val Acc for {args.dataset}: {best_val_acc:.4f}, Test Acc @ Best Val: {best_test_acc:.4f}")

    return best_val_acc, best_test_acc, model
