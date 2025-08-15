import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
# Ensure TGN is imported from models for isinstance checks
from models import AGNNet, TGN
from data_loader import partition_graph


def train_epoch_full(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    # Ensure data is on the same device as the model
    data = data.to(next(model.parameters()).device)

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

    # Ensure data is on the same device as the model
    data = data.to(next(model.parameters()).device)

    if isinstance(model, AGNNet):
        out = model(data.x, data.edge_index, edge_weight=None)
    else:
        out = model(data)

    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = (pred[mask] == data.y[mask].view(-1)).sum()
        total = mask.sum().item()
        acc = (correct.item() / total) if total > 0 else 0
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

        # For sampled batches, model outputs correspond to batch.x ordering
        # So we select logits using the batch-local train_mask
        assert out.size(0) == batch.x.size(0), f"Output size {out.size()} does not match batch.x {batch.x.size()}."
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

        # For sampled batches, predictions are aligned with batch.x order
        assert out.size(0) == batch.x.size(0), f"Output size {out.size()} does not match batch.x {batch.x.size()}."
        pred_for_batch = pred

        for i, mask in enumerate([batch.train_mask, batch.val_mask, batch.test_mask]):
            total_correct[i] += (pred_for_batch[mask] == batch.y[mask].view(-1)).sum().item()
            total[i] += mask.sum().item()

    accs = [c / t if t > 0 else 0 for c, t in zip(total_correct, total)]
    return accs



def _print_cuda_mem(prefix: str = ""):
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            print(f"{prefix}GPU Mem: free={free_gb:.2f} GiB / total={total_gb:.2f} GiB")
        except Exception:
            pass


def _train_epoch_partitions(model, data, optimizer, num_parts):
    parts = partition_graph(data, num_parts)
    total_loss = 0.0
    device = next(model.parameters()).device
    for i, part in enumerate(parts, start=1):
        print(f"[Partition {i}/{len(parts)}] nodes={part.num_nodes}, edges={part.edge_index.size(1)} -> training...")
        # move only the current partition to GPU
        part = part.to(device)
        loss = train_epoch_full(model, part, optimizer)
        total_loss += loss
        # free partition tensors explicitly
        del part
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total_loss / max(len(parts), 1)


@torch.no_grad()
def _evaluate_on_partitions(model, data, num_parts):
    model.eval()
    parts = partition_graph(data, num_parts)
    device = next(model.parameters()).device
    # accumulate correct/total across partitions for train/val/test
    totals = [0, 0, 0]
    corrects = [0, 0, 0]
    for i, part in enumerate(parts, start=1):
        print(f"[Partition {i}/{len(parts)}] evaluating...")
        part = part.to(device)
        if isinstance(model, AGNNet):
            out = model(part.x, part.edge_index, edge_weight=None)
        else:
            out = model(part)
        pred = out.argmax(dim=1)
        masks = [part.train_mask, part.val_mask, part.test_mask]
        ys = part.y
        for idx, m in enumerate(masks):
            if m is None:
                continue
            total = m.sum().item()
            if total == 0:
                continue
            corr = (pred[m] == ys[m].view(-1)).sum().item()
            totals[idx] += total
            corrects[idx] += corr
        # free tensors before next partition
        del part, out, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accs = [ (corrects[i] / totals[i]) if totals[i] > 0 else 0 for i in range(3) ]
    return accs  # [train_acc, val_acc, test_acc]


def run_training_session(
        model, data, train_loader, val_loader, test_loader,
        is_sampled, device, args
):
    """
    Orchestrates the full training and evaluation process with CUDA OOM fallback.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0
    best_test_acc = 0

    print(f"\n=== Training {args.model} on {args.dataset} for {args.epochs} epochs ===")

    # Partition fallback state
    partition_enabled = False
    chosen_parts = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        if is_sampled:
            loss = train_epoch_sampled(model, train_loader, optimizer)
            _, val_acc, _ = evaluate_sampled(model, val_loader)
            test_acc = -1
        else:
            try:
                if not partition_enabled:
                    loss = train_epoch_full(model, data, optimizer)
                    train_acc, val_acc, test_acc = evaluate_full(model, data)
                else:
                    print(f"[OOM-Fallback ACTIVE] Training with {chosen_parts} partitions")
                    loss = _train_epoch_partitions(model, data, optimizer, chosen_parts)
                    train_acc, val_acc, test_acc = _evaluate_on_partitions(model, data, chosen_parts)
            except torch.cuda.OutOfMemoryError as e:
                print("!!! CUDA OutOfMemoryError detected during full-batch training !!!")
                print(str(e))
                _print_cuda_mem(prefix="[Before fallback] ")
                print("Tip: You can also set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Try increasing number of partitions until it fits
                attempt_parts = [2, 4, 8, 16, 32]
                success = False
                for nparts in attempt_parts:
                    try:
                        print(f"--> Trying partitioned training with {nparts} parts")
                        loss = _train_epoch_partitions(model, data, optimizer, nparts)
                        train_acc, val_acc, test_acc = _evaluate_on_partitions(model, data, nparts)
                        chosen_parts = nparts
                        partition_enabled = True
                        success = True
                        print(f"--> Fallback succeeded with {nparts} parts. Will use this setting for remaining epochs.")
                        break
                    except torch.cuda.OutOfMemoryError as e2:
                        print(f"XX Still OOM with {nparts} parts. Increasing partitions...")
                        print(str(e2))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                if not success:
                    raise RuntimeError("Failed to recover from CUDA OOM even after partitioning")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not is_sampled:
                best_test_acc = test_acc

        test_acc_str = f"{test_acc:.4f}" if test_acc != -1 else "N/A"
        part_info = f" | Partitions: {chosen_parts}" if partition_enabled else ""
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc_str}{part_info}")

    print("\n=== Training Complete ===")
    if is_sampled:
        print(f"-> Best Val Acc for {args.dataset}: {best_val_acc:.4f}")
        print("   (To get final test accuracy, load best model and run on a test loader)")
    else:
        print(f"-> Best Val Acc for {args.dataset}: {best_val_acc:.4f}, Test Acc @ Best Val: {best_test_acc:.4f}")
        if partition_enabled:
            print(f"   OOM-fallback was active with {chosen_parts} partitions.")

    return best_val_acc, best_test_acc, model
