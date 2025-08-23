import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
# Using AMP via torch.amp (avoid deprecated torch.cuda.amp)
# Ensure TGN is imported from models for isinstance checks
from models import AGNNet, TGN
AGN_TYPES = (AGNNet,)
from data_loader import partition_graph
# Fallback CPU sampler if NeighborLoader is unavailable or OOM
from simple_sampler import SimpleNeighborLoader


def train_epoch_full(model, data, optimizer, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Ensure data is on the same device as the model
    data = data.to(next(model.parameters()).device)

    use_amp = bool(getattr(args, '_use_amp', False))
    scaler = getattr(args, '_grad_scaler', None)

    with torch.amp.autocast('cuda', enabled=use_amp):
        if isinstance(model, AGN_TYPES):
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

        loss = F.cross_entropy(logits, labels, label_smoothing=getattr(args, 'label_smoothing', 0.0))

    if scaler is not None and use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss.item()



@torch.no_grad()
def evaluate_full(model, data):
    model.eval()

    # Ensure data is on the same device as the model
    data = data.to(next(model.parameters()).device)

    use_amp = next(model.parameters()).is_cuda
    with torch.amp.autocast('cuda', enabled=use_amp):
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



def train_epoch_sampled(model, loader, optimizer, args):
    model.train()
    total_loss = 0

    use_amp = bool(getattr(args, '_use_amp', False))
    scaler = getattr(args, '_grad_scaler', None)

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(next(model.parameters()).device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            if isinstance(model, AGN_TYPES):
                out = model(
                    batch.x,
                    batch.edge_index,
                    edge_weight=getattr(batch, 'edge_weight', None),
                    batch=getattr(batch, 'batch', None)
                )
            else:
                out = model(batch)

            # Use only seed nodes (first batch.batch_size) for supervision in sampled training
            assert out.size(0) == batch.x.size(0), f"Output size {out.size()} does not match batch.x {batch.x.size()}."
            seed_size = int(getattr(batch, "batch_size", 0))
            if seed_size <= 0 or seed_size > out.size(0):
                # Fallback: try to infer from train_mask if available
                if hasattr(batch, "train_mask") and isinstance(batch.train_mask, torch.Tensor) and batch.train_mask.dtype == torch.bool:
                    seed_size = int(batch.train_mask.sum().item())
                else:
                    seed_size = out.size(0)
            logits = out[:seed_size]
            labels = batch.y[:seed_size].view(-1).long()

            # Filter to valid labels to avoid device-side asserts
            if labels.numel() == 0:
                continue
            valid_mask = (labels >= 0) & (labels < logits.size(1))
            if valid_mask.sum().item() == 0:
                continue

            loss = F.cross_entropy(logits[valid_mask], labels[valid_mask], label_smoothing=getattr(args, 'label_smoothing', 0.0))
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)




@torch.no_grad()
def evaluate_sampled(model, loader):
    model.eval()
    total_correct = [0, 0, 0]
    total = [0, 0, 0]
    use_amp = next(model.parameters()).is_cuda
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            if isinstance(model, AGN_TYPES):
                out = model(
                    batch.x,
                    batch.edge_index,
                    edge_weight=getattr(batch, 'edge_weight', None),
                    batch=getattr(batch, 'batch', None)
                )
            else:
                out = model(batch)

        pred = out.argmax(dim=-1)

        # For sampled batches, only the first `batch_size` nodes are the seeds to evaluate
        assert out.size(0) == batch.x.size(0), f"Output size {out.size()} does not match batch.x {batch.x.size()}."
        seed_size = int(getattr(batch, "batch_size", 0))
        if seed_size <= 0 or seed_size > pred.size(0):
            # Fallback: if masks exist, approximate using any available boolean mask count
            for m in [getattr(batch, 'train_mask', None), getattr(batch, 'val_mask', None), getattr(batch, 'test_mask', None)]:
                if isinstance(m, torch.Tensor) and m.dtype == torch.bool and m.numel() == pred.numel():
                    seed_size = int(m.sum().item())
                    break
            if seed_size <= 0 or seed_size > pred.size(0):
                seed_size = pred.size(0)

        pred_seeds = pred[:seed_size]
        y_seeds = batch.y[:seed_size].view(-1)

        # Accumulate accuracy per-split restricted to seed slice
        masks = [getattr(batch, 'train_mask', None), getattr(batch, 'val_mask', None), getattr(batch, 'test_mask', None)]
        # Filter to valid labels to avoid any invalid comparisons
        y_valid = (y_seeds >= 0) & (y_seeds < out.size(1))
        for i, m in enumerate(masks):
            if isinstance(m, torch.Tensor) and m.dtype == torch.bool:
                seed_mask = m[:seed_size]
                if seed_mask.numel() == seed_size:
                    combined_mask = seed_mask & y_valid
                    if combined_mask.any():
                        total_correct[i] += (pred_seeds[combined_mask] == y_seeds[combined_mask]).sum().item()
                        total[i] += combined_mask.sum().item()

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


def _train_epoch_partitions(model, data, optimizer, num_parts, args):
    parts = partition_graph(data, num_parts)
    total_loss = 0.0
    device = next(model.parameters()).device
    for i, part in enumerate(parts, start=1):
        print(f"[Partition {i}/{len(parts)}] nodes={part.num_nodes}, edges={part.edge_index.size(1)} -> training...")
        # move only the current partition to GPU
        part = part.to(device)
        loss = train_epoch_full(model, part, optimizer, args)
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
    use_amp = next(model.parameters()).is_cuda
    # accumulate correct/total across partitions for train/val/test
    totals = [0, 0, 0]
    corrects = [0, 0, 0]
    for i, part in enumerate(parts, start=1):
        print(f"[Partition {i}/{len(parts)}] evaluating...")
        part = part.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            if isinstance(model, AGN_TYPES):
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
    # Initialize AMP and GradScaler once per session
    try:
        dev_is_cuda = (getattr(device, 'type', None) == 'cuda') or next(model.parameters()).is_cuda
    except Exception:
        dev_is_cuda = torch.cuda.is_available()
    use_amp_cfg = bool(getattr(args, 'use_amp', True))
    use_amp = bool(use_amp_cfg and torch.cuda.is_available() and dev_is_cuda)
    setattr(args, '_use_amp', use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    setattr(args, '_grad_scaler', scaler)

    # Optimizer selection
    if getattr(args, 'optimizer', 'adamw').lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine decay with warmup over epochs (attention-friendly)
    scheduler = None
    if getattr(args, 'lr_schedule', 'cosine') == 'cosine':
        total_epochs = max(1, args.epochs)
        warmup = max(0, int(getattr(args, 'warmup_epochs', 0)))
        def lr_lambda(epoch_idx):
            # epoch_idx starts from 0
            if epoch_idx < warmup and warmup > 0:
                return float(epoch_idx + 1) / float(warmup)
            # cosine from warmup..total_epochs
            progress = (epoch_idx - warmup) / max(1, (total_epochs - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        import math
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val_acc = 0
    best_test_acc = 0

    # Early stopping configuration
    patience = getattr(args, 'early_stop_patience', None)
    epochs_without_improve = 0

    print(f"\n=== Training {args.model} on {args.dataset} for {args.epochs} epochs ===")

    # Partition fallback state
    partition_enabled = False
    chosen_parts = None

    # Sampled fallback state (constructed on demand if OOM persists)
    sampled_fallback_active = False
    sf_train_loader = None
    sf_val_loader = None
    sf_test_loader = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        # allow AGNNet to anneal k over epochs
        try:
            from models_agn_net_only import AGNNet as AGNNetOverride
        except Exception:
            AGNNetOverride = AGNNet
        if isinstance(model, (AGNNet, AGNNetOverride)) and hasattr(model, 'set_epoch'):
            if getattr(args, 'k_anneal', False):
                # enable annealing based on provided bounds
                try:
                    model.enable_k_annealing(k_min=getattr(args, 'k_min', 2), k_max=(args.k_max if args.k_max is not None else args.k), total_epochs=args.epochs)
                except Exception:
                    pass
            model.set_epoch(epoch, total_epochs=args.epochs)

        if is_sampled or sampled_fallback_active:
            tl = train_loader if is_sampled else sf_train_loader
            vl = val_loader if is_sampled else sf_val_loader
            loss = train_epoch_sampled(model, tl, optimizer, args)
            _, val_acc, _ = evaluate_sampled(model, vl)
            test_acc = -1
        else:
            try:
                if not partition_enabled:
                    loss = train_epoch_full(model, data, optimizer, args)
                    train_acc, val_acc, test_acc = evaluate_full(model, data)
                else:
                    print(f"[OOM-Fallback ACTIVE] Training with {chosen_parts} partitions")
                    loss = _train_epoch_partitions(model, data, optimizer, chosen_parts, args)
                    train_acc, val_acc, test_acc = _evaluate_on_partitions(model, data, chosen_parts)
            except torch.cuda.OutOfMemoryError as e:
                print("!!! CUDA OutOfMemoryError detected during full-batch training !!!")
                print(str(e))
                _print_cuda_mem(prefix="[Before fallback] ")
                print("Tip: You can also set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Try increasing number of partitions until it fits
                attempt_parts = [2, 4, 8, 16, 32, 64, 128]
                success = False
                for nparts in attempt_parts:
                    try:
                        print(f"--> Trying partitioned training with {nparts} parts")
                        loss = _train_epoch_partitions(model, data, optimizer, nparts, args)
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
                    print("XX Still OOM after trying partitions. Switching to sampled mini-batch fallback...")
                    # Build sampled loaders (NeighborLoader if available; else SimpleNeighborLoader)
                    # Build on CPU to minimize GPU memory pressure
                    data_cpu = data.cpu()
                    num_layers = int(getattr(args, 'num_layers', 2))
                    built = False
                    sf_train_loader = None
                    sf_val_loader = None
                    sf_test_loader = None
                    # Try torch_geometric NeighborLoader first
                    try:
                        from torch_geometric.loader import NeighborLoader as _NeighborLoader
                        # GPU-aware attempts based on available memory
                        free_gb = None
                        if torch.cuda.is_available():
                            try:
                                free_bytes = torch.cuda.mem_get_info()[0]
                                free_gb = free_bytes / (1024**3)
                            except Exception:
                                free_gb = None
                        if free_gb is not None and free_gb < 2.0:
                            neighbor_sizes = [5] * num_layers
                            batch_size = 128
                        elif free_gb is not None and free_gb < 4.0:
                            neighbor_sizes = [10] * num_layers
                            batch_size = 256
                        else:
                            neighbor_sizes = [15] * num_layers
                            batch_size = 512
                        attempts = [
                            (neighbor_sizes, batch_size),
                            ([10] * num_layers, 256),
                            ([5] * num_layers, 128),
                            ([3] * num_layers, 64),
                            ([2] * num_layers, 32),
                        ]
                        num_workers = 0 if os.name == 'nt' else 4
                        pin_memory = torch.cuda.is_available()
                        last_err = None
                        for ns, bs in attempts:
                            try:
                                sf_train_loader = _NeighborLoader(
                                    data_cpu,
                                    num_neighbors=ns,
                                    batch_size=bs,
                                    input_nodes=data_cpu.train_mask,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    persistent_workers=(num_workers > 0),
                                )
                                sf_val_loader = _NeighborLoader(
                                    data_cpu,
                                    num_neighbors=ns,
                                    batch_size=bs,
                                    input_nodes=data_cpu.val_mask,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    persistent_workers=(num_workers > 0),
                                )
                                sf_test_loader = _NeighborLoader(
                                    data_cpu,
                                    num_neighbors=ns,
                                    batch_size=bs,
                                    input_nodes=data_cpu.test_mask,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    persistent_workers=(num_workers > 0),
                                )
                                built = True
                                last_err = None
                                print(f"--> Sampled fallback using NeighborLoader with ns={ns}, bs={bs}")
                                break
                            except torch.cuda.OutOfMemoryError as e3:
                                last_err = e3
                                print("XX OOM during NeighborLoader construction; trying smaller config...")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                        if not built and last_err is not None:
                            raise last_err
                    except Exception as e_imp:
                        print(f"NeighborLoader unavailable or failed ({e_imp}). Falling back to SimpleNeighborLoader (CPU).")
                        built = False
                    # If NeighborLoader could not be built, use SimpleNeighborLoader
                    if not built:
                        ns = [5] * num_layers
                        bs = 256
                        sf_train_loader = SimpleNeighborLoader(data_cpu, num_neighbors=ns, batch_size=bs, input_nodes=data_cpu.train_mask, shuffle=True)
                        sf_val_loader = SimpleNeighborLoader(data_cpu, num_neighbors=ns, batch_size=bs, input_nodes=data_cpu.val_mask, shuffle=False)
                        sf_test_loader = SimpleNeighborLoader(data_cpu, num_neighbors=ns, batch_size=bs, input_nodes=data_cpu.test_mask, shuffle=False)
                        print(f"--> Sampled fallback using SimpleNeighborLoader with ns={ns}, bs={bs}")
                    sampled_fallback_active = True
                    # Also mark session as sampled for logging/metrics
                    is_sampled = True
                    # Run current epoch using sampled training
                    loss = train_epoch_sampled(model, sf_train_loader, optimizer, args)
                    _, val_acc, _ = evaluate_sampled(model, sf_val_loader)
                    test_acc = -1

        if scheduler is not None:
            scheduler.step()

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            if not is_sampled:
                best_test_acc = test_acc
            epochs_without_improve = 0
        else:
            if patience is not None:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                    break

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
