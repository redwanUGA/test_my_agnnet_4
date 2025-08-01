import os
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch.serialization

# Mapping from dataset name to filename within the downloaded folder
_DATASET_FILES = {
    'OGB-Arxiv': 'OGB_data.pt',
    'Reddit': 'Reddit_data.pt',
    'TGB-Wiki': 'TGB_wiki_data.pt',
    'MOOC': 'MOOC_data.pt',
}


_DEF_ROOT = 'simple_data'


def _load_pt_dataset(pt_path):
    """Load a PyG Data object from a .pt file."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"{pt_path} not found. Download datasets using DOWNLOAD_INSTRUCTIONS.md"
        )

    # Allowlist PyG classes for torch.load
    torch.serialization.add_safe_globals([
        Data, Batch, DataEdgeAttr, DataTensorAttr, GlobalStorage
    ])

    obj = torch.load(pt_path, weights_only=False)

    # Some preprocessing scripts store the Data object inside a dictionary or
    # tuple. Handle these cases by extracting the first element or common keys.
    if isinstance(obj, (list, tuple)):
        data = obj[0]
        if isinstance(data, dict) and not isinstance(data, Data):
            data = Data(**data)
    elif isinstance(obj, dict) and not isinstance(obj, Data):
        # try common keys used when saving PyG objects
        for key in ["data", "graph", "dataset"]:
            if key in obj:
                obj = obj[key]
                break
        data = obj
        # If the extracted object is still a plain dict, convert it to a
        # torch_geometric.data.Data instance. Some preprocessing scripts save
        # attributes directly in a dictionary which confuses the loader.
        if isinstance(data, dict) and not isinstance(data, Data):
            data = Data(**data)
    else:
        data = obj

    if not hasattr(data, 'train_mask'):
        num_nodes = data.num_nodes
        idx = torch.randperm(num_nodes)
        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[idx[:train_end]] = True
        data.val_mask[idx[train_end:val_end]] = True
        data.test_mask[idx[val_end:]] = True
    else:
        # flatten multi-dimensional masks by logical OR across columns
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            mask = getattr(data, mask_name)
            if mask is not None and mask.dim() > 1:
                setattr(data, mask_name, mask.any(dim=1))

    # ensure label tensor is 1D
    if isinstance(data.y, torch.Tensor) and data.y.dim() > 1 and data.y.size(1) == 1:
        data.y = data.y.view(-1)

    # ADDED THIS LINE TO FIX THE BUG
    if isinstance(data, dict):
        data = Data.from_dict(data)

    feat_dim = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    return data, feat_dim, num_classes


def load_dataset(name: str, root: str = _DEF_ROOT):
    """Load one of the preprocessed datasets from disk."""
    if name not in _DATASET_FILES:
        raise ValueError(f"Unknown dataset: {name}")

    if not os.path.exists(root):
        raise FileNotFoundError(
            f"Dataset folder '{root}' does not exist. Follow DOWNLOAD_INSTRUCTIONS.md"
        )

    print(f"\n--- Loading {name} ---")
    pt_path = os.path.join(root, _DATASET_FILES[name])
    data, feat_dim, num_classes = _load_pt_dataset(pt_path)

    print(f"✅ Dataset '{name}' loaded successfully.")
    print(f"   Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"   Features: {feat_dim}, Classes: {num_classes}")
    print(
        f"   Train nodes: {int(data.train_mask.sum())}, Val nodes: {int(data.val_mask.sum())}, Test nodes: {int(data.test_mask.sum())}")

    return data, feat_dim, num_classes


def apply_smote(data):
    """Apply SMOTE oversampling to the training nodes of a graph dataset."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError(
            "imblearn is required for SMOTE. Install with 'pip install imbalanced-learn'"
        ) from e

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    if train_idx.numel() == 0:
        return data

    X = data.x[train_idx].cpu().numpy()
    y = data.y[train_idx].cpu().numpy()
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)
    num_new = len(X_res) - len(X)
    if num_new <= 0:
        return data

    X_res = torch.tensor(X_res, dtype=data.x.dtype)
    y_res = torch.tensor(y_res, dtype=data.y.dtype)

    new_x = X_res[-num_new:].to(data.x.device)
    new_y = y_res[-num_new:].to(data.y.device)

    data.x = torch.cat([data.x, new_x], dim=0)
    data.y = torch.cat([data.y, new_y], dim=0)

    true_mask = torch.ones(num_new, dtype=data.train_mask.dtype, device=data.train_mask.device)
    false_mask = torch.zeros(num_new, dtype=data.train_mask.dtype, device=data.train_mask.device)

    data.train_mask = torch.cat([data.train_mask, true_mask], dim=0)
    data.val_mask = torch.cat([data.val_mask, false_mask], dim=0)
    data.test_mask = torch.cat([data.test_mask, false_mask], dim=0)

    new_node_ids = torch.arange(data.x.size(0) - num_new, data.x.size(0), device=data.edge_index.device)
    loop_edges = torch.stack([new_node_ids, new_node_ids], dim=0)
    data.edge_index = torch.cat([data.edge_index, loop_edges], dim=1)

    if getattr(data, 'edge_attr', None) is not None:
        edge_attr_dim = data.edge_attr.size(1)
        new_edge_attr = torch.zeros((num_new, edge_attr_dim), dtype=data.edge_attr.dtype, device=data.edge_attr.device)
        data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

    data.num_nodes = data.x.size(0)

    return data
