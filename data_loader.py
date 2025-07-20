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
    elif isinstance(obj, dict) and not isinstance(obj, Data):
        # try common keys used when saving PyG objects
        for key in ["data", "graph", "dataset"]:
            if key in obj:
                obj = obj[key]
                break
        data = obj
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
