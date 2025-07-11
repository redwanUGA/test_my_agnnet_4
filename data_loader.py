import os
import zipfile
import requests
import pandas as pd
import numpy as np
import torch

from torch_geometric.datasets import Reddit, WikiCS
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.data import Data, Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch.serialization


def _download_file(url, path):
    """Helper function to download a file with progress."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"⬇️ Downloading {os.path.basename(path)}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Download complete.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}. Error: {e}")
        raise


def _load_ogb_arxiv(root):
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_undirected

    # ✅ Allowlist necessary PyG globals for torch.load
    torch.serialization.add_safe_globals([
        Data, Batch, DataEdgeAttr, DataTensorAttr, GlobalStorage
    ])

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True

    return data, dataset.num_node_features, dataset.num_classes


def _load_reddit(root):
    """Loads the Reddit dataset."""
    dataset = Reddit(root=os.path.join(root, 'Reddit'))
    return dataset[0], dataset.num_features, dataset.num_classes


def _load_tgb_wiki(root):
    """Loads the TGB-Wiki dataset, using PyG's WikiCS as a proxy."""
    dataset = WikiCS(root=os.path.join(root, 'WikiCS'))
    data = dataset[0]
    # Use the first set of masks as the primary split
    data.train_mask = data.train_mask[:, 0]
    data.val_mask = data.val_mask[:, 0]
    # Define test mask as nodes not in train or val
    data.test_mask = ~(data.train_mask | data.val_mask)
    return data, data.num_node_features, dataset.num_classes


def _load_mooc(root):
    """Loads and processes the MOOC dataset from raw TSV files."""
    data_dir = os.path.join(root, "MOOC", "raw")
    files = {
        "features": "mooc_action_features.tsv",
        "labels": "mooc_action_labels.tsv",
        "actions": "mooc_actions.tsv"
    }

    # Simple check if data exists, assumes if one file is there, all are
    if not os.path.exists(os.path.join(data_dir, files['features'])):
        print("MOOC data not found. Generating a small synthetic version for testing purposes.")
        num_nodes = 100
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        x = torch.randn(num_nodes, 4)
        y = torch.randint(0, 2, (num_nodes,))
        idx = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[idx[:60]] = True
        val_mask[idx[60:80]] = True
        test_mask[idx[80:]] = True

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data, x.size(1), int(y.max().item()) + 1

    features_path = os.path.join(data_dir, files["features"])
    labels_path = os.path.join(data_dir, files["labels"])
    actions_path = os.path.join(data_dir, files["actions"])

    try:
        features_df = pd.read_csv(features_path, sep="\t")
        labels_df = pd.read_csv(labels_path, sep="\t")
        actions_df = pd.read_csv(actions_path, sep="\t")
    except Exception as e:
        print(f"Error reading MOOC TSV files: {e}")
        raise

    # Align nodes and create a contiguous mapping
    all_ids = pd.concat([
        features_df["ACTIONID"],
        labels_df["ACTIONID"],
        actions_df["USERID"],
        actions_df["TARGETID"]
    ]).unique()
    all_ids.sort()

    id_to_idx = {action_id: i for i, action_id in enumerate(all_ids)}
    num_nodes = len(all_ids)

    # Process features
    x = torch.zeros((num_nodes, 4), dtype=torch.float)  # MOOC has 4 features
    feat_mapped_ids = features_df["ACTIONID"].map(id_to_idx).to_numpy()
    feat_values = torch.tensor(features_df.drop("ACTIONID", axis=1).values, dtype=torch.float)
    x[feat_mapped_ids] = feat_values

    # Process labels
    y = torch.full((num_nodes,), -1, dtype=torch.long)  # Use -1 for unlabeled nodes
    label_mapped_ids = labels_df["ACTIONID"].map(id_to_idx).to_numpy()
    label_values = torch.tensor(labels_df["LABEL"].values, dtype=torch.long)
    y[label_mapped_ids] = label_values

    # Process edges
    src = actions_df["USERID"].map(id_to_idx).to_numpy()
    dst = actions_df["TARGETID"].map(id_to_idx).to_numpy()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Create masks for labeled nodes
    labeled_nodes_idx = (y != -1).nonzero(as_tuple=False).view(-1)
    np.random.shuffle(labeled_nodes_idx.numpy())  # Shuffle for random split

    train_end = int(0.6 * len(labeled_nodes_idx))
    val_end = int(0.8 * len(labeled_nodes_idx))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[labeled_nodes_idx[:train_end]] = True
    val_mask[labeled_nodes_idx[train_end:val_end]] = True
    test_mask[labeled_nodes_idx[val_end:]] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    num_classes = len(y[y != -1].unique())

    return data, x.size(1), num_classes


def load_dataset(name: str, root: str = 'data'):
    """
    Main function to load a specified dataset.

    Args:
        name (str): The name of the dataset. One of ['OGB-Arxiv', 'Reddit', 'TGB-Wiki', 'MOOC'].
        root (str, optional): The root directory to store the data. Defaults to 'data'.

    Returns:
        Tuple[Data, int, int]: A tuple containing the PyG Data object, number of node features, and number of classes.
    """
    if name not in ['OGB-Arxiv', 'Reddit', 'TGB-Wiki', 'MOOC']:
        raise ValueError(f"Unknown dataset: {name}")

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    print(f"\n--- Loading {name} ---")

    if name == 'OGB-Arxiv':
        data, feat_dim, num_classes = _load_ogb_arxiv(root)
    elif name == 'Reddit':
        data, feat_dim, num_classes = _load_reddit(root)
    elif name == 'TGB-Wiki':
        data, feat_dim, num_classes = _load_tgb_wiki(root)
    elif name == 'MOOC':
        data, feat_dim, num_classes = _load_mooc(root)

    print(f"✅ Dataset '{name}' loaded successfully.")
    print(f"   Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"   Features: {feat_dim}, Classes: {num_classes}")
    print(
        f"   Train nodes: {int(data.train_mask.sum())}, Val nodes: {int(data.val_mask.sum())}, Test nodes: {int(data.test_mask.sum())}")

    return data, feat_dim, num_classes