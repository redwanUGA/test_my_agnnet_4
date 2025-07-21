import dgl
import torch
from torch_geometric.data import Data

def pyg_to_dgl(data: Data) -> dgl.DGLGraph:
    """Convert a PyG Data object to a DGLGraph."""
    if not isinstance(data, Data):
        raise TypeError("Input must be a PyG Data object")
    g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
    if hasattr(data, 'x'):
        g.ndata['feat'] = data.x
    if hasattr(data, 'y'):
        g.ndata['label'] = data.y
    for attr in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, attr):
            g.ndata[attr] = getattr(data, attr)
    return g
