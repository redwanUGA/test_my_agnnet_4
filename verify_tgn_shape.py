import torch
from torch_geometric.data import Data
from models import TGN


def run_check():
    torch.manual_seed(0)
    num_nodes = 100
    memory_dim = 16
    out_channels = 3

    model = TGN(num_nodes=num_nodes, memory_dim=memory_dim, edge_feat_dim=1, out_channels=out_channels)

    # Simulate a sampled batch with local indexing and n_id mapping
    batch_size = 20
    n_id = torch.randperm(num_nodes)[:batch_size]
    # Create a simple chain within the local subgraph
    src = torch.arange(0, batch_size - 1, dtype=torch.long)
    dst = torch.arange(1, batch_size, dtype=torch.long)
    edge_index_local = torch.stack([src, dst], dim=0)

    x = torch.randn(batch_size, 5)
    y = torch.randint(0, out_channels, (batch_size,))

    sub_data = Data(x=x, edge_index=edge_index_local, y=y)
    sub_data.n_id = n_id

    out_sub = model(sub_data)
    print('Subgraph output shape:', tuple(out_sub.shape))

    # Now a full-batch style call (no n_id): build a sparse line graph over all nodes
    src_full = torch.arange(0, num_nodes - 1, dtype=torch.long)
    dst_full = torch.arange(1, num_nodes, dtype=torch.long)
    edge_index_full = torch.stack([src_full, dst_full], dim=0)

    full_x = torch.randn(num_nodes, 5)
    full_y = torch.randint(0, out_channels, (num_nodes,))

    full_data = Data(x=full_x, edge_index=edge_index_full, y=full_y)
    out_full = model(full_data)
    print('Full graph output shape:', tuple(out_full.shape))


if __name__ == '__main__':
    run_check()
