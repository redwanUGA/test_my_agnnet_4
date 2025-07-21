import random
from typing import List
import torch
from torch_geometric.data import Data

class SimpleNeighborLoader:
    """CPU-based fallback loader that performs naive neighbor sampling.

    This is **much** slower than :class:`torch_geometric.loader.NeighborLoader`
    but avoids the optional `torch_sparse` and `pyg_lib` extensions.
    """

    def __init__(self, data: Data, num_neighbors: List[int], batch_size: int,
                 input_nodes: torch.Tensor, shuffle: bool = True):
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)
        self._adj = self._build_adj_list(data.edge_index, data.num_nodes)

    @staticmethod
    def _build_adj_list(edge_index: torch.Tensor, num_nodes: int):
        row, col = edge_index
        adj = [[] for _ in range(num_nodes)]
        for r, c in zip(row.tolist(), col.tolist()):
            adj[r].append(c)
        return adj

    def __iter__(self):
        order = torch.randperm(len(self.input_nodes)) if self.shuffle else torch.arange(len(self.input_nodes))
        for i in range(0, len(order), self.batch_size):
            batch_nodes = self.input_nodes[order[i:i + self.batch_size]].tolist()
            nodes = set(batch_nodes)
            edges = []
            frontier = batch_nodes
            for num in self.num_neighbors:
                next_frontier = []
                for node in frontier:
                    neighs = self._adj[node]
                    if not neighs:
                        continue
                    sample = random.sample(neighs, min(num, len(neighs)))
                    for dst in sample:
                        edges.append((node, dst))
                        if dst not in nodes:
                            nodes.add(dst)
                            next_frontier.append(dst)
                frontier = next_frontier
            n_id = torch.tensor(list(nodes), dtype=torch.long)
            node_map = {n: idx for idx, n in enumerate(n_id.tolist())}
            edge_index = torch.tensor([[node_map[s], node_map[d]] for s, d in edges], dtype=torch.long).t().contiguous()
            batch = Data(x=self.data.x[n_id], edge_index=edge_index, y=self.data.y[n_id])
            # propagate original masks to subgraph
            for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                mask = getattr(self.data, mask_name)
                if mask is not None:
                    sub_mask = mask[n_id]
                else:
                    sub_mask = torch.zeros_like(n_id, dtype=torch.bool)
                setattr(batch, mask_name, sub_mask)
            batch.n_id = n_id
            yield batch
