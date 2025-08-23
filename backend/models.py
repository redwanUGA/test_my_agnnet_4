import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    TransformerConv,
    MessagePassing,
)
from torch_geometric.utils import k_hop_subgraph


# === BaselineGCN ===
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# === GraphSAGE ===
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, aggr="mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
        self.dropout = dropout
        self.aggr = aggr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# === GAT ===
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2,
                 dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             dropout=dropout)
        self.dropout = dropout
        self.heads = heads

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# === TGN Components ===
class TGNMemory(nn.Module):
    def __init__(self, num_nodes, memory_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        # Memory state should not require gradients; it is updated via GRUCell and treated as state
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        self.update_fn = nn.GRUCell(memory_dim, memory_dim)

    def forward(self, node_ids):
        return self.memory[node_ids]

    def update(self, node_ids, messages, timestamps):
        prev_mem = self.memory[node_ids]
        updated_mem = self.update_fn(messages, prev_mem)
        # Ensure dtype consistency (AMP autocast may produce Half tensors)
        if updated_mem.dtype != self.memory.dtype:
            updated_mem = updated_mem.to(dtype=self.memory.dtype)
        self.memory.data[node_ids] = updated_mem.data
        self.last_update.data[node_ids] = timestamps.float()


class MessageModule(nn.Module):
    def __init__(self, memory_dim, edge_feat_dim):
        super().__init__()
        total_input = 2 * memory_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_input, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim)
        )

    def forward(self, source_memory, target_memory, edge_features):
        msg_input = torch.cat([source_memory, target_memory, edge_features], dim=-1)
        return self.mlp(msg_input)


# === TGN Model ===
class TGN(nn.Module):
    def __init__(self, num_nodes, memory_dim, edge_feat_dim, out_channels, heads=2):
        super().__init__()
        self.memory = TGNMemory(num_nodes, memory_dim)
        self.message_module = MessageModule(memory_dim, edge_feat_dim)
        self.conv = TransformerConv(memory_dim, memory_dim, heads=heads, dropout=0.1)
        self.classifier = nn.Linear(memory_dim * heads, out_channels)

    def forward(self, data):
        # Edge indices in sampled batches are local to the subgraph. Map to global IDs if available.
        src, dst = data.edge_index
        n_id = getattr(data, 'n_id', None)
        if n_id is not None:
            src_global = n_id[src]
            dst_global = n_id[dst]
            node_ids_for_conv = n_id  # local node order corresponds to batch.x
        else:
            # Full-batch or plain subgraph without explicit n_id: indices are already local to data
            src_global, dst_global = src, dst
            # Use the current data.num_nodes to restrict computation to this (sub)graph
            node_ids_for_conv = torch.arange(data.num_nodes, device=src.device)

        # Safety: ensure indices are within memory range before any CUDA gather ops
        mem_N = int(self.memory.memory.size(0))
        if src_global.numel() > 0:
            max_idx = int(torch.max(src_global.max(), dst_global.max()).item())
            min_idx = int(torch.min(src_global.min(), dst_global.min()).item())
            if max_idx >= mem_N or min_idx < 0:
                raise RuntimeError(
                    f"TGN memory index out of range: min={min_idx}, max={max_idx}, mem_N={mem_N}. "
                    f"Hint: Initialize TGN with GLOBAL num_nodes when using partitions (Data.n_id)."
                )

        # Edge features and timestamps (optional in batches)
        timestamps = getattr(data, 'edge_time', torch.zeros(src.size(0), device=src.device))
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            edge_attr = torch.zeros((src.size(0), 1), device=src.device)

        # Build messages using global memory states, then update destination node memories
        src_mem = self.memory(src_global)
        dst_mem = self.memory(dst_global)
        messages = self.message_module(src_mem, dst_mem, edge_attr)
        self.memory.update(dst_global, messages, timestamps)

        # Convolution and classification over the current (sub)graph node set only
        node_mem = self.memory(node_ids_for_conv)
        z = self.conv(node_mem, data.edge_index)
        return self.classifier(z)


# === TGAT Components ===
class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, time_dim, 2).float() / time_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        sinusoid_inp = t.view(-1, 1) * self.inv_freq.view(1, -1)
        return torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)


# === TGAT Model ===
class TGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, num_layers=2, dropout=0.5, time_dim=16):
        super().__init__()
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        self.convs = nn.ModuleList()
        # Note: TGAT's original attention mechanism is more complex. This uses TransformerConv as a proxy.
        # Layer 1
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(
                TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# === AGNNet Components ===
class PriorityWeightedConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, alpha):
        x = self.lin(x)
        return self.propagate(edge_index, x=x, alpha=alpha)

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)


class AquaGraph:
    def __init__(self, tau=0.9, k=8, num_layers=3, soft_topk=True, edge_threshold=0.0):
        self.tau, self.k, self.K = tau, k, num_layers
        self.soft_topk = soft_topk
        self.edge_threshold = edge_threshold

    def compute_priority(self, x, x_prev, edge_index, wp, alpha_ij):
        N, src, dst = x.size(0), edge_index[0], edge_index[1]
        delta_x = (x - x_prev).abs().sum(dim=1)
        delta_agg = alpha_ij * delta_x[src]
        neigh_sum = torch.zeros(N, device=x.device).index_add_(0, dst, delta_agg)
        score = x @ wp + neigh_sum
        return torch.sigmoid(score).view(-1)

    def compute_alpha(self, x_proj, edge_index, pi, att_mlp, tau=1.0, clip=5.0, topk=None):
        src, dst = edge_index
        h_i, h_j, p_j = x_proj[dst], x_proj[src], pi[src].unsqueeze(1)
        e_ij = torch.cat([h_i, h_j, p_j], dim=1)
        e = att_mlp(e_ij).squeeze(-1)
        # stability: pre-norm will be applied outside; here clamp and temperature
        e = F.leaky_relu(e, 0.2)
        if tau is not None and tau > 0:
            e = e / tau
        e = torch.clamp(e, -clip, clip)

        # compute soft attention per-destination node with optional threshold and top-k cap
        # compute exp and denom per dst
        exp_e = torch.exp(e)
        # apply edge threshold by masking small logits first if requested
        if self.edge_threshold is not None and self.edge_threshold > 0:
            mask = e >= self.edge_threshold
            exp_e = exp_e * mask.float()
        denom = torch.zeros(x_proj.size(0), device=x_proj.device).index_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-16)

        if self.soft_topk and (topk is not None and topk > 0):
            try:
                # keep top-k edges per destination, renormalize on current device (GPU if available)
                num_edges = dst.numel()
                device = x_proj.device
                alpha_new = torch.zeros_like(alpha)
                # group by dst node via sort to process contiguous blocks
                order = torch.argsort(dst)
                dst_sorted = dst[order]
                start = 0
                # iterate over contiguous blocks of same dst
                while start < num_edges:
                    node = dst_sorted[start].item()
                    end = start
                    while end < num_edges and dst_sorted[end].item() == node:
                        end += 1
                    block_idx = order[start:end]
                    block_alpha = alpha[block_idx]
                    if block_alpha.numel() > 0:
                        k_eff = min(topk, block_alpha.numel())
                        vals, idxs = torch.topk(block_alpha, k_eff, largest=True)
                        sel = block_idx[idxs]
                        # renormalize
                        s = vals.sum() + 1e-16
                        alpha_new[sel] = vals / s
                    start = end
                alpha = alpha_new
            except torch.cuda.OutOfMemoryError:
                # CPU fallback for soft top-k renormalization to avoid CUDA OOM
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                device = alpha.device
                dst_cpu = dst.to('cpu')
                alpha_cpu = alpha.to('cpu')
                num_edges = dst_cpu.numel()
                alpha_new_cpu = torch.zeros_like(alpha_cpu)
                order = torch.argsort(dst_cpu)
                dst_sorted = dst_cpu[order]
                start = 0
                while start < num_edges:
                    node = dst_sorted[start].item()
                    end = start
                    while end < num_edges and dst_sorted[end].item() == node:
                        end += 1
                    block_idx = order[start:end]
                    block_alpha = alpha_cpu[block_idx]
                    if block_alpha.numel() > 0:
                        k_eff = min(topk, block_alpha.numel())
                        vals, idxs = torch.topk(block_alpha, k_eff, largest=True)
                        sel = block_idx[idxs]
                        s = vals.sum() + 1e-16
                        alpha_new_cpu[sel] = vals / s
                    start = end
                alpha = alpha_new_cpu.to(device)
        return alpha

    def select_subgraph(self, pi):
        # soft approach: keep nodes with probability over threshold; this is still used to restrict compute
        return (pi >= self.tau).nonzero(as_tuple=False).view(-1)

    def local_neighbors(self, nodes, edge_index, num_nodes, k):
        if nodes.numel() == 0:
            return torch.arange(num_nodes, device=edge_index.device), edge_index
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
            nodes, k, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        return sub_nodes, sub_edge_index


class PreNormBlock(nn.Module):
    def __init__(self, dim, ffn_expansion=2, dropout=0.25):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def attn(self, x, edge_index, alpha, conv: MessagePassing):
        # attention conv expects properly normalized alpha
        return conv(x, edge_index, alpha=alpha)

    def forward(self, x, edge_index, alpha, conv: MessagePassing):
        # pre-norm + residual around attention
        h = self.ln1(x)
        h = self.attn(h, edge_index, alpha, conv)
        x = x + self.dropout(F.relu(h))
        # pre-norm + residual around FFN
        h2 = self.ln2(x)
        h2 = self.ffn(h2)
        x = x + self.dropout(h2)
        return x


class AGNNet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        tau=0.9,
        k=8,
        num_layers=3,
        dropout=0.25,
        ffn_expansion=2.0,
        soft_topk=True,
        edge_threshold=0.0,
        disable_pred_subgraph=False,
        add_self_loops=True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            PreNormBlock(hidden_channels, ffn_expansion=ffn_expansion, dropout=dropout) for _ in range(num_layers)
        ])
        self.convs = nn.ModuleList([
            PriorityWeightedConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.aqua = AquaGraph(tau=tau, k=k, num_layers=num_layers, soft_topk=soft_topk, edge_threshold=edge_threshold)
        self.att_mlp = nn.Linear(2 * hidden_channels + 1, 1)
        self.wp = nn.Parameter(torch.randn(hidden_channels, 1))

        self.register_buffer('x_prev', None)

        # Annealing for k
        self._anneal_k = False
        self._k_min = max(1, min(2, k))
        self._k_max = int(k)
        self._cur_k = int(k)
        self._total_epochs = 1
        self._epoch = 0

        self.disable_pred_subgraph = disable_pred_subgraph
        self.add_self_loops = add_self_loops
        self.tau = tau

    def enable_k_annealing(self, k_min=2, k_max=None, total_epochs=100):
        self._anneal_k = True
        self._k_min = int(k_min)
        self._k_max = int(k_max if k_max is not None else self._k_max)
        self._total_epochs = max(1, int(total_epochs))

    def set_epoch(self, epoch, total_epochs=None):
        self._epoch = int(epoch)
        if total_epochs is not None:
            self._total_epochs = int(total_epochs)
        if self._anneal_k:
            t = min(1.0, max(0.0, (self._epoch - 1) / max(1, self._total_epochs - 1)))
            self._cur_k = int(round(self._k_min + t * (self._k_max - self._k_min)))
        else:
            self._cur_k = self._k_max

    def _ensure_prev(self, x):
        if self.x_prev is None or self.x_prev.size(0) != x.size(0) or self.x_prev.size(1) != x.size(1):
            self.x_prev = torch.zeros_like(x)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        num_nodes = x.size(0)
        x = self.input_proj(x)
        x = F.relu(x)

        self._ensure_prev(x)

        # Priority scores (detach to avoid straight-through)
        with torch.no_grad():
            dummy_alpha = torch.ones(edge_index.size(1), device=x.device)
            pi = self.aqua.compute_priority(x, self.x_prev, edge_index, self.wp.squeeze(), dummy_alpha)

        # Decide subgraph scope
        if self.disable_pred_subgraph:
            selected_nodes = torch.arange(num_nodes, device=x.device)
        else:
            selected_nodes = self.aqua.select_subgraph(pi)

        cur_k = int(getattr(self, '_cur_k', self.aqua.k))
        sub_nodes, sub_edge_index = self.aqua.local_neighbors(selected_nodes, edge_index, num_nodes=num_nodes, k=cur_k)

        # Guarantee self-loops for stability
        if self.add_self_loops:
            if sub_nodes.numel() > 0:
                local_n = sub_nodes.numel()
                loops = torch.arange(local_n, device=sub_nodes.device)
                loops = torch.stack([loops, loops], dim=0)
                sub_edge_index = torch.cat([sub_edge_index, loops], dim=1)

        if sub_nodes.numel() == 0:
            return torch.zeros(num_nodes, self.output_proj.out_features, device=x.device)

        sub_x = x[sub_nodes]
        sub_pi = pi[sub_nodes]

        # Compute soft attention with temperature tau and top-k cap
        alpha_ij = self.aqua.compute_alpha(sub_x, sub_edge_index, sub_pi, self.att_mlp, tau=self.tau, clip=5.0, topk=cur_k)

        # Stacked pre-norm attention + FFN blocks with residuals
        h = sub_x
        for blk, conv in zip(self.blocks, self.convs):
            h = blk(h, sub_edge_index, alpha_ij, conv)

        # Project to logits and scatter back
        sub_logits = self.output_proj(h)  # raw logits (no softmax)
        full_logits = torch.zeros(num_nodes, sub_logits.shape[1], device=x.device)
        full_logits[sub_nodes] = sub_logits

        # Update memory only on visited nodes
        self.x_prev[sub_nodes] = x[sub_nodes].detach()
        return full_logits






