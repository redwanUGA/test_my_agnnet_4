import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv as DGLGraphConv, SAGEConv as DGLSAGEConv
from torch_geometric.nn import TransformerConv, MessagePassing
from torch_geometric.utils import k_hop_subgraph


# === BaselineGCN ===
class BaselineGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = DGLGraphConv(in_channels, hidden_channels)
        self.conv2 = DGLGraphConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, g):
        x = g.ndata['feat']
        x = F.relu(self.conv1(g, x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(g, x)
        return x


# === GraphSAGE ===
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(DGLSAGEConv(in_channels, hidden_channels, 'mean'))
        for _ in range(num_layers - 2):
            self.convs.append(DGLSAGEConv(hidden_channels, hidden_channels, 'mean'))
        self.convs.append(DGLSAGEConv(hidden_channels, out_channels, 'mean'))
        self.dropout = dropout

    def forward(self, g):
        x = g.ndata['feat']
        for i, conv in enumerate(self.convs):
            x = conv(g, x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# === TGN Components ===
class TGNMemory(nn.Module):
    def __init__(self, num_nodes, memory_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=True)
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)
        self.update_fn = nn.GRUCell(memory_dim, memory_dim)

    def forward(self, node_ids):
        return self.memory[node_ids]

    def update(self, node_ids, messages, timestamps):
        prev_mem = self.memory[node_ids]
        updated_mem = self.update_fn(messages, prev_mem)
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
        src, dst = data.edge_index
        timestamps = getattr(data, 'edge_time', torch.zeros(src.size(0), device=src.device))
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            edge_attr = torch.zeros((src.size(0), 1), device=src.device)

        src_mem = self.memory(src)
        dst_mem = self.memory(dst)
        messages = self.message_module(src_mem, dst_mem, edge_attr)
        self.memory.update(dst, messages, timestamps)

        all_node_ids = torch.arange(self.memory.num_nodes, device=src.device)
        full_node_mem = self.memory(all_node_ids)
        z = self.conv(full_node_mem, data.edge_index)
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
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, alpha):
        x = self.linear(x)
        return self.propagate(edge_index, x=x, alpha=alpha)

    def message(self, x_j, alpha):
        return alpha.unsqueeze(-1) * x_j


class AquaGraph:
    def __init__(self, tau=0.9, k=2, num_layers=3):
        self.tau, self.k, self.K = tau, k, num_layers

    def compute_priority(self, x, x_prev, edge_index, wp, alpha_ij):
        N, src, dst = x.size(0), edge_index[0], edge_index[1]
        delta_x = (x - x_prev).abs().sum(dim=1)
        delta_agg = alpha_ij * delta_x[src]
        neigh_sum = torch.zeros(N, device=x.device).index_add_(0, dst, delta_agg)
        score = x @ wp + neigh_sum
        return torch.sigmoid(score).view(-1)

    def compute_alpha(self, x_proj, edge_index, pi, att_mlp):
        src, dst = edge_index
        h_i, h_j, p_j = x_proj[dst], x_proj[src], pi[src].unsqueeze(1)
        e_ij = torch.cat([h_i, h_j, p_j], dim=1)
        e = F.leaky_relu(att_mlp(e_ij).squeeze(-1), 0.2)
        exp_e = torch.exp(e)
        denom = torch.zeros(x_proj.size(0), device=x_proj.device).index_add_(0, dst, exp_e)
        return exp_e / (denom[dst] + 1e-16)

    def select_subgraph(self, pi):
        return (pi >= self.tau).nonzero(as_tuple=False).view(-1)

    def local_neighbors(self, nodes, edge_index, num_nodes):
        if nodes.numel() == 0:
            return torch.arange(num_nodes, device=edge_index.device), edge_index
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
            nodes, self.k, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        return sub_nodes, sub_edge_index


class AGNNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, tau=0.9, k=2, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList([
            PriorityWeightedConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.aqua = AquaGraph(tau=tau, k=k, num_layers=num_layers)
        self.att_mlp = nn.Linear(2 * hidden_channels + 1, 1)
        self.wp = nn.Parameter(torch.randn(hidden_channels, 1))

        self.register_buffer('x_prev', None)

        # 🧠 Caching to avoid recomputing subgraph structure
        self.cached_sub_nodes = None
        self.cached_sub_edge_index = None

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        num_nodes = x.size(0)
        x = F.relu(self.input_proj(x))

        # Initialize x_prev lazily
        if self.x_prev is None or self.x_prev.size(0) != x.size(0):
            self.x_prev = torch.zeros_like(x)

        # === Compute priority scores ===
        with torch.no_grad():
            dummy_alpha = torch.ones(edge_index.size(1), device=x.device)
            pi = self.aqua.compute_priority(x, self.x_prev, edge_index, self.wp.squeeze(), dummy_alpha)

        # === Cache or compute subgraph ===
        if self.cached_sub_nodes is None or self.training:
            selected_nodes = self.aqua.select_subgraph(pi)
            sub_nodes, sub_edge_index = self.aqua.local_neighbors(selected_nodes, edge_index, num_nodes=x.size(0))

            # Vectorized remapping using tensor indexing
            mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
            mapping[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
            mapped_edge_index = mapping[sub_edge_index]
            valid = (mapped_edge_index[0] >= 0) & (mapped_edge_index[1] >= 0)
            mapped_edge_index = mapped_edge_index[:, valid]

            self.cached_sub_nodes = sub_nodes
            self.cached_sub_edge_index = sub_edge_index
            self.cached_mapped_edge_index = mapped_edge_index
        else:
            sub_nodes = self.cached_sub_nodes
            mapped_edge_index = self.cached_mapped_edge_index

        if sub_nodes.numel() == 0:
            return torch.zeros(num_nodes, self.output_proj.out_features, device=x.device)

        # === Extract subgraph features and priority ===
        sub_x = x[sub_nodes]
        sub_pi = pi[sub_nodes]

        # === Compute attention weights ===
        alpha_ij = self.aqua.compute_alpha(sub_x, mapped_edge_index, sub_pi, self.att_mlp)

        # === Apply PriorityWeightedConv layers ===
        h = sub_x
        for conv in self.layers:
            h = conv(h, mapped_edge_index, alpha=alpha_ij)
            h = self.dropout(F.relu(h))

        # === Project to logits and fill into full logits tensor ===
        sub_logits = self.output_proj(h)
        full_logits = torch.zeros(num_nodes, sub_logits.shape[1], device=x.device)
        full_logits[sub_nodes] = sub_logits

        # === Update x_prev only on sub_nodes ===
        self.x_prev[sub_nodes] = x[sub_nodes].detach()

        return full_logits






