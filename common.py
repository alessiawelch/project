from enum import Enum, auto

from tasks.dictionary_lookup import DictionaryLookupDataset

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, SAGEConv
        
class Task(Enum):
    NEIGHBORS_MATCH = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_dataset(self, depth, train_fraction):
        if self is Task.NEIGHBORS_MATCH:
            dataset = DictionaryLookupDataset(depth)
        else:
            dataset = None

        return dataset.generate_data(train_fraction)

class GatingHybridSAGEConv(SAGEConv):
    """
    An enhanced gating aggregator that merges local and global signals.
    
    1) Local aggregator (via parent SAGEConv).
    2) More expressive global embedding is computed via a small MLP:
         - Project each node: x_proj = ReLU(lin_global1(x))
         - Mean across nodes
         - Final transform: lin_global2
    3) Node-wise gating factor computed from [local_out || global_broadcast].
       gate[i] = sigmoid( gate_lin( concat(local_out[i], global_broadcast[i]) ) )
    4) Output is a linear interpolation between local_out and global_broadcast:
       out[i] = gate[i]*local_out[i] + (1 - gate[i])*global_broadcast[i]
    5) ReLU (optional) at the end.
    
    By default, local_aggr='sum'. You can set local_aggr='mean', 'max', or 'sum'.
    """
    def __init__(self, in_channels, out_channels, local_aggr='sum', hidden_global=None):
        """
        Args:
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            local_aggr (str): Type of local aggregator for SAGEConv ('mean', 'max', 'sum', etc.).
            hidden_global (int, optional): Hidden dimension for the global MLP.
                If None, defaults to out_channels.
        """
        super().__init__(in_channels, out_channels, aggr=local_aggr)
        
        if hidden_global is None:
            hidden_global = out_channels  # default hidden dimension for the global MLP

        # (A) MLP layers for a more expressive global embedding:
        #     Instead of just mean->linear, we project each node to a hidden space,
        #     then mean-pool, then transform to out_channels.
        self.lin_global1 = nn.Linear(in_channels, hidden_global, bias=False)
        self.lin_global2 = nn.Linear(hidden_global, out_channels, bias=False)

        # (B) Gating MLP: we need 2*out_channels (local_out + global_broadcast) -> 1 scalar gate
        self.gate_lin = nn.Linear(2 * out_channels, 1)

    def forward(self, x, edge_index, size=None):
        # 1) Local aggregator pass via SAGEConv
        local_out = super().forward(x, edge_index, size=size)
        # local_out shape: [num_nodes, out_channels]

        # 2) Compute a more expressive global vector:
        #    2a) Project each node's input features -> hidden_global, then ReLU
        x_proj = F.relu(self.lin_global1(x))  
        #    2b) Mean-pool across all nodes to get a single graph embedding
        graph_emb = x_proj.mean(dim=0, keepdim=True)   # shape [1, hidden_global]
        #    2c) Transform that embedding to out_channels
        global_vec = self.lin_global2(graph_emb)       # shape [1, out_channels]

        # 3) Broadcast the global embedding to all nodes
        N = local_out.size(0)
        global_broadcast = global_vec.repeat(N, 1)     # shape [N, out_channels]

        # 4) Compute a node-wise gating factor based on both local + global signals
        concat_features = torch.cat([local_out, global_broadcast], dim=-1)  # shape [N, 2*out_channels]
        gate_logit = self.gate_lin(concat_features)                         # shape [N, 1]
        gate = torch.sigmoid(gate_logit)                                    # node-wise alpha in (0,1)

        # 5) Fuse local and global with an alpha-blend
        #    out[i] = gate[i] * local_out[i] + (1 - gate[i]) * global_broadcast[i]
        out = gate * local_out + (1.0 - gate) * global_broadcast

        # 6) Optional nonlinearity
        out = F.relu(out)
        return out

class HybridSAGEConv(SAGEConv):
    """
    A Hybrid aggregator that merges a local aggregator (e.g. sum) 
    with a global average embedding. This can help mitigate oversquashing
    by providing a 'global' shortcut each layer.
    """
    def __init__(self, in_channels, out_channels, local_aggr='sum'):
        super().__init__(in_channels, out_channels, aggr=local_aggr)
        # We'll define a linear to transform the global embedding
        # and a combine layer to merge local+global embeddings.
        self.lin_global = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_combine = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index, size=None):
        # x: Node features [N, in_channels]
        # 1) Local aggregator pass:
        x_local = super().forward(x, edge_index, size=size)  
        # This calls SAGEConv's forward, returning [N, out_channels]

        # 2) Compute global average in *input* space:
        global_vec = x.mean(dim=0, keepdim=True)        # shape [1, in_channels]
        global_vec = self.lin_global(global_vec)        # shape [1, out_channels]

        # 3) Broadcast global embedding to each node:
        N = x_local.size(0)
        global_broadcast = global_vec.repeat(N, 1)      # shape [N, out_channels]

        # 4) Merge local + global
        combined = torch.cat([x_local, global_broadcast], dim=-1)
        out = self.lin_combine(combined)                # shape [N, out_channels]
        #out = F.relu(out)                               # optional nonlinearity
        return out
        
class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    GSAGE_MEAN = auto()  # GraphSAGE with mean aggregation
    GSAGE_MAX = auto()   # GraphSAGE with max aggregation
    GSAGE_MIN = auto()   # GraphSAGE with min aggregation
    GSAGE_SUM = auto()   # GraphSAGE with sum aggregation
    GSAGE_HYBRID = auto()
    GSAGE_GATED_HYBRID_SUM = auto()
    GSAGE_GATED_HYBRID_MAX = auto()
    GSAGE_GATED_HYBRID_MEAN = auto()
    


    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim)
        elif self is GNN_TYPE.GGNN:
            return GatedGraphConv(out_channels=out_dim, num_layers=1)
        elif self is GNN_TYPE.GIN:
            return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                         nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self in {GNN_TYPE.GSAGE_MEAN, GNN_TYPE.GSAGE_MAX, GNN_TYPE.GSAGE_MIN, GNN_TYPE.GSAGE_SUM}:
            # Map enum types to aggregation strings
            aggregation_map = {
                GNN_TYPE.GSAGE_MEAN: "mean",
                GNN_TYPE.GSAGE_MAX: "max",
                GNN_TYPE.GSAGE_MIN: "min",
                GNN_TYPE.GSAGE_SUM: "sum"
            }
            aggregation_type = aggregation_map[self]
            return SAGEConv(
                in_channels=in_dim,
                out_channels=out_dim,
                aggr=aggregation_type
            )
        elif self is GNN_TYPE.GSAGE_HYBRID:
            # Return our custom Hybrid aggregator
            return HybridSAGEConv(in_dim, out_dim, local_aggr='mean')  
            # Or 'mean' / 'max' — choose whichever local aggregator you prefer.
        elif self is GNN_TYPE.GSAGE_GATED_HYBRID_MAX:
            # Return our custom Hybrid aggregator
            return HybridSAGEConv(in_dim, out_dim, local_aggr='max')
        elif self is GNN_TYPE.GSAGE_GATED_HYBRID_MEAN:
            # Return our custom Hybrid aggregator
            return HybridSAGEConv(in_dim, out_dim, local_aggr='mean')
        elif self is GNN_TYPE.GSAGE_GATED_HYBRID_SUM:
            # Return our custom Hybrid aggregator
            return HybridSAGEConv(in_dim, out_dim, local_aggr='sum')
        else:
            raise ValueError("Unsupported GNN type")


class STOP(Enum):
    TRAIN = auto()
    TEST = auto()

    @staticmethod
    def from_string(s):
        try:
            return STOP[s]
        except KeyError:
            raise ValueError()


def one_hot(key, depth):
    return [1 if i == key else 0 for i in range(depth)]
