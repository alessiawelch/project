from enum import Enum, auto

from tasks.dictionary_lookup import DictionaryLookupDataset

import torch
from torch import nn
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
            return HybridSAGEConv(in_dim, out_dim, local_aggr='max')  
            # Or 'mean' / 'max' — choose whichever local aggregator you prefer.
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
