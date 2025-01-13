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

class MaxSumSAGEConv(SAGEConv):
    def __init__(self, in_channels, out_channels, combine_mode='average', **kwargs):
        super().__init__(in_channels, out_channels, aggr='max', **kwargs)
        self.combine_mode = combine_mode
        if self.combine_mode == 'learnable':
                self.weight_max = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
                self.weight_sum = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, x, edge_index, size=None):
        # Save the old aggregation method
        old_aggr = self.aggr
        
        # Perform 'max' aggregation
        self.aggr = 'max'
        out_max = super().forward(x, edge_index, size=size)

        # Perform 'sum' aggregation
        self.aggr = 'sum'
        out_sum = super().forward(x, edge_index, size=size)

        # Restore the old aggregation method
        self.aggr = old_aggr

        # Combine outputs based on the mode
        if self.combine_mode == 'average':
            out = 0.5 * (out_max + out_sum)
        elif self.combine_mode == 'concat':
            # Handle dimension mismatch for concatenation
            print(out_max.size(-1))
            print(out_sum.size(-1))
            if out_max.size(-1) != out_sum.size(-1):
                diff = out_max.size(-1) - out_sum.size(-1)
                if diff > 0:
                    out_sum = F.pad(out_sum, (0, diff))  # Pad out_sum to match out_max
                elif diff < 0:
                    out_max = F.pad(out_max, (0, -diff))  # Pad out_max to match out_sum
            # Concatenate the aligned tensors
            out = torch.cat([out_max, out_sum], dim=-1)
        elif self.combine_mode == 'learnable':
            # Normalize the learnable weights
            total_weight = self.weight_max + self.weight_sum
            normalized_weight_max = self.weight_max / total_weight
            normalized_weight_sum = self.weight_sum / total_weight

            # Print the normalized weights
            print(f"Normalized weights: max = {normalized_weight_max.item()}, sum = {normalized_weight_sum.item()}")

            # Combine outputs using normalized weights
            out = normalized_weight_max * out_max + normalized_weight_sum * out_sum
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

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
    GSAGE_MAXSUM = auto()
    


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
            return GatingHybridSAGEConv(in_dim, out_dim, local_aggr='max')
        elif self is GNN_TYPE.GSAGE_GATED_HYBRID_MEAN:
            # Return our custom Hybrid aggregator
            return GatingHybridSAGEConv(in_dim, out_dim, local_aggr='mean')
        elif self is GNN_TYPE.GSAGE_GATED_HYBRID_SUM:
            # Return our custom Hybrid aggregator
            return GatingHybridSAGEConv(in_dim, out_dim, local_aggr='sum')
        elif self is GNN_TYPE.GSAGE_MAXSUM:
            return MaxSumSAGEConv(in_dim, out_dim, combine_mode='concat')
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
