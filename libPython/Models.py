import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Sequential, ReLU, Linear, Dropout
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.utils import dropout_edge

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(aggr="mean")
        self.mlp = Sequential(
                Linear(2*in_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels), Dropout(0.4),
                Linear(out_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels), Dropout(0.4),
                Linear(out_channels, hidden_channels), ReLU(), BatchNorm1d(hidden_channels), Dropout(0.4)
                )

    def forward(self, x, edge_index, batch=None):
        return self.propagate(edge_index, x=x, batch=batch)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p, training, k=4):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.shortcut = Sequential(Linear(in_channels, out_channels), ReLU())
        self.training = training
        self.dropout_p = dropout_p
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None: edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_p, training=self.training)
        out = super().forward(x, edge_index, batch=batch)
        out += self.shortcut(x)
        return out


class ParticleNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None, dropout_p=0.2):
        super(ParticleNet, self).__init__()
        self.gn0 = GraphNorm(in_channels)
        self.bn0 = BatchNorm1d(hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv1 = TransformerConv(in_channels, hidden_channels, dropout_p=dropout_p, edge_dim=edge_dim, training=self.training)
        self.conv2 = DynamicEdgeConv(hidden_channels, hidden_channels, hidden_channels, dropout_p=dropout_p, training=self.training, k=4)
        self.conv3 = DynamicEdgeConv(hidden_channels, hidden_channels, hidden_channels, dropout_p=dropout_p, training=self.training, k=4)
        self.dense1 = Linear(hidden_channels, hidden_channels)
        self.dense2 = Linear(hidden_channels, hidden_channels)
        self.output = Linear(hidden_channels, out_channels)
        self.dropout_p = dropout_p
    
    def forward(self, x, edge_index, edge_attribute=None, batch=None, return_attention_weights=False):
        # convolution layers
        x = self.gn0(x, batch=batch)
        conv1, attention_weights = self.conv1(x, edge_index, edge_attr=edge_attribute, return_attention_weights=True)
        conv1 = F.elu(conv1)
        conv2 = self.conv2(conv1, edge_index, batch=batch)
        conv3 = self.conv3(conv2, edge_index, batch=batch)
        x = conv1+conv2+conv3
        
        # readout layers
        x = global_mean_pool(x, batch=batch)
        
        # dense layers
        x = self.bn0(x)
        x = F.alpha_dropout(x, p=0.4)
        x = F.selu(self.dense1(x))
        x = F.alpha_dropout(x, p=0.4)
        x = F.selu(self.dense2(x))
        x = self.output(x)

        if return_attention_weights:
            return F.log_softmax(x, dim=1), attention_weights
        else:
            return F.log_softmax(x, dim=1)