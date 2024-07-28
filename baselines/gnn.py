import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

class GNNLayer(nn.Module):
    """GNN layer

    Args:
        hidden_channels (int): Number of hidden units
        **kwargs: Additional keyword arguments
    """

    def __init__(self, node_channels, edge_channels, **kwargs):
        super().__init__()

        # Message network: phi_m
        self.edge_net = nn.Sequential(
            nn.Linear(2 * node_channels + edge_channels, node_channels),
            nn.SiLU(),
            nn.Linear(node_channels, node_channels),
            nn.SiLU()
        )

        # Update network: phi_h
        self.node_net = nn.Sequential(
            nn.Linear(2 * node_channels, node_channels),
            nn.SiLU(),
            nn.Linear(node_channels, node_channels)
        )

    def forward(self, x_nodes, x_edges, edge_index):
        sender, receiver = edge_index

        # Generate the message send -> rec
        state = torch.cat((x_nodes[sender], x_nodes[receiver], x_edges), dim=1)
        message = self.edge_net(state)

        # Manually perform aggregation
        num_nodes = x_nodes.size(0)
        num_edges = sender.size(0)
        aggr = torch.zeros(num_nodes, message.size(1), dtype=message.dtype, device=message.device)
        aggr.scatter_add_(0, receiver.view(-1, 1).expand(num_edges, message.size(1)), message)

        # Pass the new state through the update network alongside x
        update = self.node_net(torch.cat((x_nodes, aggr), dim=1))
        return update


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_channels, num_layers, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.node_channels = hidden_channels
        self.edge_channels = edge_channels
        self.num_layers = num_layers
        self.out_channels = out_channels

        # Initialization of embedders for the input features
        self.embed_nodes = nn.Sequential(
            nn.Linear(self.in_channels, self.node_channels),
            nn.SiLU(),
            nn.Linear(self.node_channels, self.node_channels))

        # Initialization of hidden GNN with (optional) LSPE hidden layers
        self.layers = nn.ModuleList([
            GNNLayer(self.node_channels, self.edge_channels, **kwargs) for _ in range(self.num_layers)
        ])

        # Readout networks
        self.pre_readout = nn.Sequential(
            nn.Linear(self.node_channels, self.node_channels),
            nn.SiLU(),
            nn.Linear(self.node_channels, self.node_channels)
        )
        self.readout = nn.Sequential(
            nn.Linear(self.node_channels, self.node_channels),
            nn.SiLU(),
            nn.Linear(self.node_channels, self.out_channels)
        )

    def forward(self, data):
        x_nodes, x_edges, edge_index, batch = data.x_nodes, data.x_edges, data.edge_index, data.batch

        # Pass the node features through the node embedder
        x_nodes = self.embed_nodes(x_nodes)

        for layer in self.layers:
            x_nodes = layer(x_nodes, x_edges, edge_index) 

        # Readout
        x_nodes = self.pre_readout(x_nodes)
        x_nodes = global_add_pool(x_nodes, batch)
        out = self.readout(x_nodes)

        return torch.squeeze(out)
