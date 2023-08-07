import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn import global_add_pool


class EGNNLayer(nn.Module):
    """Standard EGNN layer

    Args:
        hidden_channels (int): Number of hidden units
        **kwargs: Additional keyword arguments
    """

    def __init__(self, hidden_channels, **kwargs):
        super().__init__()

        # Message network: phi_m
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + 1, hidden_channels),
            nn.SiLU(), nn.Linear(hidden_channels, hidden_channels), nn.SiLU())

        # Update network: phi_h
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels))

    def forward(self, x, pos, edge_index):
        send, rec = edge_index

        # Compute the distance between nodes
        dist = torch.norm(pos[send] - pos[rec], dim=1)

        # Pass the state through the message net
        state = torch.cat((x[send], x[rec], dist.unsqueeze(1)), dim=1)
        message = self.message_mlp(state)

        # Aggregate pos from neighbourhood by summing
        aggr = scatter_add(message, rec, dim=0)

        # Pass the new state through the update network alongside x
        update = self.update_mlp(torch.cat((x, aggr), dim=1))
        return update


class EGNN(nn.Module):
    """EGNN model

    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden units
        num_layers (int): Number of layers
        out_channels (int): Number of output features
        **kwargs: Additional keyword arguments
    """
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,
         **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.include_dist = kwargs['include_dist']
        
        # Initialization of embedder for the input features 
        self.embed = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))

        # Initialization of hidden EGNN with (optional) LSPE hidden layers
        layer = EGNNLayer
        self.layers = nn.ModuleList([
            layer(self.hidden_channels, **kwargs) for _ in range(self.num_layers)])

        # Readout networks
        self.pre_readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, data):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch

        # Pass the node features through the embedder
        x = self.embed(x)

        for layer in self.layers:
            out =  layer(x, pos, edge_index)
            x += out

        # Readout
        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)

        return torch.squeeze(out)
