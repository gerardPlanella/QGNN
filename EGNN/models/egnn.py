import torch
import torch.nn as nn
from torch_scatter import scatter_add

from .utils import get_pe_attribute
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


class EGNNLSPELayer(nn.Module):
    """EGNN layer augmented with LSPE

    Args:
        hidden_channels (int): Number of hidden units
        **kwargs: Additional keyword arguments
    """

    def __init__(self, hidden_channels, **kwargs):
        super().__init__()
        self.include_dist = kwargs['include_dist']

        # Message network: phi_m
        self.message_mlp = nn.Sequential(
            nn.Linear(4 * hidden_channels + 1, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.SiLU())

        # Positional message network: phi_h
        self.message_pos_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + 1, hidden_channels), nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels), nn.Tanh())

        # Update network: phi_h
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels))

        # Positional update network: phi_h
        self.update_pos_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels), nn.Tanh())

    def forward(self, x, pos, edge_index, pe):
        send, rec = edge_index

        # Initialize the message state with the node features and the positional encoding
        state = torch.cat((torch.cat([x[send], pe[send]], dim=-1),
                           torch.cat([x[rec], pe[rec]], dim=-1)), dim=1)

        # Initialize the positional message state with the positional encoding
        state_pe = torch.cat([pe[send], pe[rec]], dim=1)

        # Add the distance between nodes to the state if required
        if self.include_dist:
            dist = torch.norm(pos[send] - pos[rec], dim=1).unsqueeze(1)
            state = torch.cat((state, dist), dim=1)
            state_pe = torch.cat((state, dist), dim=1)

        # Pass both states through the message nets
        message = self.message_mlp(state)
        pos = self.message_pos_mlp(state_pe)

        # Aggregate state messages from neighbours by summing
        aggr = scatter_add(message, rec, dim=0)

        # Pass the new state through the update network alongside x
        update_state = [x, pe, aggr] if self.update_with_pe else [x, aggr]
        update = self.update_mlp(torch.cat(update_state, dim=1))

        # Aggregate pos from neighbourhood by summing
        pos_aggr = scatter_add(pos, rec, dim=0)

        # Pass the new pos state through the update network alongside pe
        update_pe = self.update_mlp_pos(torch.cat([pe, pos_aggr], dim=1))

        return update, update_pe


class EGNN(nn.Module):
    """EGNN model

    Args:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden units
        num_layers (int): Number of layers
        out_channels (int): Number of output features
        pe (str): Positional encoding type
        pe_dim (int): Positional encoding dimension
        lspe (bool): Whether to use LSPE
        **kwargs: Additional keyword arguments
    """
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels,
                 pe='rw', pe_dim=24, lspe=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.pe = pe
        self.pe_dim = pe_dim if pe != 'nope' else 0
        self.lspe = lspe
        self.include_dist = kwargs['include_dist']

        # Pre-condition in case we are using LSPE
        assert not (self.pe == 'nope' and self.lspe), "LSPE has to have initialized PE."
        
        # Initialization of embedder for the input features (node features + (optional) PE dim)
        self.embed = nn.Sequential(
            nn.Linear(self.in_channels + self.pe_dim, self.hidden_channels), nn.SiLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels))

        # If the LSPE framework is used, add another embedder
        if self.lspe:
            self.embed_pe = nn.Sequential(
                nn.Linear(self.pe_dim, self.hidden_channels), nn.SiLU(),  # what is this 15?
                nn.Linear(self.hidden_channels, self.hidden_channels))

        # Initialization of hidden EGNN with (optional) LSPE hidden layers
        layer = EGNNLSPELayer if self.lspe else EGNNLayer
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

        # Add the positional encoding to the node features
        if self.pe != 'nope':
            # Pass the node features through the embedder
            pe = getattr(data, get_pe_attribute(self.pe))
            x = torch.cat([x, pe], dim=-1)

            # In the case of LSPE, pass PE through embedder
            pe = self.embed_pe(pe) if self.lspe else pe

        # Pass the node features through the embedder
        x = self.embed(x)

        for layer in self.layers:
            if self.lspe:
                # In the case of LSPE, pass PE through embedder
                out, pe_out = layer(x, pos, edge_index, pe)
                x += out
                pe += pe_out
            else:
                # Otherwise, just pass the node features
                x = x + layer(x, pos, edge_index)

        # Readout
        x = self.pre_readout(x)
        x = global_add_pool(x, batch)
        out = self.readout(x)

        return torch.squeeze(out)
