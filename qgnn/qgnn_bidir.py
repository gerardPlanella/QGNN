import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

class GNNLayerWithEdgeDirectionInputs(nn.Module):
    def __init__(self, node_channels, edge_channels):
        super(GNNLayerWithEdgeDirectionInputs, self).__init__()
        # Define networks for forward and backward edge updates
        self.edge_net_forward = nn.Sequential(
            nn.Linear(2 * node_channels + edge_channels, edge_channels),
            nn.ReLU(),
            nn.Linear(edge_channels, edge_channels)
        )
        self.edge_net_backward = nn.Sequential(
            nn.Linear(2 * node_channels + edge_channels, edge_channels),
            nn.ReLU(),
            nn.Linear(edge_channels, edge_channels)
        )
        # Define the node update network
        self.node_net = nn.Sequential(
            nn.Linear(node_channels + 2 * edge_channels, node_channels),  # Adjust for combined edge feature size
            nn.ReLU(),
            nn.Linear(node_channels, node_channels)
        )

    def forward(self, x_nodes, x_edges_fwd, x_edges_bwd, edge_index):
        sender, receiver = edge_index

        # Update forward edge features
        state_fwd = torch.cat((x_nodes[sender], x_nodes[receiver], x_edges_fwd), dim=1)
        x_edges_fwd_updated = self.edge_net_forward(state_fwd)

        # Update backward edge features
        state_bwd = torch.cat((x_nodes[receiver], x_nodes[sender], x_edges_bwd), dim=1)
        x_edges_bwd_updated = self.edge_net_backward(state_bwd)

        # Combine the updated forward and backward edge features
        x_edges_combined = torch.cat((x_edges_fwd_updated, x_edges_bwd_updated), dim=1)

        # Aggregation step for node updates
        num_nodes = x_nodes.size(0)
        aggr = torch.zeros(num_nodes, x_edges_combined.size(1), dtype=x_edges_combined.dtype, device=x_edges_combined.device)
        
        # Aggregate combined edge features to receiver nodes
        aggr.scatter_add_(0, receiver.view(-1, 1).expand_as(x_edges_combined), x_edges_combined)

        # Update node features using aggregated combined edge features
        node_state = torch.cat((x_nodes, aggr), dim=1)
        x_nodes_updated = self.node_net(node_state)

        return x_nodes_updated, x_edges_fwd_updated, x_edges_bwd_updated



class QGNN(nn.Module):
    def __init__(self, 
                 node_in_channels, node_channels, node_out_channels, 
                 edge_in_channels, edge_channels, edge_out_channels, 
                 num_layers, global_out_channels, **kwargs):
        super().__init__()
        # node
        self.node_in_channels = node_in_channels
        self.node_channels = node_channels
        self.node_out_channels = node_out_channels
        # edge
        self.edge_in_channels = edge_in_channels
        self.edge_channels = edge_channels
        self.edge_out_channels = edge_out_channels
        # other
        self.num_layers = num_layers
        self.global_out_channels = global_out_channels

        self.use_pbc = kwargs.get("use_pbc", False)

        # Initialization of embedders for the input features
        self.embed_nodes = nn.Sequential(
            nn.Linear(self.node_in_channels + (1 if self.use_pbc else 0), self.node_channels),
            nn.SiLU(),
            nn.Linear(self.node_channels, self.node_channels)
        )
        self.embed_edges = nn.Sequential(
            nn.Linear(self.edge_in_channels + (1 if self.use_pbc else 0), self.edge_channels),
            nn.SiLU(),
            nn.Linear(self.edge_channels, self.edge_channels)
        )

        # Initialization of the hidden layers
        self.layers = nn.ModuleList([
            GNNLayerWithEdgeDirectionInputs(self.node_channels, self.edge_channels) for _ in range(self.num_layers)
        ])

        # Readout networks
        self.node_readout = nn.Sequential(
            nn.Linear(self.node_channels, self.node_channels),
            nn.SiLU(),
            nn.Linear(self.node_channels, self.node_out_channels)
        )
        self.edge_readout = nn.Sequential(
            nn.Linear(self.edge_channels*2, self.edge_channels),
            nn.SiLU(),
            nn.Linear(self.edge_channels, self.edge_out_channels)
        )
        # Global readout from nodes
        self.global_readout = nn.Sequential(
            nn.Linear(self.node_out_channels, self.node_channels),
            nn.SiLU(),
            nn.Linear(self.node_channels, self.global_out_channels)
        )

    def forward(self, data):
        _x_nodes, _x_edges, edge_index, batch, pbc = data.x_nodes, data.x_edges, data.edge_index, data.batch, data.pbc.to(dtype=torch.float32)


        if torch.any(torch.isnan(_x_nodes)):
            raise ValueError(f"x_nodes contains nan: {_x_nodes}")
        
        # print the weights of embed_nodes
        # print(self.embed_nodes[0].weight)

        if self.use_pbc:
            node_pbc = torch.index_select(pbc, dim=0, index=batch).unsqueeze(1)
            edge_to_graph = torch.index_select(batch, 0, edge_index[0])
            edge_pbc = torch.index_select(pbc, 0, edge_to_graph).unsqueeze(1)

            x_nodes = torch.concatenate([_x_nodes, node_pbc], dim=1)
            x_edges = torch.concatenate([_x_edges, edge_pbc], dim=1)
            
        # Embed nodes and edges
        x_edges = self.embed_edges(x_edges)

        x_edges_fwd = x_edges
        x_edges_bwd = x_edges
        
        x_nodes = self.embed_nodes(x_nodes)

        if torch.any(torch.isnan(x_nodes)):
            raise ValueError(f"x_nodes contains nan: {x_nodes}, {data.x_nodes}")

        for layer in self.layers:
            x_nodes, x_edges_fwd, x_edges_bwd = layer(x_nodes, x_edges_fwd, x_edges_bwd, edge_index) 

        # Readout
        x_nodes = self.node_readout(x_nodes)        # x_nodes.shape  = [batch_size*num_nodes, node_out_channels]
        x_edges = self.edge_readout(torch.cat((x_edges_fwd, x_edges_bwd), dim=1)) # x_edges.shape  = [batch_size*num_edges*2, edge_out_channels]
        x_global = global_add_pool(x_nodes, batch)  # x_global.shape = [batch_size, node_out_channels]
        x_global = self.global_readout(x_global)    # x_global.shape = [batch_size, global_out_channels]

        if torch.any(torch.isnan(x_nodes)) or torch.any(torch.isnan(x_edges)) or torch.any(torch.isnan(x_global)):
            raise ValueError(f"Output contains nan: x_nodes={x_nodes}, x_edges={x_edges}, x_global={x_global}")

        return torch.squeeze(x_nodes), torch.squeeze(x_edges), torch.squeeze(x_global)     