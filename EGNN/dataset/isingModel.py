import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected

class IsingModelDataset(Dataset):
    def __init__(self, root, data_file, transform=None, pre_transform=None):
        super(IsingModelDataset, self).__init__(root, transform, pre_transform)
        
        self.data_file = data_file
        # Load your .npy file
        self.data_array = np.load(os.path.join(self.root, self.data_file), allow_pickle=True)
        
    def len(self):
        return len(self.data_array)
    
    def get(self, idx):
        _, ground_state_energy, node_rdm, edge_rdm = self.data_array[idx]
            
        # Flatten node and edge matrices
        node_rdm_flat = [matrix.flatten() for matrix in node_rdm]
        edge_rdm_flat = [matrix.flatten() for matrix in edge_rdm]
        
        ground_state_energy = torch.tensor(ground_state_energy, dtype=torch.float32)
        
        # Generate edge indices for a fully connected graph
        num_nodes = len(node_rdm)  # Number of nodes
        edge_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        edge_indices_undirected = to_undirected(edge_indices)
        
        # Create a PyTorch Geometric Data object
        pyg_data = Data(x=node_rdm_flat, edge_index=edge_indices_undirected, edge_attr=edge_rdm_flat, y=ground_state_energy)
        
        return pyg_data


# Create an instance of your custom dataset and call process
dataset = IsingModelDataset(root='C:\\Users\\gerar_0ev1q4m\\OneDrive\\Documents\\AI\\QGNN\\src\\QGNN\\data', data_file='nk_12.npy')

# Example of accessing a specific data point
data_point = dataset[0]
