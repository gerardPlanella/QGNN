import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected

class IsingModelDataset(Dataset):
    def __init__(self, root, data_file, transform=None, pre_transform=None):
        super(IsingModelDataset, self).__init__(root, transform, pre_transform)
        
        self.data_file = data_file
        # Load .npy file
        self.data_array = np.load(os.path.join(self.root, self.data_file), allow_pickle=True)
        
    def len(self):
        return len(self.data_array)

    def get(self, idx, flatten=False):
        hamiltonian, ground_state_energy, node_rdms, edge_rdms = self.data_array[idx]

        # Generate edge indices for a fully connected graph
        num_nodes = len(node_rdms)  # Number of nodes
        edge_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
        edge_indices_undirected = to_undirected(edge_indices)

        # Convert to PyTorch tensors
        ground_state_energy = torch.tensor(ground_state_energy, dtype=torch.float32)
        h = torch.tensor(hamiltonian['h'], dtype=torch.float32)
        g = torch.tensor(hamiltonian['g'], dtype=torch.float32)
        
        # Get coupling and local fields from Hamiltonian
        J = hamiltonian['J']
        edge_indices_undirected = edge_indices_undirected.clone().detach() 
        J = torch.tensor(np.array([J[index] for index in edge_indices_undirected]), dtype=torch.float32)
        local_field_strength = torch.stack([h, g], dim=1)

        # Flatten node and edge matrices
        if flatten:
            node_rdms = [rdm.flatten() for rdm in node_rdms]
            edge_rdms = [rdm.flatten() for rdm in edge_rdms]

        # Create a PyTorch Geometric Data object
        pyg_data = Data(x_nodes=local_field_strength, x_edges=J, edge_index=edge_indices_undirected, y_quantum_node=node_rdms, y_quantum_edge=edge_rdms, y=ground_state_energy)

        return pyg_data

if __name__ == "__main__":
    # Create an instance of your custom dataset and call process
    data_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/nk_(12,)_False.npy"
    dataset = IsingModelDataset(root=os.path.dirname(data_file), data_file=os.path.basename(data_file))

    # Example of accessing a specific data point
    rand_idx = np.random.randint(0, len(dataset))
    data_point = dataset[rand_idx]
    print(data_point)
    #data = np.load("C:\\Users\\gerar_0ev1q4m\\OneDrive\\Documents\\AI\\QGNN\\src\\QGNN\\data\\nk_(12,)_False.npy", allow_pickle = True)
