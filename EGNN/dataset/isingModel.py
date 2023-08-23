import os
import numpy as np
import torch
import pickle
from tqdm import tqdm  # Import tqdm for the progress bar
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_undirected


class IsingModelDataset(Dataset):
    def __init__(self, root, data_file, transform=None, pre_transform=None, flatten=False, verbose=True):
        super(IsingModelDataset, self).__init__(root, transform, pre_transform)
        
        self.data_file = data_file
        self.flatten = flatten
        self.verbose = verbose  # Add a verbose flag

        # Load .npy file and preprocess all data points
        self.data_array = self._load_and_preprocess_data()

    def len(self):
        return len(self.data_array)

    def get(self, idx):
        # Return the preprocessed data point at the given index
        return self.data_array[idx]

    def _load_and_preprocess_data(self):
        """
        Load and preprocess the entire dataset.

        Returns:
            data_array: A list of preprocessed data points.
        """
        data_array = np.load(os.path.join(self.root, self.data_file), allow_pickle=True)
        preprocessed_data_array = []

        # Use tqdm for progress bar
        if self.verbose:
            data_array = tqdm(data_array, desc="Preprocessing", dynamic_ncols=True)

        for data in data_array:
            preprocessed_data_array.append(self._preprocess_data(data))

        return preprocessed_data_array

    def _preprocess_data(self, data):
        """
        Custom data point pre-processing method.

        Args:
            data: The data point to preprocess.

        Returns:
            preprocessed_data_point: The preprocessed data point.
        """
        hamiltonian, ground_state_energy, node_rdms, edge_rdms = data

        # Generate edge indices for a fully connected graph
        num_nodes = len(node_rdms)  # Number of nodes
        edge_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)

        # Convert to PyTorch tensors
        ground_state_energy = torch.tensor(ground_state_energy, dtype=torch.float32)

        # Hamiltonian parameters
        J = torch.tensor(hamiltonian['J'][edge_indices[0], edge_indices[1]], dtype=torch.float32)  # in case you need a version without the star ;)
        #J = torch.tensor(hamiltonian['J'][*edge_indices], dtype=torch.float32)
        h = torch.tensor(hamiltonian['h'], dtype=torch.float32)
        g = torch.tensor(hamiltonian['g'], dtype=torch.float32)
        local_field_strength = torch.stack([h, g], dim=1)

        # Create a PyTorch Geometric Data object
        pyg_data = Data(x_nodes=local_field_strength, x_edges=J, edge_index=edge_indices, y_quantum_node=node_rdms, y_quantum_edge=edge_rdms, y=ground_state_energy)

        return pyg_data

    def save(self, file_path):
        """
        Save the dataset to a file using pickle.

        Args:
            file_path (str): Path to the file where the dataset will be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        """
        Load a dataset from a file using pickle.

        Args:
            file_path (str): Path to the file containing the dataset.

        Returns:
            dataset: An instance of the IsingModelDataset class containing the loaded data.
        """
        with open(file_path, 'rb') as file:
            loaded_dataset = pickle.load(file)
        return loaded_dataset

# Example usage:
data_file = os.path.dirname(os.path.abspath(__file__)) + "/../../data/nk_(12,)_False.npy"
dataset = IsingModelDataset(root=os.path.dirname(data_file), data_file=os.path.basename(data_file))

# Save the dataset
save_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/100_nk_(12,)_False.pkl"
dataset.save(save_path)

# Load the saved dataset
loaded_dataset = IsingModelDataset.load(save_path)

# Access data points from the loaded dataset
rand_idx = np.random.randint(0, len(loaded_dataset))
data_point = loaded_dataset[rand_idx]
print(data_point)
