import os
import torch
from torch_geometric.data import Dataset
import numpy as np



class IsingModelDataset(Dataset):
    def __init__(self, file_path, transform=None, pre_transform=None):
        super(IsingModelDataset, self).__init__(os.path.dirname(file_path), transform, pre_transform)

        self.file_path = file_path
        self.data_array = torch.load(self.file_path)
        #Check if the data field original_index exists, if not, add it
        if not hasattr(self.data_array[0], 'original_index'):
            print("Adding original_index field to data points")
            self.idxs = torch.arange(len(self.data_array))
        else:
            print("Using original_index field from data points")
            self.idxs = torch.tensor([data.original_index for data in self.data_array])
        print("Loaded %d data points from %s" % (len(self.data_array), self.file_path))

    def len(self):
        return len(self.data_array)


    def get(self, idx):
        data_point = self.data_array[idx]
        original_index = self.idxs[idx]
        data_point.original_index = original_index
        # Return the preprocessed data point at the given index
        return data_point

    @classmethod
    def load(cls, file_path):
        """
        Load a dataset from a file.

        Args:
            file_path (str): Path to the file containing the dataset.

        Returns:
            dataset: An instance of the IsingModelDataset class containing the loaded data.
        """
        return IsingModelDataset(file_path)

def calculate_energy_single_graph(one_rdms, two_rdms, node_params, edge_params):
    # The energy is then the sum of the local energies, i.e. Tr(Hρ) = sum_i Tr(g_i σ_zρ_i) + sum_i Tr(h_i σ_x ρ_i) + sum_ij Tr(J_ij kron(σ_zσ_z) ρ_ij)

    # Imaginary part is 0
    pauli_z = torch.Tensor([[1, 0], [0, -1]])
    pauli_x = torch.Tensor([[0, 1], [1, 0]])
    
    if not torch.is_tensor(one_rdms):
        one_rdms = torch.Tensor(one_rdms)
    if not torch.is_tensor(two_rdms):
        two_rdms = torch.Tensor(two_rdms)
    if not torch.is_tensor(node_params):
        node_params = torch.Tensor(node_params)
    if not torch.is_tensor(edge_params):
        edge_params = torch.Tensor(edge_params)

    #Energy from the one-body RDMs sum_i Tr(g_i pauli_z ρ_i) + sum_i Tr(h_i pauli_x ρ_i)
    energy = 0
    for node in range(len(one_rdms)):
        energy += node_params[node, 1] * torch.trace(torch.mm(pauli_x, one_rdms[node])) + node_params[node, 0] * torch.trace(torch.mm(pauli_z, one_rdms[node]))
    
    #Energy from the two-body RDMs sum_ij Tr(J_ij kron(pauli_z pauli_z) ρ_ij)
    for edge in range(len(two_rdms)):
        energy += edge_params[edge][0] * torch.trace(torch.mm(torch.kron(pauli_z, pauli_z), two_rdms[edge]))
    

    return energy

# Same function but for batched data and no loops
def calculate_energy_batched(one_rdms, two_rdms, node_params, edge_params, batch_index, edge_index):
    # Imaginary part is 0
    pauli_z = torch.Tensor([[1.0, 0.0], [0.0, -1.0]]).to(dtype=torch.float64)
    pauli_x = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).to(dtype=torch.float64)
    
    if not torch.is_tensor(one_rdms):
        one_rdms = torch.Tensor(one_rdms).to(dtype=torch.float64)
    if not torch.is_tensor(two_rdms):
        two_rdms = torch.Tensor(two_rdms).to(dtype=torch.float64)
    if not torch.is_tensor(node_params):
        node_params = torch.Tensor(node_params).to(dtype=torch.float64)
    if not torch.is_tensor(edge_params):
        edge_params = torch.Tensor(edge_params).to(dtype=torch.float64)
    
    # Multiply each 1-RDM by pauli_z and pauli_x multplying with the correct parameter
    one_rdms_pauli_z = torch.einsum('bij,jk->bik', one_rdms, pauli_z)  
    one_rdms_pauli_x = torch.einsum('bij,jk->bik', one_rdms, pauli_x)

    # Perform trace and multiply with correct parameter
    local_energy = torch.einsum('bii,b->b', one_rdms_pauli_x, node_params[:, 1]) \
        + torch.einsum('bii,b->b', one_rdms_pauli_z, node_params[:, 0])
    
    two_rdms_kron_pauli_z = torch.einsum('bij,jk->bik', two_rdms, torch.kron(pauli_z, pauli_z))

    if len(edge_params.squeeze().size()) == 0:
        interaction_energy = torch.einsum('bii,b->b', two_rdms_kron_pauli_z, edge_params.squeeze(0))
    else:
        interaction_energy = torch.einsum('bii,b->b', two_rdms_kron_pauli_z, edge_params.squeeze())
    

    unique_batches, batch_energy_node_idx = torch.unique(batch_index, return_inverse=True)
    batch_local_energies = torch.zeros_like(unique_batches, dtype=torch.float64)

    batch_local_energies.scatter_add_(0, batch_energy_node_idx, local_energy)

    # Now we need to add the interaction energy to the correct batch, so we need to find the correct batch for each edge
    batch_interaction_energies = torch.zeros_like(batch_local_energies, dtype=torch.float64)
    # First we need to find the batch for each node in the edge
    batch_edge_index = batch_index[edge_index][0, :]
    # Now we need to find the unique batch for each edge, we know that no graph will connect edges between two batches
    unique_batches, batch_edge_idx = torch.unique(batch_edge_index, return_inverse=True)

    batch_interaction_energies.scatter_add_(0, batch_edge_idx, interaction_energy)

    batch_energies = batch_local_energies + batch_interaction_energies

    #Sort batch ene
    

    return batch_energies, unique_batches


if __name__ == "__main__":
    # Example usage:
    data_file = os.path.dirname(os.path.abspath(__file__)) + "/data/MPS_10x1_N2000_PBCFalse.pt"
    # Load the dataset
    dataset = IsingModelDataset.load(data_file)
    # Access data points from the loaded dataset
    rand_idx = int(torch.randint(len(dataset), (1,)))
    data_point = dataset[rand_idx]
    print(data_point)
    print(data_point.y_node_rdms[0])
    print(data_point.y_edge_rdms[0])
    
    batch = torch.zeros_like(torch.Tensor(data_point.x_nodes.shape[0]), dtype=torch.long)
    energy, _ = calculate_energy_batched(data_point.y_node_rdms, data_point.y_edge_rdms, data_point.x_nodes, data_point.x_edges, batch, data_point.edge_index)
    energy2 = calculate_energy_single_graph(data_point.y_node_rdms, data_point.y_edge_rdms, data_point.x_nodes, data_point.x_edges)
    #Compare with real energy:
    print("Energy from RDMs (single graph):", energy2)
    print("Energy from RDMs (batched):", energy)
    print("Energy from data:", data_point.y_energy)


    #Lets calculate the energy for each datapoint, and plot the results vs the real energy
    energies = []
    real_energies = []
    for i in range(len(dataset)):
        print("Calculating energy for data point %d" % i)
        data_point = dataset[i]
        batch = torch.zeros_like(torch.Tensor(data_point.x_nodes.shape[0]), dtype=torch.long)
        energy, _ = calculate_energy_batched(data_point.y_node_rdms, data_point.y_edge_rdms, data_point.x_nodes, data_point.x_edges,batch, data_point.edge_index)
        energies.append(energy[0])
        real_energies.append(data_point.y_energy)
    # We want the plot to show the error in the energy, so we subtract the real energy from the calculated energy
    energies = np.array(energies) - np.array(real_energies)
    # As x axis, we want the real energy
    real_energies = np.array(real_energies)
    # Plot the results, first sort the points by real energy
    sorted_idx = np.argsort(real_energies)
    real_energies = real_energies[sorted_idx]
    energies = energies[sorted_idx]

    #Plot as percentage error
    energies = energies / real_energies
    
    import matplotlib.pyplot as plt
    #Scatter plot but make the points smaller so we can see the distribution
    plt.scatter(real_energies, energies, s=0.1)
    
    plt.xlabel("Real energy")
    plt.ylabel("Error in energy")
    plt.show()




    
    
    


