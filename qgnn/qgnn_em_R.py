import torch
import torch.nn as nn
from lib.utils import create_random_rdm_torch_batched
from lib.agg import aggregate_messages_by_product_optimized as unitary_aggregation
import sys
sys.path.append("..")
from lib.rdm import calculate_energy_batched


#   - QGNN_Mixing_Layer:
#       - This class is used to update the eigenvalues of the 1-RDMs
#       - It takes as input the 1-RDMs/2-RDMs, the node parameters, and the edge parameters
#       - It outputs the updated 1-RDMs

#TODO: Alternative to use Bloch Vector Representation instead of RDM Flattening for NNs



class QGNN_Mixing_Layer(nn.Module):
    def __init__(self, node_channels_in, message_dim, node_attribute_size = 2, edge_attribute_size = 1, use_phase=False, use_complex=False):
        super().__init__()
        self.use_complex = use_complex
        # Define the network for eigenvalue updates
        self.node_channels_in = node_channels_in
        self.node_channels_out = message_dim
        self.use_phase = use_phase

        self.node_channels_prod = torch.prod(torch.tensor(node_channels_in))
        self.node_channels_sum = torch.sum(torch.tensor(node_channels_in))

        if self.use_complex:
            message_out_dim = message_dim
        else:
            message_out_dim = node_channels_in[0]*2

        self.phi_message_Real= nn.Sequential(
            nn.Linear(self.node_channels_prod*2 + node_attribute_size*2 + edge_attribute_size*2, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, message_out_dim),
            nn.SiLU(),
        )
        
        if self.use_complex:
            self.phi_message_Imag= nn.Sequential(
                nn.Linear(self.node_channels_prod*2 + node_attribute_size*2 + edge_attribute_size*2, message_dim),
                nn.SiLU(),
                nn.Linear(message_dim, message_dim),
                nn.SiLU(),
            )

            self.phi_message_Merge = nn.Sequential(
                nn.Linear(2*message_dim, message_dim),
                nn.SiLU(),
                nn.Linear(message_dim, node_channels_in[0]*2),
                nn.SiLU(),
            )

        
        self.phi_eig = nn.Sequential(
            nn.Linear(self.node_channels_sum + node_channels_in[0] * 2 + (0 if not self.use_phase else self.node_channels_prod), message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, node_channels_in[0]),
            nn.Softmax(dim=-1) #Softmax so produced eigenvalues sum to 1
        )

    
    def forward(self, h_i, edge_index, node_params, edge_params):
        if len(edge_index) > 0:
            sender, receiver = edge_index
        else:
            #If the input graph is composed of two nodes, when predicting two RDMs on the transformed graph, the edge_index will be empty
            sender = receiver = torch.zeros_like(torch.tensor([0], device=h_i.device))

        # Flatten h_i for sender and receiver
        h_i_flat_sender = h_i[sender].view(sender.size(0), -1)  
        h_i_flat_receiver = h_i[receiver].view(receiver.size(0), -1)

        if edge_params is not None:
            num_edges = h_i_flat_sender.shape[0]

            if num_edges > 1:
                edge_params_reshaped = edge_params[edge_index].view(num_edges, -1)
            else:
                edge_params_reshaped = edge_params.repeat(1, 2)
                
            if self.use_complex:
                concat_tensors_real = torch.cat((h_i_flat_sender.real, h_i_flat_receiver.real, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1)
                concat_tensors_imag = torch.cat((h_i_flat_sender.imag, h_i_flat_receiver.imag, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1)
            else: 
                concat_tensors = torch.cat((h_i_flat_sender, h_i_flat_receiver, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1)
        else:
            if self.use_complex:
                concat_tensors_real = torch.cat((h_i_flat_sender.real, h_i_flat_receiver.real, node_params[sender], node_params[receiver]), dim=1)
                concat_tensors_imag = torch.cat((h_i_flat_sender.imag, h_i_flat_receiver.imag, node_params[sender], node_params[receiver]), dim=1)
            else:
                concat_tensors = torch.cat((h_i_flat_sender, h_i_flat_receiver, node_params[sender], node_params[receiver]), dim=1)

        if self.use_complex:
            #Create I_ij complex and real concatenating real or complex part of the features
            A_ij_real = self.phi_message_Real(concat_tensors_real)
            A_ij_imag = self.phi_message_Imag(concat_tensors_imag)

            I_ij = self.phi_message_Merge(torch.cat((A_ij_real, A_ij_imag), dim=1))
        else:
            I_ij = self.phi_message_Real(concat_tensors)


        # Aggregate messages at each node
        num_nodes = node_params.size(0)
        aggregated_messages = torch.zeros(num_nodes, I_ij.shape[1], dtype=I_ij.dtype, device=I_ij.device) 
        receiver_expanded = receiver.unsqueeze(1).repeat(1, I_ij.shape[1])
        aggregated_messages.scatter_add_(0, receiver_expanded, I_ij)

        # Eigenvalue Decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(h_i)
        
        if h_i.dtype == torch.complex64 or h_i.dtype == torch.complex128:
            eigenvectors = eigenvectors.detach() #Detach eigenvectors from the computation graph to avoid phase issues

            eigenvector_magnitudes = torch.norm(eigenvectors, dim=-1).requires_grad_(True)
        else:
            eigenvector_magnitudes = torch.norm(eigenvectors, dim=-1)

        if self.use_phase and self.use_complex:
            eigenvector_phases = torch.angle(eigenvectors).flatten(start_dim=1).requires_grad_(True)
            combined_features = torch.cat([aggregated_messages, eigenvalues, eigenvector_magnitudes, eigenvector_phases], dim=1)
        else:
            combined_features = torch.cat([aggregated_messages, eigenvalues, eigenvector_magnitudes], dim=1)


        lambda_updated = self.phi_eig(combined_features)

        # Update the eigenvalues using aggregated messages and eigenvalues
        lambda_updated = lambda_updated.to(eigenvectors.dtype)

        h_i_updated = torch.bmm(eigenvectors, torch.bmm(torch.diag_embed(lambda_updated), eigenvectors.conj().transpose(-2, -1)))

        return h_i_updated
                                                                                                            
    

class QGNN_Rot2_PermInv_Layer(nn.Module):
    def __init__(self, node_channels, hidden_dim, node_attribute_size = 2, edge_attribute_size = 1, use_complex=False):
        super().__init__()
        self.flat_node_channels_in = torch.prod(torch.tensor(node_channels))
        self.use_complex = use_complex
        # Network to output three Euler angles
        self.phi_angles = nn.Sequential(
            nn.Linear(2*self.flat_node_channels_in, 3),  # Output three angles for Euler rotation
            nn.Tanh() # To ensure that the angles are in the range [-pi, pi]
        )

        self.phi_I_Real = nn.Sequential(
            nn.Linear(self.flat_node_channels_in*2 + node_attribute_size*2 + edge_attribute_size*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.flat_node_channels_in)
        )

        if self.use_complex:
            self.phi_I_Imag = nn.Sequential(
                nn.Linear(self.flat_node_channels_in*2 + node_attribute_size*2 + edge_attribute_size*2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.flat_node_channels_in)
            )

    def forward(self, h_i, edge_index, node_params, edge_params):
        sender, receiver = edge_index

        # Flatten h_i for sender and receiver
        h_i_flat_sender = h_i[sender].view(sender.size(0), -1)  
        h_i_flat_receiver = h_i[receiver].view(receiver.size(0), -1)

        num_edges = h_i_flat_sender.shape[0]
        if num_edges > 1:
            edge_params_reshaped = edge_params[edge_index].view(num_edges, -1)
        else:
            edge_params_reshaped = edge_params.repeat(1, 2)

        if self.use_complex:
            #Create I_ij complex and real concatenating real or complex part of the features
            A_ij_real = self.phi_I_Real(torch.cat((h_i_flat_sender.real, h_i_flat_receiver.real, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1))
            A_ij_imag = self.phi_I_Imag(torch.cat((h_i_flat_sender.imag, h_i_flat_receiver.imag, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1))

        else:
            A_ij = self.phi_I_Real(torch.cat((h_i_flat_sender, h_i_flat_receiver, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1))

        # Define Pauli matrices
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64).to(h_i.device)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64).to(h_i.device)

        if self.use_complex:
            # Obtain the rotation angles
            angles = self.phi_angles(torch.cat((A_ij_imag, A_ij_real), dim=1))
        else:
            angles = self.phi_angles(A_ij)
        m_ij = angles * torch.tensor([torch.pi, torch.pi, torch.pi], dtype=torch.float32, device=angles.device) #Message angles are in the range [-pi, pi]

        # Aggregate messages using scatter add
        num_nodes = node_params.size(0)
        # Create receiver_expanded with shape [num_edges, num_nodes]
        receiver_expanded = receiver.unsqueeze(1).expand(-1, 3)  # Expand along the second dimension

        # Initialize aggregated_messages with the correct shape
        aggregated_messages = torch.zeros(num_nodes, 3, dtype=m_ij.dtype, device=m_ij.device)

        # Use scatter_add to accumulate messages
        aggregated_messages.scatter_add_(0, receiver_expanded, m_ij)

        #Scale angles between [-pi, pi]
        aggregated_messages = (aggregated_messages + torch.pi) % (2 * torch.pi) - torch.pi

        #Convert angles to unitaries
        aggregated_unitaries = torch.matrix_exp(-1j * aggregated_messages[:, 0].unsqueeze(-1).unsqueeze(-1) * sigma_z / 2) @ \
                            torch.matrix_exp(-1j * aggregated_messages[:, 1].unsqueeze(-1).unsqueeze(-1) * sigma_y / 2) @ \
                            torch.matrix_exp(-1j * aggregated_messages[:, 2].unsqueeze(-1).unsqueeze(-1) * sigma_z / 2)


        aggregated_unitaries_dagger = aggregated_unitaries.transpose(-2, -1).conj()  # U^dagger for each matrix in the batch        

        identity =torch.eye(aggregated_unitaries.shape[-1], dtype=aggregated_unitaries.dtype, device=aggregated_unitaries.device) \
            .unsqueeze(0) \
            .repeat(aggregated_unitaries.shape[0], 1, 1)
        
        is_unitary = torch.allclose(torch.bmm(aggregated_unitaries, aggregated_unitaries_dagger), identity)
        assert is_unitary, "Aggregated messages are not unitary"

        h_i_updated = torch.bmm(aggregated_unitaries, torch.bmm(h_i, aggregated_unitaries_dagger))
        
        return h_i_updated
    

class QGNN_Rot_PermInv_Layer(nn.Module):
    def __init__(self, node_channels_in, hidden_dim, node_attribute_size = 2, edge_attribute_size = 1, use_complex=False):
        super().__init__()
        self.node_channels_in = node_channels_in
        self.use_complex = use_complex
        self.flat_node_channels_in = torch.prod(torch.tensor(node_channels_in))

        self.phi_I_Real = nn.Sequential(
            nn.Linear(self.flat_node_channels_in*2 + node_attribute_size*2 + edge_attribute_size*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.flat_node_channels_in)
        )

        if self.use_complex:
            self.phi_I_Imag = nn.Sequential(
                nn.Linear(self.flat_node_channels_in*2 + node_attribute_size*2 + edge_attribute_size*2, hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.flat_node_channels_in)
            )
        

    def forward(self, h_i, edge_index, node_params, edge_params):
        if len(edge_index) > 0:
            sender, receiver = edge_index
        else:
            #If the input graph is composed of two nodes, when predicting two RDMs on the transformed graph, the edge_index will be empty
            sender = receiver = torch.zeros_like(torch.tensor([0], device=h_i.device))

        # h_i.shape  = [batch_size*num_nodes, 2, 2] or [batch_size*num_nodes, 4, 4]

        # Flatten h_i for sender and receiver
        h_i_flat_sender = h_i[sender].view(sender.size(0), -1)  
        h_i_flat_receiver = h_i[receiver].view(receiver.size(0), -1)

        

        if edge_params is not None:
            num_edges = h_i_flat_sender.shape[0]

            if num_edges > 1:
                edge_params_reshaped = edge_params[edge_index].view(num_edges, -1)
            else:
                edge_params_reshaped = edge_params.repeat(1, 2)
                
            if self.use_complex:
                concat_tensors_real = torch.cat((h_i_flat_sender.real, h_i_flat_receiver.real, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1)
                concat_tensors_imag = torch.cat((h_i_flat_sender.imag, h_i_flat_receiver.imag, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1)
            else: 
                concat_tensors_real = torch.cat((h_i_flat_sender, h_i_flat_receiver, node_params[sender], node_params[receiver], edge_params_reshaped), dim=1)
        else:
            if self.use_complex:
                concat_tensors_real = torch.cat((h_i_flat_sender.real, h_i_flat_receiver.real, node_params[sender], node_params[receiver]), dim=1)
                concat_tensors_imag = torch.cat((h_i_flat_sender.imag, h_i_flat_receiver.imag, node_params[sender], node_params[receiver]), dim=1)
            else:
                concat_tensors_real = torch.cat((h_i_flat_sender, h_i_flat_receiver, node_params[sender], node_params[receiver]), dim=1)

        if self.use_complex:
            #Create I_ij complex and real concatenating real or complex part of the features
            A_ij_real = self.phi_I_Real(concat_tensors_real)
            A_ij_imag = self.phi_I_Imag(concat_tensors_imag)

        
            A_ij_real = A_ij_real.view(-1, h_i.size(1), h_i.size(2))
            A_ij_imag = A_ij_imag.view(-1, h_i.size(1), h_i.size(2))

            #Arbitrary Complex Matrix Messages
            m_ij = A_ij_real + 1j * A_ij_imag #Shape = [M, 2, 2] or [M, 4, 4], M > N, N is number of nodes
        else:
            A_ij_real = self.phi_I_Real(concat_tensors_real)
            A_ij_real = A_ij_real.view(-1, h_i.size(1), h_i.size(2))
            m_ij = A_ij_real

        #Flatten messages and aggregate
        m_ij_flat = m_ij.view(m_ij.size(0), -1)
        aggregated_messages = torch.zeros(h_i.size(0), m_ij_flat.size(1), dtype=m_ij_flat.dtype, device=m_ij_flat.device)
        receiver_expanded = receiver.unsqueeze(1).repeat(1, m_ij_flat.size(1))
        aggregated_messages.scatter_add_(0, receiver_expanded, m_ij_flat)

        #Reshape to original shape
        aggregated_messages = aggregated_messages.view(-1, h_i.size(1), h_i.size(2))

        #Perform average by dividing by number of messages per receiver
        num_messages_per_receiver = torch.bincount(receiver, minlength=h_i.size(0))
        #Replace 0 with 1 to avoid division by zero
        num_messages_per_receiver[num_messages_per_receiver == 0] = 1
        aggregated_messages = aggregated_messages / num_messages_per_receiver.unsqueeze(-1).unsqueeze(-1).to(aggregated_messages.dtype)


        abs_sum = torch.sum(torch.abs(aggregated_messages), dim=(1, 2))

        # Replace zero matrices with identity matrix
        zero_matrices_mask = abs_sum == 0
        identity_matrix = torch.eye(aggregated_messages.shape[1], dtype=aggregated_messages.dtype, device=aggregated_messages.device)
        aggregated_messages[zero_matrices_mask] = identity_matrix

        #Convert messages to unitaries
        aggregated_messages_skew_herm = 0.5 * (aggregated_messages - aggregated_messages.conj().transpose(-2, -1))
        aggregated_messages_unitary = torch.matrix_exp(aggregated_messages_skew_herm)
        aggregated_messages_unitary_dagger = aggregated_messages_unitary.transpose(-2, -1).conj()  # U^dagger for each matrix in the batch
       
        # identity =torch.eye(aggregated_messages_unitary.shape[-1], dtype=aggregated_messages_unitary.dtype, device=aggregated_messages.device) \
        #     .unsqueeze(0) \
        #     .repeat(aggregated_messages.shape[0], 1, 1)
        
        # is_unitary = torch.allclose(torch.bmm(aggregated_messages_unitary, aggregated_messages_unitary_dagger), identity)
        # assert is_unitary, "Aggregated messages are not unitary"

        h_i_updated = torch.bmm(aggregated_messages_unitary, torch.bmm(h_i, aggregated_messages_unitary_dagger))
        
        return h_i_updated
    


class QGNN_EM(nn.Module):
    def __init__(self, one_rdm_dims, one_rdm_hidden_rot_dim, one_rdm_hidden_mixing_dim,
                 two_rdm_dims, two_rdm_hidden_rot_dim, two_rdm_hidden_mixing_dim,
                 num_layers, device, **kwargs):
        super().__init__()
        # Initialize layers
        self.use_simplified = kwargs['kwargs'].get("use_simplified", False)
        self.complex_init = kwargs['kwargs'].get("complex_init", False)
        self.calculate_energy = kwargs['kwargs'].get("calculate_energy", True)
        self.use_eigenvector_phase = kwargs['kwargs'].get("use_eigenvector_phase", False)
        self.device = device
        self.rdm_dataset = kwargs['kwargs'].get("rdm_dataset", False)
        self.use_complex = False

        self.one_rdm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.one_rdm_layers.append(QGNN_Mixing_Layer(one_rdm_dims, one_rdm_hidden_mixing_dim, use_phase=self.use_eigenvector_phase, use_complex=self.use_complex))

            if self.use_simplified:
                self.one_rdm_layers.append(QGNN_Rot2_PermInv_Layer(one_rdm_dims, one_rdm_hidden_rot_dim, use_complex=self.use_complex))
            else:
                self.one_rdm_layers.append(QGNN_Rot_PermInv_Layer(one_rdm_dims, one_rdm_hidden_rot_dim, use_complex=self.use_complex))

        

        self.two_rdm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.two_rdm_layers.append(QGNN_Mixing_Layer(two_rdm_dims, two_rdm_hidden_mixing_dim, node_attribute_size=5, edge_attribute_size=0, use_phase=self.use_eigenvector_phase, use_complex=self.use_complex))
        
            self.two_rdm_layers.append(QGNN_Rot_PermInv_Layer(two_rdm_dims, two_rdm_hidden_rot_dim, node_attribute_size=5, edge_attribute_size=0, use_complex=self.use_complex))


        # self.global_readout = nn.Sequential(
        #     nn.Linear(self.node_out_channels, self.node_channels),
        #     nn.SiLU(),
        #     nn.Linear(self.node_channels, self.global_out_channels)
        # )



    def transform_graph(self, _x_nodes, _x_edges, edge_index):
        num_edges = edge_index.size(1)

        # Step 1: Create new node features
        new_node_features_list = [
            torch.cat([_x_nodes[edge_index[0, i]], _x_nodes[edge_index[1, i]], _x_edges[i]])
            for i in range(num_edges)
        ]
        new_node_features = torch.stack(new_node_features_list).to(self.device)

        # Step 2: Efficiently determine new edges
        adjacency_list = {}
        for i in range(num_edges):
            node_a, node_b = edge_index[:, i]
            adjacency_list.setdefault(node_a.item(), []).append(i)
            adjacency_list.setdefault(node_b.item(), []).append(i)

        new_edges_set = set()
        for connected_edges in adjacency_list.values():
            for i in range(len(connected_edges)):
                for j in range(i + 1, len(connected_edges)):
                    edge_pair = tuple(sorted([connected_edges[i], connected_edges[j]]))
                    new_edges_set.add(edge_pair)

        new_edges = torch.tensor(list(new_edges_set)).t().to(self.device)


        return new_node_features, new_edges



    def forward(self, data):
        data = data.to(self.device)
        _x_nodes, _x_edges, edge_index, batch = data.x_nodes, data.x_edges, data.edge_index, data.batch

        tf_node_features_2, tf_edges_2 = self.transform_graph(_x_nodes, _x_edges, edge_index)

        if self.rdm_dataset:
            one_rdms = data.x_node_rdms.to(self.device)
            two_rdms = data.x_edge_rdms.to(self.device)
        else:
            one_rdms = create_random_rdm_torch_batched(2, _x_nodes.size(0), complex=self.complex_init).to(self.device)
            two_rdms = create_random_rdm_torch_batched(4, _x_edges.size(0), complex=self.complex_init).to(self.device)
        
        if torch.any(torch.isnan(_x_nodes)):
            raise ValueError(f"x_nodes contains nan: {_x_nodes}")
        
        for layer in self.one_rdm_layers:
            one_rdms = layer(one_rdms, edge_index, _x_nodes, _x_edges)
        
        for layer in self.two_rdm_layers:
            two_rdms = layer(two_rdms, tf_edges_2, tf_node_features_2, None)

        if self.calculate_energy:
            #We calculate the ground state energy from the 1-RDMs, 2-RDMs and Hamiltonian parameters
            x_global, batch_idx = calculate_energy_batched(one_rdms, two_rdms, _x_nodes, _x_edges, batch, edge_index)
        else:
            raise NotImplementedError("Energy prediction not implemented yet")


        if torch.any(torch.isnan(one_rdms)) or torch.any(torch.isnan(two_rdms)) or torch.any(torch.isnan(x_global)):
            raise ValueError(f"Output contains nan: one_rdms={one_rdms}, two_rdms={two_rdms}, x_global={x_global}")

        return torch.squeeze(one_rdms), torch.squeeze(two_rdms), torch.squeeze(x_global)