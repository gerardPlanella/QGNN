from typing import Iterator
import torch
import torch.nn as nn
from itertools import combinations
import quimb.tensor as qtn
from tqdm import tqdm
from collections import defaultdict
from ordered_set import OrderedSet
import quimb as qu
from qbp_gnn.tensor_network import tn_tensors_to_torch, build_dual_tn, unbuild_dual_tn, generate_tensor_network, visualize_tensor_network, dynamic_tensor_mult, compute_partial_trace
from qbp_gnn.tensor_network import contract_physical_indices as contract_phys_ind

import sys
sys.path.append(".")
from lib.rdm import calculate_energy_batched
from dataset.ising.isingModel import IsingModelDataset
from baselines.tensorNetworks import TensorNetworkRunner, DMRG_QUIMB, SimpleUpdate
from lib.utils import parse_hamiltonian


def matrix_sqrt(A, dtype=torch.cfloat):
    """Compute the square root of a positive semi-definite matrix, assuming A is Hermitian."""
    vals, vecs = torch.linalg.eigh(A)
    sqrt_vals = torch.sqrt(torch.clamp(vals, min=0))  # Ensure eigenvalues are non-negative
    sqrt_vals_complex = torch.diag_embed(sqrt_vals).to(dtype=dtype)  # Use complex type
    return vecs @ sqrt_vals_complex @ vecs.conj().transpose(-2, -1)

def fidelity_torch(rho, sigma, dtype=torch.cfloat):
    """Calculate the fidelity between two density matrices rho and sigma."""
    # Compute the square root of rho
    sqrt_rho = matrix_sqrt(rho, dtype=dtype)
    
    # Compute the product sqrt_rho * sigma * sqrt_rho
    middle_product = sqrt_rho @ sigma @ sqrt_rho
    
    # Compute the square root of the middle product
    sqrt_middle_product = matrix_sqrt(middle_product, dtype=dtype)
    
    # Compute the trace of the sqrt_middle_product and then square the result for fidelity
    trace_value = torch.trace(sqrt_middle_product)
    
    # Return the square of the trace, since fidelity is the square of the trace of the square root of the middle product
    fid = trace_value.real**2
    return torch.clip(fid, torch.zeros_like(fid), torch.ones_like(fid))

class PSD_MessageModule(nn.Module):
    def __init__(self, z_dim, bond_dim, hidden_dim, dtype=torch.float32, psd_check_tol = 1e-4, device='cpu'):
        super(PSD_MessageModule, self).__init__()
        self.messages = defaultdict(list)
        self.dtype = dtype
        self.edge_index = []
        self.int_to_node = {}
        self.node_to_int = {}
        self.inds = {}
        self.psd_check_tol = psd_check_tol
        self.updated_messages = {}
        self.device = device
        self.z_dim = z_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim

        self.psi_A = nn.Sequential(
            nn.Linear(z_dim*2, hidden_dim, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, (bond_dim * (bond_dim + 1)) // 2, dtype=dtype)
        )

    
    def get_message(self, key):
        messages = self.messages.get(key)
        message = messages[-1] if messages else None
        return message, self.inds.get(key)

    def check_message(self, message=None, key=None):
        with torch.no_grad():
            if message is None:  
                matrix, _ = self.get_message(key)
            else:
                matrix = message
            #Check for inf or NaN values
            if torch.isinf(matrix).any() or torch.isnan(matrix).any():
                print(f"Message {key if key is not None else ''} has inf or NaN values")
                return False
            eigenvalues = torch.linalg.eigvalsh(matrix)
            ispsd = torch.all(eigenvalues >= -self.psd_check_tol)

            if not ispsd:
                print(f"Message {key if key is not None else ''} is not positive semi-definite")
                print(f"Eigenvalues: {eigenvalues}")
                print(f"Message: {matrix} ")
                
        return ispsd 

    
    def initialise_message(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=-1)
        A_ij = self.psi_A(z)
        A_ij = A_ij.view(-1, self.bond_dim, self.bond_dim)
        A_ij = 0.5 * (A_ij + A_ij.transpose(-2, -1).conj())
        #Normalize the message
        A_ij = A_ij / torch.trace(A_ij)

        return A_ij

    def update_message(self, key, message, inds=None, commit = False):
        if key not in self.messages:
            raise ValueError(f"Message {key} does not exist")
        else:
            if commit:
                self.add_message(key, message, inds=inds, recalculate_edge_index=False)
            else:
                self.updated_messages[key] = message
    
    def commit_updates(self):
        #Can only commit if all messages have been updated, convert keys to set to check if all keys in updated_messages are in messages
        assert len(set(self.updated_messages.keys()).difference(set(self.messages.keys()))) == 0, "Not all messages have been updated"
        message_keys = list(self.updated_messages.keys())
        for key in message_keys:
            self.messages[key].append(self.updated_messages.get(key))
            del self.updated_messages[key]

        assert len(self.updated_messages) == 0, "Not all messages have been committed"
                  
    def add_message(self, key, message, recalculate_edge_index=False, inds=None):
        self.messages[key].append(message)
        if recalculate_edge_index:
            self.edge_index, self.int_to_node, self.node_to_int = self.get_edge_index()
        if inds is not None:
            self.inds[key] = inds

    def __len__(self):
        return len(self.messages)


    def clear(self):
        #Function to empty the lists in messages
        self.messages.clear()
        self.inds.clear()
        self.int_to_node.clear()
        self.node_to_int.clear()
        self.updated_messages.clear()

    def attach_to_computation(self):
        for key in self.messages.keys():
            for message in self.messages[key]:
                self.messages[key][message] = nn.Parameter(self.messages[key][message], requires_grad=False)

    def initialise_structure(self, tn, contracted = False):
        # Add a message for each edge in the TN in both directions
        # The key will be given by the tags of the tensors at the ends of the edge "tag1->tag2"
        tn_copy = tn.copy()
        
        if not contracted:
            physical_indices = OrderedSet([ind for ind in tn_copy.ind_map if ind.startswith('k')])

            for phys_ind in physical_indices:
                tn_copy.contract_ind(phys_ind)

        # Iterate over virtual legs to obtain edges
        edges = []
        for ind in tn_copy.ind_map:
            connected_tensors = list(tn_copy.ind_map[ind])
            ind_edges = [(a, b) for a, b in combinations(connected_tensors, 2)] \
                    + [(b, a) for a, b in combinations(connected_tensors, 2)]
            edges.extend(ind_edges) 
        
        # Remove duplicates
        edges = list(OrderedSet(edges))

        #Find the message keys
        for tidx1, tidx2 in edges:
            ttag1 = tn_copy.tensor_map[tidx1].tags
            ttag2 = tn_copy.tensor_map[tidx2].tags
            message_inds = OrderedSet()

            if not contracted:
                #We save the original indices before contracting the physical indices   
                ttag_names = []
                tensor_id_pair = []
                conj_tensor_id_pair = []
                for ttag in [ttag1, ttag2]:
                    ttag_name = ""
                    conj_node_tags = [item for item in ttag if item.endswith('*')]
                    node_tags = [item for item in ttag if not item.endswith('*')]
                    node_tid = tn._get_tids_from_tags(node_tags, which='all')
                    conj_node_tid = tn._get_tids_from_tags(conj_node_tags, which='all')
                    assert len(node_tid) == 1, "More than one node found when contracting physical indices"
                    assert len(conj_node_tid) == 1, "More than one node found when contracting physical indices"
                   
                    tids_conv = node_tid.union(conj_node_tid)
                    num_inds = len(tids_conv)
                    assert num_inds <= 2, "More than two nodes contracted when contracting physical indices"
                    for i in range(num_inds): 
                        id = tids_conv.popleft()
                        ttag_name += str(id)
                        if i < num_inds - 1:
                            tensor_id_pair.append(id)
                            ttag_name += "_"
                        else:
                            conj_tensor_id_pair.append(id)
                    ttag_names.append(ttag_name)

                assert len(ttag_names) == 2, "The tensor tags are not of length 2"

                key = f"{ttag_names[0]}->{ttag_names[1]}"

                #Save the inds that connect the tensors, from the first tensor to the second
                common_inds = OrderedSet(tn.tensor_map[tensor_id_pair[0]].inds).intersection(tn.tensor_map[tensor_id_pair[1]].inds)
                common_inds_conj = OrderedSet(tn.tensor_map[conj_tensor_id_pair[0]].inds).intersection(tn.tensor_map[conj_tensor_id_pair[1]].inds)  
                message_inds = common_inds.union(common_inds_conj).difference(OrderedSet(physical_indices))             
            else:
                key = f"{tidx1}->{tidx2}"
                message_inds = OrderedSet(tn.tensor_map[tidx1].inds).intersection(OrderedSet(tn.tensor_map[tidx2].inds))
                
            #Make a tuple of the dimension of the tensor that has each message_ind
            message_shape = []
            for ind in message_inds:
                message_shape.append(tn_copy.ind_size(ind))

            #Convert to tuple
            message_shape = tuple(message_shape)

            assert torch.prod(torch.tensor(message_shape)) == self.bond_dim**2, f"Message {key} has incorrect shape"

            message = None
            self.add_message(key, message, inds=message_inds)
        
        self.updated_messages.clear()
        # Recalculate the edge index in the end
        self.edge_index, self.int_to_node, self.node_to_int = self.get_edge_index()
    
    def initialize_messages(self, state):
        self.messages.clear()
        #Initialization of the messages through Cholesky decomposition
        L_flat = self.psi_A(state)
        # Initialize a zero matrix for each sample in the batch, with dimensions (bond_dim x bond_dim)
        # This matrix will hold the lower triangular entries.
        L = torch.zeros((L_flat.shape[0], self.bond_dim, self.bond_dim), dtype=L_flat.dtype, device=L_flat.device)
        # Generate the indices for the lower triangular part of the matrix
        # Offset = 0 means the diagonal and below it
        indices = torch.tril_indices(row=self.bond_dim, col=self.bond_dim, offset=0, device=L.device)
        # Assign the elements of the flat output from the neural network to the lower triangular part of L
        # L_flat contains the entries for the lower triangular matrix including the diagonal, shaped appropriately.
        L[:, indices[0], indices[1]] = L_flat
        # Compute the resulting PSD matrix As using batch matrix multiplication.
        # L * L^T ensures the matrix is symmetric and positive semi-definite.
        # L.transpose(-2, -1) computes the transpose of the last two dimensions of L,
        # effectively transposing each matrix in the batch.
        As = torch.bmm(L, L.transpose(-2, -1))

        for i in range(self.edge_index.size(1)):
            int_node1 = self.edge_index[0, i].item()
            int_node2 = self.edge_index[1, i].item()
            node1 = self.int_to_node[int_node1]
            node2 = self.int_to_node[int_node2]
            key = f"{node1}->{node2}"
            self.add_message(key, As[i], recalculate_edge_index=False)
            assert self.check_message(key=key), f"Message {key} is not PSD"
            #Set reverse message to the same value
            key = f"{node2}->{node1}"
            self.add_message(key, As[i].transpose(-2, -1), recalculate_edge_index=False)
            

    def iterate_messages(self, shuffle=True):
        keys = list(self.messages.keys())
        if shuffle:
            indices = torch.randperm(len(keys))
        else:
            indices = torch.arange(len(keys))

        for idx in indices:
            key = keys[idx.item()]
            yield key, self.get_message(key) 

    def get_edge_index(self):
        keys = list(self.messages.keys())
        nodes = list(OrderedSet(key.split("->")[0] for key in keys) | OrderedSet(key.split("->")[1] for key in keys))

        def sort_key(node_label):
            parts = node_label.split('_')
            if len(parts) == 1:
                # Only one part, no underscore, treat it as (x, 0)
                return (int(parts[0]), 0)
            elif len(parts) == 2:
                # Two parts, format x_y, treat it as (x, y)
                return (int(parts[0]), int(parts[1]))
            else:
                # Unexpected format, raise an error
                raise ValueError("Node label format is incorrect")

        # Sort nodes based on the custom key
        nodes_sorted = sorted(nodes, key=sort_key)

        #Check that the key is the same as the node before _ if present
        for node in nodes_sorted:
            parts = node.split("_")
            if len(parts) == 2:
                assert parts[0] == node.split("_")[0], f"Node {node} has incorrect format"
            elif len(parts) == 1:
                assert parts[0] == node, f"Node {node} has incorrect format"
            else:
                raise ValueError(f"Node {node} has incorrect format")


        # Create a mapping from the unique keys to integers
        node_to_int = {node: i for i, node in enumerate(nodes_sorted)}
        int_to_node = {i: node for i, node in enumerate(nodes_sorted)}

        # Convert the keys to integers using the mapping
        senders = [node_to_int[key.split("->")[0]] for key in keys]
        receivers = [node_to_int[key.split("->")[1]] for key in keys]

        # Create edge index tensor
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)

        return edge_index, int_to_node, node_to_int
    
    def get_neighbours(self, node_name, exclude_nodes=[]):
        node_idx = self.node_to_int.get(node_name)
        assert node_idx is not None, f"Node {node_name} not found"
        senders = self.edge_index[0]
        receivers = self.edge_index[1]

        exclude_nodes = [self.node_to_int[n] for n in exclude_nodes]

        #Find the neighbours of the node
        neighbours = receivers[senders == node_idx]
        if len(exclude_nodes) > 0:
            neighbours = [n for n in neighbours if n not in exclude_nodes]

        # Convert the neighbours to node names
        neighbour_names = [self.int_to_node[int(n)] for n in neighbours]

        return neighbour_names
    
class GenericTNModel(torch.nn.Module):
    def __init__(self, tensor_dtype=torch.cfloat, contract_physical_indices=True, device='cpu', num_layer=None):
        super(GenericTNModel, self).__init__()
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.contract_physical_indices = contract_physical_indices  # Whether to contract physical indices in the TN

    def build_dual(self, tn, tn_type):
        # Convert the TN to a dual TN
        tn_dual = build_dual_tn(tn, contract_phys_ind = self.contract_physical_indices)
        tn_dual = tn_dual.equalize_norms(1.0)
        params, skeleton = qtn.pack(tn_dual)
        requires_grad = [not any('*' in str(tag) for tag in list(skeleton.tensor_map[int(idx)].tags)) for idx in range(len(skeleton.tensor_map))]
        names = ["dual_tn_tensor_" if not requires_grad[idx] else "primal_tn_tensor_" for idx in range(len(skeleton.tensor_map))]
        tensor_list = [nn.Parameter(val.contiguous(), requires_grad=requires_grad[idx]) for idx, val in enumerate(params.values())]
        # tensor_list = [nn.Parameter(val, requires_grad=True) for val in params.values()]
        for idx, param in enumerate(tensor_list):
            self.register_parameter(names[idx]+str(idx), param)
        tensors = nn.ParameterList(tensor_list)
        tensors = tensors.to(self.device)

        self.tensors = tensors
    
        return tensors, skeleton, tn_type
    
    def unpack(self, tn):
        torch_params, skeleton, _ = tn
        # Convert the parameters back to a tensor network inside the forward pass
        params_dict = {idx: p.data for idx, p in enumerate(torch_params)}
        tn = qtn.unpack(params_dict, skeleton)
        return tn

    def unbuild_dual(self, dual_tn):
        _, _, tn_type = dual_tn
        dual_tn = self.unpack(dual_tn)

        if not self.contract_physical_indices:
            # Convert the dual TN to the original TN
            tn = unbuild_dual_tn(dual_tn, tn_type=tn_type)
            return tn
        else:
            raise ValueError("Cannot unbuild the dual TN if physical indices have been contracted")

    
    @staticmethod
    def get_partition_function(tn):
        return tn.contract(all)
    
class QBP(GenericTNModel):
    def __init__(self, params, tn, tn_type, datapoint = None):
        tensor_dtype = params['tensor_dtype']
        contract_physical_indices = params['contract_physical_indices']
        tol = params.get('tol', 1e-6)
        max_iter = params['max_iter']
        improvement_rate_threshold = params.get('improvement_rate_threshold', 1e-2)
        stable_convergence_required = params.get('stable_convergence_required', 15)
        device = params['device']
        z_dim = params['z_dim']
        bond_dim = params['bond_dim']
        phys_dim = params['phys_dim']
        hidden_dim = params['hidden_dim_msg']
        check_convergence = params['check_convergence']
        
        super(QBP, self).__init__(tensor_dtype=tensor_dtype, contract_physical_indices=contract_physical_indices, device=device)
        self.messages = PSD_MessageModule(z_dim=z_dim, bond_dim=bond_dim, hidden_dim=hidden_dim, dtype=tensor_dtype, device=device)
        self.tol = tol
        self.max_iter = max_iter
        self.only_bp = False
        self.improvement_rate_threshold = improvement_rate_threshold
        self.stable_convergence_required = stable_convergence_required
        self.show_progress_bar = params.get('show_progress_bar', False)
        self.dual_tn = None
        self.set_tn(tn, tn_type)
        if datapoint is not None:
            self.set_datapoint(datapoint)
        self.check_convergence = check_convergence
        self.bond_dim = bond_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.phys_dim = phys_dim

    def set_datapoint(self, datapoint):
        self.datapoint = datapoint

    def set_tn(self, tn, tn_type):
        self.tn = tn
        self.tn_type = tn_type
        self.energy = None
        self.hamiltonian = None
        self.one_rdms = {}
        self.two_rdms = {}
        self.dual_tn = self.build_dual(self.tn.copy(), tn_type=self.tn_type)

        #Print what tensors have NaN values
        tensors, _, _ = self.dual_tn
        for i, tensor in enumerate(tensors):
            if torch.isnan(tensor).any():
                print(f"Tensor {i} has NaN values")

        #Assert no tensor has NaN values
        assert not any([torch.isnan(tensor).any() for tensor in tensors]), "One of the tensors has NaN values"

        self.messages.clear()
        self.initialize_message_structure()

    def initialize_message_structure(self):
        if self.contract_physical_indices:
            _, skeleton, _ = self.dual_tn
            self.messages.initialise_structure(skeleton, contracted=self.contract_physical_indices)
        else:
            dual_tn_unpacked = self.unpack(self.dual_tn)
            self.messages.initialise_structure(dual_tn_unpacked, contracted=self.contract_physical_indices)

        self.edge_index, self.int_to_node, self.node_to_int = self.messages.get_edge_index()

    @staticmethod
    def normalize(matrix, id=False, reshape=True):
        # Normalize message to have unit norm, enhancing numerical stability

        if matrix.dim() > 2:
            mat_copy = matrix.clone()
            sqr_dim = torch.sqrt(torch.tensor(torch.numel(matrix)))
            assert sqr_dim % 1 == 0, "The number of elements in the matrix is not a perfect square"
            mat_copy = mat_copy.reshape(int(sqr_dim), int(sqr_dim))
            if id:
                trace = torch.ones(1, dtype=mat_copy.dtype)
            else:
                trace = torch.trace(mat_copy)

            if reshape:
                matrix = mat_copy
        else:
            trace = torch.trace(matrix)
        
        if torch.isclose(trace, torch.zeros(1, dtype=matrix.dtype, device=matrix.device)):
            return matrix
        
        return matrix / trace
        
    # From a sender and receiver, extract the indices that connect them, the physical indices of the sender, and all the indices of the sender
    def extract_indices(self, sender, receiver=None):
        _, skeleton, _ = self.dual_tn
        sender_indices = sender.split("_")
        
        # If we have more than one sender index or receiver index, we need to trace over the physical indices
        # We can check the skeleton to check what the physical indices are in the tensor
        contracted_tn = len(sender_indices) == 1

        assert len(sender_indices) > 0 and len(sender_indices) <=2, "Sender Indices Incorrect"


        inds = []
        for idx in sender_indices:
            inds+=skeleton.tensor_map[int(idx)].inds
        
        if receiver is not None:
            receiver_indices = receiver.split("_")
            assert len(receiver_indices) > 0 and len(receiver_indices) <=2, "Receiver Indices Incorrect"
            receiver_indices = receiver.split("_")

            receiver_inds = []
            for idx in receiver_indices:
                receiver_inds+=skeleton.tensor_map[int(idx)].inds

        if contracted_tn:
            common_inds = OrderedSet()
        else:
            #Only keep duplicate indices to find the physical indices Ta-Ta*
            common_inds = [item for item in inds if inds.count(item) > 1]
            common_inds = OrderedSet(common_inds) 
            
        #All of the indices in Ta,Ta*
        inds = OrderedSet(inds) 

        if receiver is not None:
            #Find the indices that connect the sender to the receiver
            sender_receiver_inds = inds.intersection(receiver_inds)
        else:
            sender_receiver_inds = OrderedSet()

        assert len(common_inds) <= 1, "More than one common index found between Ta and Ta*"

        return inds, common_inds, sender_receiver_inds
    
    def compute_rdm(self, tensor_idx_int=[]):
        # Get the senders and receivers from the edge index
        order = len(tensor_idx_int)

        if order <= 0:
            raise ValueError("Order must be positive")
        if order >1:
            pass

        tensors, skeleton, _ = self.dual_tn
        nodes = [self.int_to_node.get(idx) for idx in tensor_idx_int]
        int_nodes = [[int(str_val) for str_val in node.split("_")] for node in nodes]
        tensor_nodes = [[tensors[idx] for idx in idx_list] for idx_list in int_nodes]
        #Join all elements to one list of tensors
        tensor_nodes = [tensor for sublist in tensor_nodes for tensor in sublist]

        # Validate input
        unique_combos = set()
        for node_list in int_nodes:
            unique_combos.add(tuple(node_list))
        if len(unique_combos) != order:
            raise ValueError("All input tensors must have unique node combinations")

        common_inds = OrderedSet()
        for node in nodes:
            _, common_inds_node, _ = self.extract_indices(node)
            common_inds = common_inds.union(common_inds_node)

        if len(common_inds) == 0:
            raise NotImplementedError("No common indices found, cannot compute RDMs")

        # Get all neighbour messages excluding messages from input tensors
        all_neighbours = {}
        message_keys = []
        for target_node in nodes:
            neighbours = [n for n in self.messages.get_neighbours(target_node) ]
            neighbours = [n for n in neighbours if n not in nodes]

            if len(neighbours) > 0:
                all_neighbours[target_node] = neighbours
            message_keys.extend([f"{neighbour}->{target_node}" for neighbour in neighbours])
        
        tensor_ind_maps = []
        node_indices = []
        for idx in int_nodes:
            for ind in idx:
                node_indices += list(skeleton.tensor_map[ind].inds)
                tensor_ind_maps.append(skeleton.tensor_map[ind].inds)

        if len(all_neighbours) > 0:
            # Get messages from neighbours
            message_tup = [self.messages.get_message(m_key) for m_key in message_keys]
            assert all([m[0] is not None for m in message_tup]), "Some messages are None"

            # Collect messages from all neighbours
            neighbour_iter = iter(all_neighbours.items())
            iter_val = next(neighbour_iter)
            messages_list, message_inds = self.collect_neighbour_messages(node_indices, iter_val[1], iter_val[0])
            if len(nodes) > 1:
                for node, neighbours in neighbour_iter:
                    messages_list_new, message_inds_new = self.collect_neighbour_messages(node_indices, neighbours, node)
                    messages_list = messages_list + messages_list_new
                    message_inds = torch.cat([message_inds, message_inds_new], dim=0)
        else:
            #This case happens on an MPS where a node is at the end or start of the chain and sending to its neighbour
            messages_list = []
            message_inds = torch.tensor([])
            
        
        # Obtain the index of the common indices in the tensors
        output_inds = torch.tensor([i for i, idx in enumerate(node_indices) if idx in common_inds])

        # Obtain the index of the common indices in the tensors in the skeleton (assuming same for all nodes)
        output_ind_pos = []
        for node in int_nodes:  # Iterate same number of times as input tensors
            for idx in node:
                tensor_ids = skeleton.tensor_map[idx].inds
                #Check what common index is in the tensor
                common_inds_tensor = common_inds.intersection(tensor_ids)
                assert len(common_inds_tensor) == 1, "More than one common index found in the tensor"
                output_ind_pos.append(skeleton.tensor_map[idx].inds.index(common_inds_tensor[0]))

        reorder_phys_inds = False
        if not self.contract_physical_indices and order > 1:
            reorder_phys_inds = True     
        rdm = compute_partial_trace(tensor_nodes, tensor_ind_maps, messages_list, message_inds, output_inds, output_ind_pos, node_indices)
        
        #Save the output indices of the RDM so they can be used later if necessary
        rdm_inds = [node_indices[i] for i in output_inds]
        tensor_inds = [val for sublist in int_nodes for val in sublist]

        #Ensures output is k0k1k2k0*k1*k2 etc.
        def sort_and_reorder_tensor_and_additional_lists(tensor, index_list, sort_first, *additional_lists):
            if sort_first:
                # Step 1: Create a sorting order based on the index list to group and sort indices
                grouped_order = sorted(range(len(index_list)), key=lambda x: (index_list[x][:-1], int(index_list[x][-1])))
            else:
                # Use the natural order if sorting is not required
                grouped_order = list(range(len(index_list)))

            # Step 2: Sort the tensor according to this grouped order
            sorted_tensor = tensor.permute(grouped_order)
            sorted_index_list = [index_list[i] for i in grouped_order]

            # Sort additional lists with the same grouped order
            sorted_additional_lists = [[lst[i] for i in grouped_order] for lst in additional_lists]
            
            # Step 3: Calculate the new order for interleaving original and conjugate indices
            new_order = [i // 2 + (i % 2) * (len(sorted_index_list) // 2) for i in range(len(sorted_index_list))]

            # Step 4: Reorder the sorted tensor and lists according to new_order
            reordered_tensor = sorted_tensor.permute(new_order)
            reordered_index_list = [sorted_index_list[i] for i in new_order]
            reordered_additional_lists = [[sorted_lst[i] for i in new_order] for sorted_lst in sorted_additional_lists]

            # Adjust the output format to be simpler when there's only one additional list
            if len(reordered_additional_lists) == 1:
                reordered_additional_lists = reordered_additional_lists[0]

            return reordered_tensor, reordered_index_list, reordered_additional_lists
        

        if reorder_phys_inds:
            rdm, rdm_inds, (tensor_inds) = sort_and_reorder_tensor_and_additional_lists(rdm, rdm_inds, True, tensor_inds)

        
        #Normalize the RDM
        rdm = QBP.normalize(rdm)

        return rdm, (rdm_inds, tensor_inds)
            
    def collect_neighbour_messages(self, tensor_inds, neighbours, receiver):
        message_keys = [f"{neighbour}->{receiver}" for neighbour in neighbours]
        #Get the messages from the neighbours
        message_tup = [self.messages.get_message(m_key) for m_key in message_keys]
        #Assert no message is None
        assert all([m[0] is not None for m in message_tup]), "Some messages are None"

        #Stack the messages and indices
        messages_list = [m[0] for m in message_tup]
        messages_ind_keys = [list(m[1]) for m in message_tup]
        #Each message has two indices, we must verify that the first one's index in tensor_inds is before the second one's index, if not swap them
        for i, message_inds in enumerate(messages_ind_keys):
            if tensor_inds.index(message_inds[0]) > tensor_inds.index(message_inds[1]):
                messages_list[i] = messages_list[i].t()
                messages_ind_keys[i] = [message_inds[1], message_inds[0]]
            

        #We now have to convert the index keys from message_ind_keys to their position in the contracted_inds list, in a one liner
        message_inds = torch.stack([torch.tensor([list(tensor_inds).index(ind) for ind in message_inds]) for message_inds in messages_ind_keys])
        

        #As a sanity check that each index appears in message_sender_inds once
        assert all([len(OrderedSet(message_inds[i])) == len(message_inds[i]) for i in range(len(message_inds))]), "Some indices appear more than once in the message indices"

        return messages_list, message_inds
    
    def compute_message(self, key):
        # We implement the message computation for Belief Propagation in Tensor Networks from Alkabetz et al. 2020,

        tensors, skeleton, _ = self.dual_tn

        assert not any([torch.isnan(tensor).any() for tensor in tensors]), "One of the tensors has NaN values"

        assert not any([torch.isnan(t).any() for t in tensors]), "One of the tensors has NaN values"

        #We will refer to Ta,Ta* as sender, Tb,Tb* as receiver
        sender, receiver = key.split("->")
        # Get the tensors at the sender and receiver nodes
        sender_indices = sender.split("_")
        receiver_indices = receiver.split("_")

        # If we have more than one sender index or receiver index, we need to trace over the physical indices
        # We can check the skeleton to check what the physical indices are in the tensor
        contracted_tn = len(sender_indices) == 1 and len(receiver_indices) == 1

        _, common_inds, sender_receiver_inds = self.extract_indices(sender, receiver)
        
        if not contracted_tn: #Contract physical indices of sender

            #Find Position of the common indices in the tensor
            pos = [skeleton.tensor_map[int(idx)].inds.index(ind) for idx in sender_indices for ind in common_inds]

            #Get the tensors and
            is_conj = [any('*' in str(tag) for tag in list(skeleton.tensor_map[int(idx)].tags)) for idx in sender_indices]
            tensor_data = [tensors[int(idx)] for idx in sender_indices]

            #Contraction should be TaTa* so we reorder the tensors if necessary
            if is_conj[0] and not is_conj[1]:
                tensor_data = [tensor_data[1], tensor_data[0]]
                pos = [pos[1], pos[0]]
                
            assert pos[0] == pos[1]
            #Contract the tensors by moving the common index from pos to the end
            tensors_aux = [torch.movedim(t, pos[0], -1) for t in tensor_data]
            pos[0] = len(tensors_aux[0].shape) - 1
            pos[1] = len(tensors_aux[1].shape) - 1
            #Contract the tensors using a matrix multiplication, contracted tensor will be of (dims(Ta) - 1) + (dims(Ta*) - 1)
            contracted = dynamic_tensor_mult(tensors_aux[0], tensors_aux[1], dims=([pos[0]], [pos[1]]))
            contracted_inds = list(skeleton.tensor_map[int(sender_indices[0])].inds) + list(skeleton.tensor_map[int(sender_indices[1])].inds)
            contracted_inds = [item for item in contracted_inds if item not in common_inds]
            assert len(contracted_inds) == contracted.ndim, "Number of contracted indices does not match the number of dimensions of the contracted tensor"
        else:
            contracted = tensors[int(sender_indices[0])]
            contracted_inds = list(skeleton.tensor_map[int(sender_indices[0])].inds)
        #This gives us TaTa* we now compute the product of the messages
        #Get the neighbours of the sender excluding the receiver
        neighbours = self.messages.get_neighbours(sender, exclude_nodes=[receiver])

        if len(neighbours) <= 0:
            #If the sender has no neighbours, we return the contracted tensors according to the message equation
            message = contracted
            output_inds = list(range(0, contracted.ndim))
            output_ind_labels = [contracted_inds[i] for i in output_inds]
        else:
            #We collect the messages from the neighbours
            messages_list, message_inds = self.collect_neighbour_messages(contracted_inds, neighbours, sender)

            #Also get the indices of message_sender_inds in contracted_inds
            output_inds = torch.tensor([list(contracted_inds).index(ind) for ind in sender_receiver_inds])
            output_ind_labels = [contracted_inds[i] for i in output_inds]

            #As a sanity check that each index appears in message_sender_inds once
            assert all([len(OrderedSet(message_inds[i])) == len(message_inds[i]) for i in range(len(message_inds))]), "Some indices appear more than once in the message indices"
            # We must perform a partial trace on the contracted tensor with each message accross the indices in message_inds
            message = compute_partial_trace([contracted], [contracted_inds], messages_list, message_inds, output_inds, output_inds, contracted_inds)
            assert message is not None, "Message is None"
            message = QBP.normalize(message)
        
        #Assert the output ind labels match the ones in the saved message, in the same order
        message_output_ind_labels = list(self.messages.get_message(key)[1])
        assert all([output_ind_labels[i] == message_output_ind_labels[i] for i in range(len(output_ind_labels))]), "The output indices do not match the saved message indices"
            
        #Now calculate the change in the message
        old_message, old_inds = self.messages.get_message(key)
        assert old_message.shape == message.shape, 'The shape of the old message does not match the shape of the new message: f"{old_message.shape} vs {message.shape}"'
        assert OrderedSet(old_inds) == OrderedSet(sender_receiver_inds), "The indices of the old message do not match the indices of the new message"
        assert old_message is not None, "Old message is None"
        delta = QBP.calculate_distance(message, old_message, dtype=message.dtype)
        return message, delta
    
    def get_named_parameters(self):
        for node_idx in range(len(self.dual_tn[0])):
            yield f"tensor_{node_idx}", self.dual_tn[0][node_idx]
        
        for key in self.messages.messages.keys():
            yield f"message_{key}", self.messages.messages[key]
    
    @staticmethod
    def calculate_distance(matrix1, matrix2, method = "frobenius", dtype=torch.cfloat):
        if method == "frobenius":
            return torch.dist(matrix1, matrix2)
        elif method == "trace distance":
            delta = matrix1 - matrix2
            _, s, _ = torch.linalg.svd(delta)
            return 0.5 * torch.sum(torch.abs(s), dim=-1)
        elif method == "fidelity":
            return fidelity_torch(matrix1, matrix2, dtype)
        else:
            raise ValueError("Method not recognised")


    def estimate_local_hamiltonian(self, one_rdms, n):
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=self.tensor_dtype, device=self.device)
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=self.tensor_dtype, device=self.device)
        I = torch.eye(2, dtype=self.tensor_dtype, device=self.device)

        H_n = self.datapoint.x_nodes[n][0] * sigma_z + self.datapoint.x_nodes[n][1] * sigma_x
        N = len(one_rdms)
        for j in range(N):
            if j != n:
                H_n = H_n + self.datapoint.x_nodes[j][0] * torch.trace(one_rdms[j] @ sigma_z) * I + self.datapoint.x_nodes[j][1] * torch.trace(one_rdms[j] @ sigma_x) * I
        
        edge_index = self.datapoint.edge_index
        completed_edges = []
        #Add Interaction terms by considering neighbours
        for i in range(edge_index.shape[1]):
            #Skip if interaction term is 0
            node1 = edge_index[0, i].item()
            node2 = edge_index[1, i].item()
            if self.datapoint.x_edges[i] == 0:
                continue
            if (node1, node2) in completed_edges:
                continue
            if node1 == n or node2 == n: 
                if node1 == n:
                    node = node2
                else:
                    node = node1
                H_n = H_n + self.datapoint.x_edges[i] * torch.trace(one_rdms[node] @ sigma_z) * sigma_z    
            if node1 != n and node2 != n:
                H_n = H_n + self.datapoint.x_edges[i] * torch.trace(one_rdms[node1] @ sigma_z) * torch.trace(one_rdms[node2] @ sigma_z) * I
            completed_edges.append((node2, node1))
            completed_edges.append((node1, node2))

        return H_n

    def calculate_energy_mean_field(self):
        one_rdms, _, _ = self.prepare_data()
        energy = torch.zeros(1, dtype=self.tensor_dtype, device=self.device)
        for n in range(len(one_rdms)):
            rho_n = one_rdms[n]
            H_n = self.estimate_local_hamiltonian(one_rdms, n)
            energy += torch.trace(rho_n @ H_n)  # Calculate trace of the product and sum up

        return energy[0] / len(one_rdms)
    
    def prepare_data(self, x_nodes, edge_index):
        data_edge_index = edge_index
        two_rdms = []
        one_rdms = []

        #We sort the rdms to match the order of x_nodes and x_edges
        for i in range(data_edge_index.shape[1]):
            node1 = self.int_to_node[data_edge_index[0, i].item()]
            node2 = self.int_to_node[data_edge_index[1, i].item()]
            if (node1, node2) not in self.two_rdms:
                #FIXME: This is a temporary fix, we should not be calculating the two RDMs for non-neighbours
                # However, for the energy calculation this works as we multiply by I which doesnt affect the output
                two_rdms.append(torch.eye(4, dtype=self.tensor_dtype, device=self.device))
            else:
                two_rdms.append(self.two_rdms[(node1, node2)][0])
        
        for i in range(len(x_nodes)):
            node = self.int_to_node[i]
            if node not in self.one_rdms:
                raise ValueError(f"One RDM for node {node} not found")
            else:
                one_rdms.append(self.one_rdms[node][0])

        batch_index = torch.tensor([1] * len(one_rdms), device=self.device)

        one_rdms = torch.stack(one_rdms)
        two_rdms = torch.stack(two_rdms)
        
        one_rdms = one_rdms.to(self.device)
        two_rdms = two_rdms.to(self.device)
        return one_rdms, two_rdms, batch_index


    def calculate_energy(self, x_nodes=None, x_edges=None, edge_index=None): #Doesn't localize the Hamiltonian
        if self.energy is not None:
            return self.energy
        if x_nodes is None and x_edges is None and edge_index is None:
            assert self.datapoint is not None, "No datapoint provided"
            x_nodes = self.datapoint.x_nodes
            x_edges = self.datapoint.x_edges
            edge_index = self.datapoint.edge_index
        one_rdms, two_rdms, batch_index = self.prepare_data(x_nodes, edge_index)
        energy = calculate_energy_batched(one_rdms=one_rdms, two_rdms=two_rdms, node_params=x_nodes, edge_params=x_edges, edge_index=edge_index, batch_index=batch_index)
        return energy[0]

    #TODO: Use
    def maintain_conj_tensors(self):
        if not self.contract_physical_indices:
            for key in self.messages.node_to_int.keys():
                nodes = [int(node) for node in key.split("_")]
                #Set the conjugate tensor to be the complex conjugate of the original tensor, detaching it from the computation graph
                self.dual_tn[0][nodes[1]] = self.dual_tn[0][nodes[0]].conj()
                # self.dual_tn[0][nodes[1]].requires_grad = True

    def forward(self, z, edge_index_out, x_nodes=None, x_edges=None):
        #edge_index_out represents a fully connected graph whilst self.edge_index represents the TN graph
        sender, receiver = self.edge_index
        state = torch.cat((z[sender], z[receiver]), dim=1)
        self.energy = None

        self.messages.initialize_messages(state)
        self.maintain_conj_tensors()


        assert len(self.dual_tn) > 0, "The tensor network has not been set"
        assert len(z) == len(self.tn.tensors), "The number of parameters does not match the number of tensors in the tensor network"
        assert len(self.messages) > 0, "The messages have not been initialised"

              
        
        prev_change = float("inf")  # Store the previous change for comparison
        stable_converge_count = 0  # Count how many times the change has been below the threshold consecutively
        iter_change = []  # Store the change in messages for each iteration

        for iter in tqdm(range(self.max_iter), disable=not self.show_progress_bar):
            deltas = []
            for i, (key, (message, _)) in enumerate(self.messages.iterate_messages()):
                # Compute the message
                message_new, delta = self.compute_message(key)
                assert (not torch.isnan(delta).any()) and (not torch.isinf(delta).any()), f"Delta from Message {key} has NaN or inf values"
                m_ok = self.messages.check_message(message=message_new)
                if not m_ok:
                    print(f"Message {key} is not correct")
                assert m_ok, f"Message {key} is not correct"
                deltas.append(delta)
                if delta == 0:
                    continue
                # Update the message
                self.messages.update_message(key, message_new, commit=False)  # Assuming commit=True updates the message
            
            self.messages.commit_updates()  # Commit all message updates after the iteration
            
            change = torch.max(torch.abs(torch.stack(deltas)))  # Maximum absolute change across all messages
            rate_of_improvement = abs((change - prev_change) / prev_change) if prev_change != float("inf") else float("inf")
            iter_change.append(change)
            
            if self.check_convergence:
                if change <= self.tol or rate_of_improvement < self.improvement_rate_threshold:
                    stable_converge_count += 1
                    if stable_converge_count >= self.stable_convergence_required:
                        if self.show_progress_bar:
                            print(f"Converged after {iter+1} iterations, with maximum change {change}")
                        break
                else:
                    stable_converge_count = 0  # Reset if conditions are not met
            
            prev_change = change  # Update previous change for next iteration's comparison

        if (stable_converge_count < self.stable_convergence_required) and self.check_convergence:
            print(f"Did not converge after {self.max_iter} iterations, with last change {change}")

        #Attach Messages to computation graph
        # self.messages.attach_to_computation()

        unique_node_int = OrderedSet(self.edge_index[0].tolist() + self.edge_index[1].tolist())
        
        self.rdms = []
        for node_int in unique_node_int:
            rdm = self.compute_rdm(tensor_idx_int=[node_int])
            self.rdms.append(rdm[0])
            self.one_rdms[self.int_to_node[node_int]] = rdm

        self.rdms2 = []
        #Iterate through edge index to calculate the two RDMs
        for i in range(edge_index_out.shape[1]):
            intNode1 = edge_index_out[0, i].item()
            intNode2 = edge_index_out[1, i].item()
            node1 = self.int_to_node[intNode1]
            node2 = self.int_to_node[intNode2]

            directed_edge = torch.tensor([intNode1, intNode2])
            exists = ((self.edge_index.t() == directed_edge).all(dim=1)).any()
            
            if not exists:
                #That way they do not add information when aggregating
                self.rdms2.append(torch.zeros((4,4), dtype=self.tensor_dtype, device=self.device))
            else:
                if (node1, node2) not in self.two_rdms:
                    self.two_rdms[(node1, node2)] = self.compute_rdm(tensor_idx_int=[edge_index_out[0, i].item(), edge_index_out[1, i].item()])
                else:
                    self.two_rdms[(node1, node2)] = self.compute_rdm(tensor_idx_int=[edge_index_out[0, i].item(), edge_index_out[1, i].item()])    
                self.rdms2.append(self.two_rdms[(node1, node2)][0])

        

        #Stack the RDMs
        self.rdms = torch.stack(self.rdms)
        self.rdms2 = torch.stack(self.rdms2)
        if x_nodes is not None and x_edges is not None:
            self.energy = self.calculate_energy(x_nodes, x_edges, edge_index_out)
        return self.rdms, self.rdms2, self.energy

    def parameters(self):
        num_tensors = len(self.dual_tn[0])

        return self.dual_tn[0][0:num_tensors//2]
    
    
    def getRDMs(self):
        if len(self.one_rdms) == 0 or len(self.two_rdms) == 0:
            print("The one or two RDMs have not been calculated, please run the forward method first")
        return self.one_rdms, self.two_rdms
    
    def getEnergy(self):
        if self.energy is None:
            print("The energy has not been calculated, please run the forward method first")
        return self.energy.cpu().detach()
    
class QGNN_Layer(nn.Module):
    def __init__(self, params):
        super(QGNN_Layer, self).__init__()
        self.z_dim = params['z_dim']
        self.hidden_dim = params['hidden_dim']
        self.hidden_dim_msg = params['hidden_dim_msg']
        self.hidden_dim_z = params['hidden_dim_z']
        
        shape = params['grid_extent']
        if len(shape) == 1:
            shape = (shape[0], 1)
        self.Lx, self.Ly = shape

        self.phys_dim = params['phys_dim']
        self.bond_dim = params['bond_dim']
        self.dtype = params['tensor_dtype']
        self.device = params['device']
        self.pbc = params['pbc']
        self.normalize_tn_init = params['normalize_tn_init']

        self.tn, _, self.tn_type = generate_tensor_network(self.Lx, self.Ly, bond_dim=self.bond_dim, phys_dim=self.phys_dim, pbc=self.pbc,  dtype=self.dtype, normalize = self.normalize_tn_init)
        self.QBP = QBP({**params, 'check_convergence': False}, tn=self.tn, tn_type=self.tn_type)

        self.z_net = nn.Sequential(
            nn.Linear(self.phys_dim*2 + 2*(2*(2**self.phys_dim)) + self.z_dim + 1, self.hidden_dim_z, dtype=self.dtype),
            nn.SiLU(),
            nn.Linear(self.hidden_dim_z, self.z_dim, dtype=self.dtype)
        )

    def calculate_energy(self, x_nodes=None, x_edges=None, edge_index=None):
        return self.QBP.calculate_energy(x_nodes, x_edges, edge_index)


    def forward(self, z, edge_index, x_nodes=None, x_edges=None):
        sender, receiver = edge_index
        num_edges = edge_index.shape[1]
        num_nodes = z.shape[0]

        onerdms, twordms, energy = self.QBP(z, edge_index, x_nodes=x_nodes, x_edges=x_edges)

        #Flatten twordms
        twordms = twordms.view(twordms.shape[0], -1)
        #Flatten onerdms
        onerdms = onerdms.view(onerdms.shape[0], -1)
        
        #Aggregate two rdms 
        twordm_aggr = torch.zeros(num_nodes, twordms.shape[1], dtype=twordms.dtype, device=twordms.device)
        twordm_aggr.scatter_add_(0, receiver.view(-1, 1).expand(num_edges, twordms.shape[1]), twordms)

        states = torch.cat((onerdms, twordm_aggr, z, energy.repeat(z.shape[0], 1).to(z.dtype)), dim=1)
        z_new = self.z_net(states)

        return z_new, onerdms, twordms

class QGNN(nn.Module):
    def __init__(self, params):
        super(QGNN, self).__init__()
        self.num_layers = params['num_layers']
        self.tensor_dtype = params['tensor_dtype']
        self.device = params['device']
        self.use_residual = params['use_residual']
        self.z_dim = params['z_dim']

        shape = params['grid_extent']
        if len(shape) == 1:
            shape = (shape[0], 1)
        self.Lx, self.Ly = shape

        self.layers = nn.ModuleList([QGNN_Layer(params) for _ in range(self.num_layers)])

        #Calculate the degree of a node in a fully connected graph from the number of nodes
        fc_degree = self.Lx * self.Ly - 1

        self.embed_z = nn.Sequential(
            nn.Linear(fc_degree + 2, 16, dtype=self.tensor_dtype),
            nn.ReLU(),
            nn.Linear(16, self.z_dim, dtype=self.tensor_dtype),
        )

    def create_node_features(self, x_nodes, x_edges, edge_index):
        # Get the number of nodes and the size of the edge features
        num_nodes = x_nodes.shape[0]
        edge_feature_size = x_edges.shape[1]

        #Cast to tenor_dtype    
        x_nodes = x_nodes.to(self.tensor_dtype)
        x_edges = x_edges.to(self.tensor_dtype)

        num_edge_features = num_nodes - 1

        # Initialize a tensor to hold the node features
        node_features = torch.zeros((num_nodes, 2 + num_edge_features*edge_feature_size), dtype=self.tensor_dtype, device=self.device)

        # Add the x_nodes features to the node features
        node_features[:, :2] = x_nodes

        # For each node, gather the edge features of the edges connected to it and stack them to the node features
        for i in range(num_nodes):
            edge_indices = (edge_index == i).any(dim=0)
            edge_features = x_edges[edge_indices]
            node_features[i, 2:] = edge_features.flatten()
        
        return node_features
        

    def forward(self, datapoint, format_output=False):
        shape, x_nodes, x_edges, edge_index = datapoint.grid_extent, datapoint.x_nodes, datapoint.x_edges, datapoint.edge_index

        if len(shape) == 1:
            shape = (shape[0], 1)
        Lx, Ly = shape

        assert Lx == self.Lx and Ly == self.Ly, "Grid extent does not match the one used to initialise the model"

        node_features = self.create_node_features(x_nodes, x_edges, edge_index)
        z = self.embed_z(node_features)

        for i, layer in enumerate(self.layers):
            _z, onerdms, twordms = layer(z, edge_index, x_nodes, x_edges)
            if self.use_residual:
                z = z + _z
            else:
                z = _z
        
        energy = self.layers[-1].calculate_energy(x_nodes, x_edges, edge_index)
        if not format_output:
            onerdms, twordms = self.layers[-1].QBP.getRDMs()
        else:
            onerdms, twordms, _ = self.layers[-1].QBP.prepare_data(x_nodes, edge_index)
        return energy, onerdms, twordms







        