from typing import Iterator
import torch
import torch.nn as nn
from itertools import combinations
import quimb.tensor as qtn
from tqdm import tqdm
from ordered_set import OrderedSet
import quimb as qu

import sys
sys.path.append(".")


from qbp_gnn.tensor_network import tn_tensors_to_torch, build_dual_tn, unbuild_dual_tn, generate_tensor_network, visualize_tensor_network, dynamic_tensor_mult, compute_partial_trace
from qbp_gnn.tensor_network import contract_physical_indices as contract_phys_ind
from collections import defaultdict

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
    def __init__(self, dtype=torch.float32, psd_check_tol = 1e-8, device='cpu'):
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
                
        return ispsd 
    
    # def initialise_message(self, shape):
    #     A = torch.rand(shape, dtype=self.dtype)
    #     cmplx = torch.is_complex(A)

    #     #assert len(shape) == 2 and shape[0] == shape[1], f"Message shape {shape} is not 2D and square"
    
    #     if cmplx:
    #         A = (2 * A) - (1 + 1j) # Make A's values between [-1 -1j, 1 + 1j]
    #         # For complex numbers, ensure Hermitian by adding conjugate transpose
    #         H = 0.5 * (A + A.conj().t())
    #     else:
    #         A = (2 * A) - 1 # Make A's values between [-1, 1]
    #         # For real numbers, ensure symmetric by averaging with transpose
    #         H = 0.5 * (A + A.t())
        
    #     # Make G Hermitian (or symmetric)
    #     # H = 0.5*(A + A.conj().t()) if cmplx else 0.5 * (A + A.t())

    #     # Construct a PSD matrix by multiplying H by its conjugate transpose (or transpose)
    #     PSD = H @ H.conj().t() if cmplx else H @ H.t()
       
    #     return PSD
    
    
    def initialise_message(self, shape):
        # Determine if the dtype is complex
        is_complex = self.dtype.is_complex

        # if is_complex:
        #     # Initialize real and imaginary parts separately
        #     real_part = 2 * torch.rand(shape, dtype=torch.float32) - 1
        #     imag_part = 2 * torch.rand(shape, dtype=torch.float32) - 1
        #     # Combine into a complex tensor
        #     A = torch.complex(real_part, imag_part)
        # else:
        #     # Initialize real tensor
        #     A = 2 * torch.rand(shape, dtype=torch.float32) - 1
        
        # # Ensure the matrix is Hermitian (complex) or symmetric (real)
        # if is_complex:
        #     A = 0.5 * (A + A.conj().transpose(-2, -1))
        # else:
        #     A = 0.5 * (A + A.transpose(-2, -1))

        eps = 0.01  # Small noise
        if self.dtype.is_complex:
            
            if self.dtype == torch.cfloat:
                real_part = torch.full(shape, 0.5, dtype=torch.float32) + eps * (2 * torch.rand(shape, dtype=torch.float32) - 1)
                imag_part = torch.full(shape, 0.5, dtype=torch.float32) + eps * (2 * torch.rand(shape, dtype=torch.float32) - 1)
                A = real_part + 1j * imag_part
            elif self.dtype == torch.cdouble:
                real_part = torch.full(shape, 0.5, dtype=torch.float64) + eps * (2 * torch.rand(shape, dtype=torch.float64) - 1)
                imag_part = torch.full(shape, 0.5, dtype=torch.float64) + eps * (2 * torch.rand(shape, dtype=torch.float64) - 1)
                A = real_part + 1j * imag_part
        else:
            A = torch.full(shape, 0.5, dtype=self.dtype) + eps * (2 * torch.rand(shape, dtype=self.dtype) - 1)
        
        # Construct a PSD matrix by multiplying by its conjugate transpose (complex) or transpose (real)
        PSD = A.matmul(A.conj().transpose(-2, -1)) if is_complex else A.matmul(A.transpose(-2, -1))
        PSD = PSD.to(self.device)
        PSD.requires_grad = True
   
        return PSD

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
        self.messages.clear()
        self.inds.clear()
        self.int_to_node.clear()
        self.node_to_int.clear()
        self.updated_messages.clear()

    def attach_to_computation(self):
        for key in self.messages.keys():
            for message in self.messages[key]:
                self.messages[key][message] = nn.Parameter(self.messages[key][message], requires_grad=False)

    def initialise(self, tn, check_PSD=True, contracted = False):
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

            message = self.initialise_message(message_shape)
            if check_PSD:
                ok = False
                num_tries = 0
                while not ok and num_tries < 3:
                    ok = self.check_message(message)
                    if not ok:
                        message = self.initialise_message(message_shape)
                        num_tries += 1
            assert ok, f"Message {key} is not positive semi-definite after {num_tries} tries, did not save message"

            self.add_message(key, message, inds=message_inds)
        
        self.updated_messages.clear()
        # Recalculate the edge index in the end
        self.edge_index, self.int_to_node, self.node_to_int = self.get_edge_index()
    
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
    def __init__(self, tensor_dtype=torch.cfloat, contract_physical_indices=True, device='cpu'):
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
        tensor_list = [nn.Parameter(val.contiguous(), requires_grad=requires_grad[idx]) for idx, val in enumerate(params.values())]
        # tensor_list = [nn.Parameter(val, requires_grad=True) for val in params.values()]
        tensors = nn.ParameterList(tensor_list)
        tensors = tensors.to(self.device)
    
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

class HamiltonianParams():

    def __init__(self, x_nodes, x_edges, edge_index) -> None:
        self.x_nodes = x_nodes
        self.x_edges = x_edges
        self.edge_index = edge_index

    @staticmethod
    def find_edge_index(edge_index, edge):
        a, b = edge
        # Find the mask where the first row equals `a` and the second row equals `b`
        mask = (edge_index[0] == a) & (edge_index[1] == b)
        # Use the mask to find the indices
        edge_indices = mask.nonzero(as_tuple=False)
        if edge_indices.numel() == 0:
            return None  # The edge does not exist
        else:
            return edge_indices[0].item()  # Return the first occurrence index
        
    def get_hamiltonian_param(self, key1, key2):
        key1 = int(key1.split("_")[0])
        if key2 is not None:
            key2 = int(key2.split("_")[0])
        
        if key2 is None:
            return self.x_nodes[key1]
        else:
            idx = HamiltonianParams.find_edge_index(self.edge_index, (key1, key2))
            if idx is not None:
                return self.x_edges[idx]
            else:
                raise ValueError(f"Edge {key1}->{key2} not found in the edge index of Hamiltonian parameters")
        
class QBP_QGNN(GenericTNModel):
    def __init__(self, params):
        tensor_dtype = params.get('tensor_dtype', torch.cfloat)
        contract_physical_indices = params.get('contract_physical_indices', True)
        tol = params.get('tol', 1e-6)
        max_iter = params.get('max_iter', 1000)
        improvement_rate_threshold = params.get('improvement_rate_threshold', 1e-2)
        stable_convergence_required = params.get('stable_convergence_required', 15)
        device = params.get('device', 'cpu')
        
        super(QBP_QGNN, self).__init__(tensor_dtype=tensor_dtype, contract_physical_indices=contract_physical_indices, device=device)
        self.messages = PSD_MessageModule(dtype=tensor_dtype, device=device)
        self.tol = tol
        self.max_iter = max_iter
        self.only_bp = False
        self.improvement_rate_threshold = improvement_rate_threshold
        self.stable_convergence_required = stable_convergence_required
        self.show_progress_bar = params.get('show_progress_bar', False)
        

    def set_datapoint(self, tn, tn_type, datapoint):
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

        # if self.contract_physical_indices:
        #     _, skeleton, _ = self.dual_tn
        #     self.messages.initialise(skeleton, contracted=self.contract_physical_indices)
        # else:
        #     dual_tn_unpacked = self.unpack(self.dual_tn)
        #     self.messages.initialise(dual_tn_unpacked, contracted=self.contract_physical_indices)

        # self.edge_index, self.int_to_node, self.node_to_int = self.messages.get_edge_index()

        # if h_params is not None:
        #     assert len(h_params) == 3, "The Hamiltonian parameters must be a tuple of (x_nodes(Nx2), x_edges(Ex1), edge_index(2xE))"
        #     self.hamiltonian = HamiltonianParams(*h_params)
        # else:
        #     self.hamiltonian = None

        if datapoint is not None:
            self.datapoint = datapoint.to(self.device)
        else:
            self.datapoint = None

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
        rdm = QBP_QGNN.normalize(rdm)

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
            message = QBP_QGNN.normalize(message)
        
        #Assert the output ind labels match the ones in the saved message, in the same order
        message_output_ind_labels = list(self.messages.get_message(key)[1])
        assert all([output_ind_labels[i] == message_output_ind_labels[i] for i in range(len(output_ind_labels))]), "The output indices do not match the saved message indices"
            
        #Now calculate the change in the message
        old_message, old_inds = self.messages.get_message(key)
        assert old_message.shape == message.shape, 'The shape of the old message does not match the shape of the new message: f"{old_message.shape} vs {message.shape}"'
        assert OrderedSet(old_inds) == OrderedSet(sender_receiver_inds), "The indices of the old message do not match the indices of the new message"
        assert old_message is not None, "Old message is None"
        delta = QBP_QGNN.calculate_distance(message, old_message, dtype=message.dtype)
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
    
    def prepare_data(self):
        data_edge_index = self.datapoint.edge_index
        two_rdms = []
        one_rdms = []

        #We sort the rdms to match the order of x_nodes and x_edges
        for i in range(data_edge_index.shape[1]):
            node1 = self.int_to_node[data_edge_index[0, i].item()]
            node2 = self.int_to_node[data_edge_index[1, i].item()]
            if (node1, node2) not in self.two_rdms:
                # raise ValueError(f"Two RDM for edge {node1}->{node2} not found")
                #FIXME: This is a temporary fix, we should not be calculating the two RDMs for non-neighbours
                two_rdms.append(torch.eye(4, dtype=self.tensor_dtype, device=self.device))
            else:
                two_rdms.append(self.two_rdms[(node1, node2)][0])
        
        for i in range(len(self.datapoint.x_nodes)):
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


    def calculate_energy(self): #Doesn't localize the Hamiltonian
        one_rdms, two_rdms, batch_index = self.prepare_data()
        energy = calculate_energy_batched(one_rdms=one_rdms, two_rdms=two_rdms, node_params=self.datapoint.x_nodes, edge_params=self.datapoint.x_edges, edge_index=self.datapoint.edge_index, batch_index=batch_index)
        return energy[0]

    def maintain_conj_tensors(self):
        if not self.contract_physical_indices:
            for key in self.messages.node_to_int.keys():
                nodes = [int(node) for node in key.split("_")]
                #Set the conjugate tensor to be the complex conjugate of the original tensor, detaching it from the computation graph
                self.dual_tn[0][nodes[1]] = self.dual_tn[0][nodes[0]].conj()
                # self.dual_tn[0][nodes[1]].requires_grad = True

    def initialize_messages(self):
        self.messages.clear()

        if self.contract_physical_indices:
            _, skeleton, _ = self.dual_tn
            self.messages.initialise(skeleton, contracted=self.contract_physical_indices)
        else:
            dual_tn_unpacked = self.unpack(self.dual_tn)
            self.messages.initialise(dual_tn_unpacked, contracted=self.contract_physical_indices)

        self.edge_index, self.int_to_node, self.node_to_int = self.messages.get_edge_index()

    def forward(self, format_output = False):
        assert len(self.dual_tn) > 0, "The tensor network has not been set"

        with torch.no_grad():
            self.initialize_messages()
        
        assert len(self.messages) > 0, "The messages have not been initialised"

        #We do this to maintain the conjugate tensors being always the conjugate of the original tensors
        self.maintain_conj_tensors()  
             
        
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
            
            if change <= self.tol or rate_of_improvement < self.improvement_rate_threshold:
                stable_converge_count += 1
                if stable_converge_count >= self.stable_convergence_required:
                    if self.show_progress_bar:
                        print(f"Converged after {iter+1} iterations, with maximum change {change}")
                    break
            else:
                stable_converge_count = 0  # Reset if conditions are not met
            
            prev_change = change  # Update previous change for next iteration's comparison

        if stable_converge_count < self.stable_convergence_required:
            print(f"Did not converge after {self.max_iter} iterations, with last change {change}")

        #Attach Messages to computation graph
        # self.messages.attach_to_computation()

        unique_node_int = OrderedSet(self.edge_index[0].tolist() + self.edge_index[1].tolist())

        for node_int in unique_node_int:
            self.one_rdms[self.int_to_node[node_int]] = self.compute_rdm(tensor_idx_int=[node_int])

        #Iterate through edge index to calculate the two RDMs
        for i in range(self.edge_index.shape[1]):
            intNode1 = self.edge_index[0, i].item()
            intNode2 = self.edge_index[1, i].item()
            node1 = self.int_to_node[intNode1]
            node2 = self.int_to_node[intNode2]
            if (node1, node2) not in self.two_rdms:
                self.two_rdms[(node1, node2)] = self.compute_rdm(tensor_idx_int=[self.edge_index[0, i].item(), self.edge_index[1, i].item()])
            else:
                self.two_rdms[(node1, node2)] = self.compute_rdm(tensor_idx_int=[self.edge_index[0, i].item(), self.edge_index[1, i].item()])    


        if self.datapoint is not None:
            self.energy = self.calculate_energy()
        else:
            self.energy = None  # Energy has not been calculated
        if not format_output:
            return self.energy, self.one_rdms, self.two_rdms
        else:
            one_rdms, two_rdms, _ = self.prepare_data()
            return self.energy, one_rdms, two_rdms

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

def getRDMs(tn, Lx, Ly, one_rdm_keys, two_rdm_keys):
    one_rdms = {}

    two_rdms = {}

    m, n = Lx, Ly

    #Calculate dims from Lx and Ly using a square lattice 
    dims = [[2] * n] * m
        

    def compute_rdm(peps, sites, dims):
        """
        Compute the RDM for a list of sites in a PEPS.

        Parameters:
        peps (PEPS): The PEPS representing the quantum state.
        sites (List of tuples): The coordinates of the sites.
        dims (list): The dimensions of the Hilbert space at each site.

        Returns:
        numpy.ndarray: The 2-RDM of the specified sites.
        """          
        return qu.normalize(qu.partial_trace(peps, dims=dims, keep=sites))
    
    #First contract all the physical indices of a copy of the TN
    tn_copy = tn.copy()
    ground_state = tn_copy.to_dense()

    def pos(node, n):
        # calculate x,y coordinates from node index
        y = node % n[1]
        x = (node - y) // n[1]
        return x,y

    for node_label in one_rdm_keys:
        node_idx = int(node_label.split('_')[0]) # Only obtain the first as this is not the dual tensor network

        one_rdms[node_label] = torch.tensor(compute_rdm(ground_state, [pos(node_idx, (m,n))], dims))

    for node_label_1, node_label_2 in two_rdm_keys:
        edge = (int(node_label_1.split('_')[0]), int(node_label_2.split('_')[0]))
        two_rdms[(node_label_1, node_label_2)] = torch.tensor(compute_rdm(ground_state, [pos(edge[0], (m,n)), pos(edge[1], (m,n))], dims))

    return one_rdms, two_rdms


if __name__ == "__main__":
    #Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    contract_physical_indices = False
    dtype = torch.float64
    Lx, Ly = 4,1
    # edges = [(0, 1), (1, 2), (2, 3), (1, 4), (3, 1)]
    # # random J values
    # J = np.random.rand(len(edges))
    # tn, _, tn_type = generate_tensor_network(edges=edges, J=J, bond_dim=3, dtype=dtype, normalize = False)
    #FIXME: Doesnt work with PEPS for PBC
    tn, _, tn_type = generate_tensor_network(Lx, Ly, bond_dim=3, pbc=False, dtype=dtype, normalize = False)

    params = {
        'tensor_dtype': dtype,
        'contract_physical_indices': contract_physical_indices,
        'max_iter': 200,
        'tol': 1e-10,
        'improvement_rate_threshold': 1e-3,
        'stable_convergence_required': 1,
        'show_progress_bar': False,
        'device': device
    }

    defaults_DMRG_QUIMB = {
    'max_bond': 30,
    'cutoff': 1.e-10,
    'tol': 1.e-6,
    'verbosity': 0
    }

    defaults_SU = {
        'chi': 15,
        'bond_dim': 2,
        'num_iters': 100,
        'tau': [0.1, 0.01, 0.001]
    }

    # baseline = TensorNetworkRunner(DMRG_QUIMB, defaults_DMRG_QUIMB)
    baseline = TensorNetworkRunner(SimpleUpdate, defaults_SU)
    model = QBP_QGNN(params=params)


    model.set_datapoint(tn, tn_type=tn_type, datapoint=None)
    
    
    import time

    start = time.time()
    _, one_rdms, two_rdms = model()
    end_time = time.time() - start
    print(f"Time BP: {end_time}")


    start = time.time()
    one_rdms_q, two_rdms_q = getRDMs(tn, Lx, Ly, one_rdm_keys=one_rdms.keys(), two_rdm_keys=two_rdms.keys())
    end_time = time.time() - start
    print(f"Time Contraction: {end_time}")

    errors = []
    for one_rdm_key in one_rdms.keys():
        error = QBP_QGNN.calculate_distance(one_rdms[one_rdm_key][0].cpu(), one_rdms_q[one_rdm_key], method="fidelity", dtype=dtype)
        print(f"Error for {one_rdm_key}: {error}")
        errors.append(error)
        # assert torch.allclose(one_rdms[one_rdm_key][0], one_rdms_q[one_rdm_key], atol=1e-6), f"One RDM {one_rdm_key} does not match"
    num_errors = len(errors)
    print(f"Average between one_rdms and one_rdms_q: {sum(errors)/num_errors}")

    #Sum of errors between two_rdms and two_rdms_q
    sum_error = 0
    num_errors = 0
    for key in two_rdms.keys():
        error = QBP_QGNN.calculate_distance(two_rdms[key][0].cpu(), two_rdms_q[key], method="fidelity", dtype=dtype)
        print(f"Error for {key}: {error}")
        sum_error += error
        num_errors += 1
    print(f"Average between two_rdms and two_rdms_q: {sum_error/num_errors}")

    data_file = "dataset\ising\data\PEPS_3x3_N2000_PBCFalse.pt"
    # data_file = "dataset\ising\data\MPS_180_N20_10_PBCFalse.pt"
    dataset = IsingModelDataset.load(data_file)
    """ 
    idx = torch.randint(0, len(dataset), (1,)).item()
    data_point = dataset[20]

    shape = data_point.grid_extent
    if len(shape) == 1:
        shape = (shape[0], 1)
    Lx, Ly = shape

    tn, _, tn_type = generate_tensor_network(Lx, Ly, bond_dim=2, pbc=False, dtype=dtype, normalize = False)
    model.set_datapoint(tn, tn_type=tn_type, datapoint=data_point)

    energy, one_rdms, two_rdms = model() 

    print(f"Energy: {energy}")   

    params['device'] = device


    model = model.to(device)
    data_point = data_point.to(device)

    model.set_datapoint(tn, tn_type=tn_type, datapoint=data_point)

    energy, one_rdms, two_rdms = model()

    print(f"Energy: {energy}") 
    """
    #Lets run BP in 20 random datapoints and save the results of simple energy vs energy then plot the differences per datapoint
    energies = []
    energies_simple = []
    prev_idx = []

    for i in range(1):
        # idx = torch.randint(0, len(dataset), (1,)).item()
        idx = 90
        while idx in prev_idx:
            idx = torch.randint(0, len(dataset), (1,)).item()
        data_point = dataset[idx]
        print(f"Data Point {idx}")
        print(f"Shape {data_point.grid_extent}")
        energy, rdms = baseline.run(data_point)
        one_rdms = rdms[0]
        two_rdms = rdms[1]
        ground_state = baseline.algorithm.getGroundState()
        # ground_state.compress(max_bond=min(ground_state.bond_sizes())) #DMRG generates a MPS with different bond sizes, we need to compress it so they are all the same
        tn_tensors_to_torch(ground_state, dtype=dtype)
        prev_idx.append(idx)
        if data_point.pbc:
            #FIXME: BP does not work with PBC, need to fix this
            continue
        shape = data_point.grid_extent
        if len(shape) == 1:
            shape = (shape[0], 1)
        Lx, Ly = shape
        
        model.set_datapoint(ground_state, tn_type=baseline.tn_type, datapoint=data_point.to(device))
        _, one_rdms_bp, two_rdms_bp = model()
        energies.append(model.calculate_energy().cpu().detach().numpy())
        energy_mean_field = model.calculate_energy_mean_field().cpu().detach().numpy()
        data_point = data_point.to('cpu')
        
        #Print Values
        print(f"Label Energy: {data_point.y_energy}")
        print(f"DMRG/SU Energy: {energy}")
        print(f"Energy BP: {energies[-1][0]}")
        print(f"Energy Mean Field: {energy_mean_field}")

        print(f"Energy difference: {data_point.y_energy - energies[-1][0]}")
        print(f"DMRG/SU Energy difference: {data_point.y_energy - energy}")
        print(f"Mean Field Energy difference: {data_point.y_energy - energy_mean_field}")

        #Create dictionary with dmrg rdms to follow same format as BP
        one_rdms_dmrg = {}
        two_rdms_dmrg = {}
        two_rdms_labels = {}
        one_rdms_labels = {}

        for key in one_rdms_bp.keys():
            one_rdms_dmrg[key] = torch.tensor(one_rdms[int(key.split('_')[0])], dtype=dtype)
            one_rdms_labels[key] = torch.tensor(data_point.y_node_rdms[int(key.split('_')[0])])
        
        for key in two_rdms_bp.keys():
            edge = (int(key[0].split('_')[0]), int(key[1].split('_')[0]))
            # Ensure both (u, v) and (v, u) are considered the same edge
            reversed_edge = (edge[1], edge[0])
            found = False
            for i in range(data_point.edge_index.shape[1]):
                current_edge = data_point.edge_index[0, i].item(), data_point.edge_index[1, i].item()
                if current_edge == edge or current_edge == reversed_edge:
                    two_rdms_dmrg[key] = torch.tensor(two_rdms[i], dtype=dtype)
                    two_rdms_labels[key] = torch.tensor(data_point.y_edge_rdms[i], dtype=dtype)
                    found = True
                    break
            assert found, f"Edge {edge} not found in data point edge index"
            
        one_rdms_bp_q, two_rdms_bp_q = getRDMs(ground_state, Lx, Ly, one_rdm_keys=one_rdms_bp.keys(), two_rdm_keys=two_rdms_bp.keys())

        #Print the distance between the one RDMs
        for key in one_rdms_bp.keys():
            error = QBP_QGNN.calculate_distance(one_rdms_labels[key], one_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
            if not torch.allclose(error, torch.ones(1, dtype=error.dtype), atol=1e-6):
                print(f"[BP-Label]Error for {key}: {error}")
            # print(f"[BP-Label]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(one_rdms_dmrg[key], one_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
            # print(f"[DMRG-BP]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(one_rdms_dmrg[key], one_rdms_labels[key], method="fidelity", dtype=dtype)
            # print(f"[DMRG-Label]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(one_rdms_labels[key], one_rdms_bp_q[key], method="fidelity", dtype=dtype)
            # print(f"[BPQ-Label]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(one_rdms_dmrg[key], one_rdms_bp_q[key], method="fidelity", dtype=dtype)
            # print(f"[DMRG-BPQ]Error for {key}: {error}")



        #Print the distance between the two RDMs, use edge_index to find the correct two RDM
        for key in two_rdms_bp.keys():
            error = QBP_QGNN.calculate_distance(two_rdms_labels[key], two_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
            # print(f"[BP-Label]Error for {key}: {error}")
            if not torch.allclose(error, torch.ones(1, dtype=error.dtype), atol=1e-6):
                print(f"[BP-Label]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(two_rdms_dmrg[key], two_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
            # print(f"[DMRG-BP]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(two_rdms_dmrg[key], two_rdms_labels[key], method="fidelity", dtype=dtype)
            # print(f"[DMRG-Label]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(two_rdms_labels[key], two_rdms_bp_q[key], method="fidelity", dtype=dtype)
            # print(f"[BPQ-Label]Error for {key}: {error}")
            # error = QBP_QGNN.calculate_distance(two_rdms_dmrg[key], two_rdms_bp_q[key], method="fidelity", dtype=dtype)
            # print(f"[DMRG-BPQ]Error for {key}: {error}")



            


print(f"Done!")