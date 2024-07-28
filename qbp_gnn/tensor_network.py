import numpy as np
import quimb.tensor as qtn
import torch

def create_cyclic_peps(Lx, Ly, bond_dim=3, phys_dim=2, dtype="complex64"):
    # Create a random PEPS
    peps = qtn.PEPS.rand(Lx, Ly, bond_dim=bond_dim, phys_dim=phys_dim, dtype=dtype)

    # Add cyclic indices to each tensor
    for x in range(Lx):
        for y in range(Ly):
            tensor = peps[x, y]
            existing_inds = tensor.inds
            cyclic_inds = []
            phys_ind = existing_inds[-1]

            if 'k' not in phys_ind:
                raise ValueError("The last index of the tensor should be the physical index")
            # Physical index, use it to check the position
            phys_x, phys_y = map(int, phys_ind[1:].split(','))
            
            # Infer the type of the index based on its name
            if len(existing_inds) - 1 == 2:
                #Corner case
                if (phys_x == 0 and phys_y == 0) or (phys_x == Lx-1 and phys_y == Ly-1):
                    cyclic_inds.append(f'c{0},{0}_{Lx - 1},{Ly - 1}')
                elif (phys_x == 0 and phys_y == Ly-1) or (phys_x == Lx-1 and phys_y == 0):
                    cyclic_inds.append(f'c{0},{Ly - 1}_{Lx - 1},{0}')
            elif len(existing_inds) - 1 == 3:
                # Edge case
                if phys_x == 0 or phys_x == Lx - 1:
                    cyclic_inds.append(f'c{0},{phys_y}_{Lx - 1},{phys_y}')
                elif phys_y == 0 or phys_y == Ly - 1:
                    cyclic_inds.append(f'c{phys_x},{0}_{phys_x},{Ly - 1}')

            if len(cyclic_inds) > 0:
                peps[x,y] = qtn.rand_tensor([bond_dim] * (len(cyclic_inds) + len(existing_inds) - 1) + [phys_dim], 
                                            list(existing_inds[0:-1]) + cyclic_inds + [existing_inds[-1]],
                                            tags=tensor.tags, dtype=dtype)

    return peps

def generate_structure_matrix(tn, phys_dim=2):
    num_tensors = len(tn.tensors)
    edges = {}
    edge_counter = 0

    # First pass to identify all unique connections (edges) and assign them an index
    for i, tensor1 in enumerate(tn.tensors):
        for ind_pos, ind in enumerate(tensor1.inds[:-1]):  # Ignore the last index (physical index)
            for j, tensor2 in enumerate(tn.tensors):
                if i < j and ind in tensor2.inds:
                    if ind not in edges:
                        edges[ind] = edge_counter
                        edge_counter += 1

    # Initialize the structure matrix with zeros
    structure_matrix = np.zeros((num_tensors, edge_counter), dtype=int)

    # Second pass to fill the structure matrix based on the identified connections
    for i, tensor1 in enumerate(tn.tensors):
        for ind_pos, ind in enumerate(tensor1.inds[:-1]):  # Ignore the last index
            if ind in edges:
                edge_idx = edges[ind]
                dimensionality_value = ind_pos + phys_dim  # Calculate dimensionality value
                structure_matrix[i, edge_idx] = dimensionality_value  # Assign dimensionality value
                # Find the other tensor connected by this index and assign dimensionality
                for j, tensor2 in enumerate(tn.tensors):
                    if i != j and ind in tensor2.inds:
                        ind_pos2 = tensor2.inds.index(ind)
                        dimensionality_value2 = ind_pos2 + phys_dim
                        structure_matrix[j, edge_idx] = dimensionality_value2

    return structure_matrix


def generate_tensor_network(Lx=None, Ly=None, bond_dim=4, phys_dim=2, pbc=False, edges = None, J = None, dtype=torch.cfloat, normalize = False):
    """
    Generate a tensor network (MPS or PEPS) with optional periodic boundary conditions,
    along with its structure matrix.

    Parameters:
    - Lx (int): The length of the tensor network; number of sites for MPS, or width for PEPS.
    - Ly (int, optional): The height for PEPS. If not provided or Ly=1, generates an MPS.
    - bond_dim (int): The bond dimension for the tensor network.
    - pbc (bool): Whether to use periodic boundary conditions. Default is False (OBC).
    
    Returns:
    - tn (TensorNetwork): The generated tensor network (MPS or PEPS).
    - sm (numpy.ndarray): The structure matrix of the tensor network.
    - network_type (str): 'MPS' or 'PEPS' indicating the type of the generated network.
    - edges (list): List of edges in the tensor network, if none is provided, either 
    """

    dtype_str = dtype.__str__().split('.')[1]

    if Ly == 1:
        Ly = None
    
    if Lx==1 and Ly is not None:
        Lx = Ly
        Ly = 1

    if edges is None:
        if (Ly is None or Ly == 1):
            # Generate an MPS
            #TODO: Investigate effect of Normalize on getting NaN values for large systems
            tn = qtn.MPS_rand_state(Lx, bond_dim=bond_dim, phys_dim=phys_dim, cyclic=pbc, dtype=dtype_str, normalize=normalize)
            network_type = 'MPS'
        elif (Ly is not None and Lx is not None) and (Ly > 1 and Lx > 1):
            # Generate a PEPS
            if not pbc:
                tn = qtn.PEPS.rand(Lx, Ly, bond_dim=bond_dim, phys_dim=phys_dim, dtype=dtype_str)
                if normalize:
                    tn = tn.normalize()
            else: 
                tn = create_cyclic_peps(Lx, Ly, bond_dim = bond_dim, phys_dim=phys_dim, dtype=dtype_str)

            network_type = 'PEPS'
        else:
            raise ValueError("Invalid combination of Lx and Ly")
    else:
        for (a,b), J_ab in zip(edges, J):
            a, b, J_ab = int(a), int(b), float(J_ab)
            if J_ab != 0:
                edges.append((a, b))
        tn = qtn.TN_from_edges_rand(edges, D=bond_dim, phys_dim=phys_dim, dtype=dtype_str)
        if normalize:
            raise NotImplementedError("Normalization for TN from edges is not implemented yet")
        network_type = 'Generic'
    sm = generate_structure_matrix(tn)

    tn_tensors_to_torch(tn, dtype=dtype)
                
    return tn, sm, network_type

# This function is used to perform a dynamic tensor multiplication using einsum
# It takes two tensors and a tuple of lists of dimensions to multiply
# Dims has to be a tuple of lists ([dims1], [dims2]) where dims1 and dims2 are the dimensions to multiply
# If batched is set to True, the function will perform a batched multiplication, where the first dimension of the tensors must match
def dynamic_tensor_mult(tensor1, tensor2, dims, batched=False):    
    assert type(dims) == tuple and len(dims) == 2, "dims must be a tuple of lists ([dims1], [dims2])"
    dims1, dims2 = dims
    # Create einsum path dynamically based on the tensor dimensions and specified dims for multiplication
    path1 = ''.join([f"{chr(97+i)}" for i in range(tensor1.dim())])
    path2 = ''.join([f"{chr(97+tensor1.dim()+i)}" for i in range(tensor2.dim())])

    if batched:
        assert tensor1.shape[0] == tensor2.shape[0], "Batch dimensions must match for batched multiplication"
        batch_dim_1 = path1[0]
        batch_dim_2 = path2[0]
    
    # Adjust path2 based on dims2 to match the dims1 in path1 for multiplication
    for d1, d2 in zip(dims1, dims2):
        path2 = path2[:d2] + path1[d1] + path2[d2+1:]
    
    # Construct the result path excluding the multiplied dimensions
    result_path = ''.join([path1[i] for i in range(tensor1.dim()) if i not in dims1])
    result_path += ''.join([path2[i] for i in range(tensor2.dim()) if path2[i] not in path1])

    if batched: 
        path2 = path2.replace(batch_dim_2, batch_dim_1)
        result_path = result_path.replace(batch_dim_1, '')
        result_path = result_path.replace(batch_dim_2, '')
        result_path = batch_dim_1 + result_path

    
    einsum_path = f"{path1},{path2}->{result_path}"   
    
    # Perform the operation
    result = torch.einsum(einsum_path, tensor1, tensor2)
    return result



def compute_partial_trace(tensor_nodes, tensor_ind_maps, messages_list, message_inds, output_ind, output_ind_pos, indices):
    #Check if not enough letters to index the tensors
    if len(indices) > 26:
        raise ValueError("Too many indices to index the tensors")
    index_paths = ''.join([f"{chr(97+i)}" for i in range(len(indices))])
    output_path = ''.join([f"{index_paths[idx]}" for idx in output_ind])
    
    tensor_paths = []
    for i, tensor_inds in enumerate(tensor_ind_maps):
        output_pos = output_ind_pos[i]
        tensor_paths.append(''.join([f"{index_paths[indices.index(idx)]}" for idx in tensor_inds]))
        #Change the output path to the correct index
        out_tensor_path = tensor_paths[i][output_pos]
        tensor_paths[i] = tensor_paths[i].replace(out_tensor_path, output_path[i])
        
    message_paths = []
    for message_inds in message_inds:
        message_paths.append(''.join([f"{index_paths[idx]}" for idx in message_inds]))

    tensor_paths = ','.join(tensor_paths)
    message_paths = ','.join(message_paths)

    if len(message_paths) > 0:
        einsum_str = f"{tensor_paths},{message_paths}->{output_path}"
        operands = tensor_nodes + messages_list
    else:
        einsum_str = f"{tensor_paths}->{output_path}"
        operands = tensor_nodes

    try:
        result = torch.einsum(einsum_str, *operands)
        return result
    except RuntimeError as e:
        print(f"Error in einsum: {e}")
        print(f"Einsum string: {einsum_str}")
        print(f"Operands: {operands}")
        #Print Dimensions of each operand
        for op in operands:
            print(f"Operand shape: {op.shape}")
        raise e


def visualize_tensor_network(tn):
    """
    Visualize a tensor network using quimb's built-in graphing functionality, with a title.

    Parameters:
    - tn (TensorNetwork): The tensor network to visualize.
    """
    tn.draw()

def unbuild_dual_tn(dual_tn, tn_type = None):
    """
    Obtain the original TN from the dual TN, splitting the contracted physical indices,
    and removing the complex conjugate tensors.
    
    Parameters:
    - dual_tn (TensorNetwork): The dual tensor network.
    
    Returns:
    - tn (TensorNetwork): The original tensor network.
    """
    # Create a copy of the dual TN to avoid modifying it
    tn = dual_tn.copy()

    # Identify and remove the complex conjugate tensors
    conjugate_tensor_tags = [tag for tag in tn.tags if '*' in tag]
    tn.delete(tags=conjugate_tensor_tags, which='any')

    if tn_type is not None:
        if tn_type == 'MPS':
            qtn.MatrixProductState.from_TN(tn)
        elif tn_type == 'PEPS':
            qtn.PEPS.from_TN(tn)

    return tn

def build_dual_tn(tn, contract_phys_ind=False):
    """
    Create the dual TN obtained by stacking the TN with its compllex conkugate,
    and contracting the physical indices.
    
    
    Parameters:
    - tn (TensorNetwork): The tensor network to visualize.
    
    Returns:
    - tn_dual (TensorNetwork): The dual tensor network.
    """
    tn_conj = tn.H.copy()

    # Iterate over the tensors in the conjugate network and update their tags
    for tensor in tn_conj:
        new_tags = {tag + '*' for tag in tensor.tags}
        tensor.modify(tags=new_tags)

    # Now combine the updated conjugate network with the original
    tn_dual = tn & tn_conj

    if contract_phys_ind:
        contract_physical_indices(tn_dual)
    return tn_dual

def contract_physical_indices(tn):
    physical_indices = [ind for ind in tn.ind_map if ind.startswith('k')]
    for ind in physical_indices:
        tn.contract_ind(ind)
    return

def tn_tensors_to_torch(tn, dtype=torch.float32):
    """
    Convert all the tensors in the TN to torch tensors through an in-place operation.
    
    Parameters:
    - tn (TensorNetwork): The tensor network to convert.
    - dtype (torch.dtype): The data type for the torch tensors.
    """
    tn.apply_to_arrays(lambda x: torch.tensor(x, dtype=dtype))

def are_equal(tn1, tn2):
    """
    Check if two tensor networks are equal by comparing their tensors and index maps.
    
    Parameters:
    - tn1 (TensorNetwork): The first tensor network.
    - tn2 (TensorNetwork): The second tensor network.
    
    Returns:
    - bool: True if the tensor networks are equal, False otherwise.
    """
    # Check if the index maps are equal
    # Check the keys of the index maps, and then check the values

    if tn1.ind_map != tn2.ind_map:
        return False

    # Check if the tensors are equal
    for tensor1, tensor2 in zip(tn1, tn2):
        if not (tensor1.data == tensor2.data).all() or not tensor1.inds == tensor2.inds or not tensor1.tags == tensor2.tags or not tensor1.dtype == tensor2.dtype:
            return False

    return True

if __name__ == "__main__":
    #Create sample TN from edge pairs
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 0)]
    # random J values
    J = np.random.rand(len(edges))
    tn_gen, sm_gen, _ = generate_tensor_network(edges=edges, J=J)

    # Generate and visualize a PEPS with OBC
    tn_peps, sm_peps, _ = generate_tensor_network(5, bond_dim=3, pbc=False)

    # Generate and visualize a PEPS with OBC
    tn_peps_pbc, sm_peps_pbc, _ = generate_tensor_network(2, 2, bond_dim=3, pbc=True)

    # Generate and visualize an MPS with PBC
    tn_mps_pbc, sm_mps_pbc, _ = generate_tensor_network(2, bond_dim=3, pbc=True)

    # Generate and visualize an MPS with PBC
    tn_mps, sm_mps, _ = generate_tensor_network(2, bond_dim=3, pbc=False)

    test_tn = tn_peps

    # Visualize one of the generated tensor networks and its dual
    dual_tn = build_dual_tn(test_tn)

    visualize_tensor_network(test_tn)
    visualize_tensor_network(dual_tn)

    dual_tn_copy = dual_tn.copy()

    contract_physical_indices(dual_tn_copy)
    visualize_tensor_network(dual_tn_copy) 
       
    tn_orig = unbuild_dual_tn(dual_tn)


    # Check if the original and unbuild TNs are equal by comparing their tensors and index maps
    assert are_equal(test_tn, tn_orig)


    print("Done!")



   
    
    



    