import functools
import itertools
import torch
import numpy as np


def su(n, include_identity=False, skip_complex=False, normalize=False, device='cpu'):
    base = torch.zeros((n, n), dtype=torch.float32 if skip_complex else torch.complex64, device=device)

    basis = []

    if include_identity:
        identity = base.clone()
        for i in range(n):
            identity[i, i] = 1
        if normalize:
            identity = np.sqrt(2 / n) * identity
        basis.append(identity)

    for i in range(n):
        for j in range(i + 1, n):
            m = base.clone()
            m[i, j] = 1
            m[j, i] = 1
            basis.append(m)

            if not skip_complex:
                m = base.clone()
                m[i, j] = -1j
                m[j, i] = 1j
                basis.append(m)

    for i in range(1, n):
        m = base.clone()
        for j in range(i):
            m[j, j] = 1
        m[i, i] = -i
        if i > 1:
            m = np.sqrt(2 / (i * (i + 1))) * m
        basis.append(m)

    if normalize:
        basis = [m / np.sqrt(2) for m in basis]

    return torch.stack(basis)

def pauli_basis(n, include_identity=True, skip_complex=False, normalize=False, device='cpu'):
    norm_factor = 1 / np.sqrt(2 ** n) if normalize else 1

    def reduce_norm(l):
        first = norm_factor * l[0]
        if len(l) == 1:
            return first
        rest = functools.reduce(torch.kron, l[1:])
        return torch.kron(first, rest)

    basis = []
    I, X, Y, Z = su(2, include_identity=True, device=device)
    for i in itertools.product([I, X, Y, Z], repeat=n):
        if skip_complex and sum([m is Y for m in i]) % 2 == 1:
            continue
        basis_el = reduce_norm(i)
        if skip_complex:
            basis_el = basis_el.real
        basis.append(basis_el)

    if not include_identity:
        basis = basis[1:]

    return torch.stack(basis)

def initialize_basis_dict(device='cpu'):
    return {
        ('pauli', 2): pauli_basis(1, skip_complex=True, include_identity=True, device=device),
        ('pauli', 3): pauli_basis(1, skip_complex=False, include_identity=True, device=device),
        ('pauli', 9): pauli_basis(2, skip_complex=True, include_identity=True, device=device),
        ('pauli', 15): pauli_basis(2, skip_complex=False, include_identity=True, device=device),
        ('su', 2): su(2, skip_complex=True, include_identity=True, device=device),
        ('su', 3): su(2, skip_complex=False, include_identity=True, device=device),
        ('su', 9): su(4, skip_complex=True, include_identity=True, device=device),
        ('su', 15): su(4, skip_complex=False, include_identity=True, device=device)
    }

def rdm_from_bloch_vec(bloch_vecs, basis_type='pauli', basis_dict=None):
    if basis_dict is None:
        basis_dict = initialize_basis_dict(bloch_vecs.device)
    basis = basis_dict[(basis_type, bloch_vecs.shape[-1])]

    n = int(np.log2(basis[0].shape[0]))
    bloch_vecs = bloch_vecs.type(basis.dtype)
    rdms = 2 ** (-n) * (basis[0] + torch.einsum('...j,jkl->...kl', bloch_vecs, basis[1:]))
    return rdms

def is_rdm(rdm):
    """ Check whether the given object is a valid RDM. """
    # cast to numpy
    rdm = rdm.numpy()
    if rdm.shape[0] != rdm.shape[1]:
        return False
    # hermitian
    if not np.allclose(rdm, rdm.conj().T):
        return False
    # non-negative eigenvalues
    if not np.all(np.linalg.eigvalsh(rdm) >= 0):
        return False
    # trace normalized
    if not np.allclose(np.trace(rdm), 1):
        return False
    return True

def torch_func_svd(mat, func):
    U, S, V = torch.linalg.svd(mat)
    return U @ torch.diag_embed(func(S)) @ V.transpose(-2,-1)

def torch_func_eigh(hmat, func):
    D, U = torch.linalg.eigh(hmat)
    return U @ torch.diag_embed(func(D)) @ U.transpose(-2,-1)

def trace_distance(rdms1, rdms2):
    """ Compute the trace distance between two sets of RDMs in a batched manner. """
    delta = rdms1 - rdms2
    _, s, _ = torch.linalg.svd(delta)
    trace_dist = torch.sum(s, dim=-1) / 2  # divide by 2 because the definition involves 1/2 Tr(|A|)
    
    return torch.mean(trace_dist)

def compute_trace_loss(pred_rdms, epsilon=1e-10):
    trace_diff = torch.abs(torch.linalg.matrix_norm(pred_rdms, ord='nuc', dim=(-2, -1)) - 1)
    trace_loss = torch.maximum(trace_diff - epsilon, torch.zeros_like(trace_diff))
    return torch.mean(trace_loss)

def compute_psd_penalty(pred_rdms, epsilon=1e-10):
    eigvals = torch.linalg.eigvalsh(pred_rdms)
    negative_eigvals = torch.minimum(eigvals, torch.zeros_like(eigvals))
    psd_penalty = torch.sum(negative_eigvals ** 2, dim=-1)
    return torch.mean(psd_penalty)

def quantum_loss(rdms1, rdms2, distance_loss, alpha=0.5, beta = 1e-2, epsilon=1e-10):
    psd_loss = (alpha/beta) * compute_psd_penalty(rdms1, epsilon=epsilon)
    trace_loss = ((1-alpha)/beta) * compute_trace_loss(rdms1, epsilon=epsilon)
    loss = distance_loss(rdms1, rdms2)
    return loss + psd_loss + trace_loss


def squared_hilbert_schmidt_distance(rdms1, rdms2):
    delta_squared = (rdms1 - rdms2) ** 2
    hs_distance_squared = torch.sum(delta_squared, dim=(-2, -1))
    return torch.mean(hs_distance_squared)

def conditional_fidelity_loss(rdms1, rdms2, high_penalty=10, epsilon=1e-10):
    # Check for PSD
    eigvals1 = torch.linalg.eigvalsh(rdms1)
    eigvals2 = torch.linalg.eigvalsh(rdms2)

    is_psd1 = torch.all(eigvals1 > -epsilon, dim=-1)
    is_psd2 = torch.all(eigvals2 > -epsilon, dim=-1)
    is_psd = is_psd1 & is_psd2


    fidelity_loss = torch.zeros_like(is_psd1, dtype=torch.float)

    # Calculate penalties for non-PSD RDMs
    penalties1 = torch.sum(torch.minimum(eigvals1, torch.zeros_like(eigvals1))**2, dim=-1)
    penalties2 = torch.sum(torch.minimum(eigvals2, torch.zeros_like(eigvals2))**2, dim=-1)
    penalties = high_penalty + penalties1 + penalties2

    # Compute fidelity only for PSD pairs
    psd_mask = is_psd

    print("psd:",psd_mask.sum().item(),"/",psd_mask.numel())

    if psd_mask.any():
        fidelity_loss[psd_mask] = fidelity(rdms1[psd_mask], rdms2[psd_mask])
    
    fidelity_loss[~psd_mask] = penalties[~psd_mask]
    return torch.mean(fidelity_loss)

def to_undirected_and_remove_reverses(edge_index, device):
    # Step 1: Combine and sort node pairs to create a canonical representation
    sorted_edges, _ = torch.sort(edge_index, dim=0)
    
    # Step 2: Remove duplicates
    # Convert sorted edges to a set of tuples to identify unique edges easily
    edge_set = {tuple(edge.cpu().numpy()) for edge in sorted_edges.t()}
    unique_edges = torch.tensor(list(edge_set), dtype=edge_index.dtype).t().to(device)

    # Optional: Sort the unique edges for consistency and easier readability
    _, perm = unique_edges.sort(dim=1)
    unique_edges = unique_edges[:, perm[0]]

    return unique_edges

def calculate_energy_batched(one_rdms, two_rdms, node_params, edge_params, batch_index, edge_index):
    # #Convert edge_index to undirected edges by removing reverse edges
    # edge_index = to_undirected_and_remove_reverses(edge_index, one_rdms.device)

    # Imaginary part is 0
    pauli_z = torch.Tensor([[1.0, 0.0], [0.0, -1.0]]).to(dtype=one_rdms.dtype).to(one_rdms.device)
    pauli_x = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).to(dtype=one_rdms.dtype).to(one_rdms.device)
    

    # one_rdms = torch.Tensor(one_rdms).to(dtype=torch.float64)
    # two_rdms = torch.Tensor(two_rdms).to(dtype=torch.float64)
    node_params = torch.Tensor(node_params).to(dtype=one_rdms.dtype).to(one_rdms.device)
    edge_params = torch.Tensor(edge_params).to(dtype=one_rdms.dtype).to(one_rdms.device)
    
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

    local_energy = local_energy.real.to(torch.float64)

    batch_local_energies.scatter_add_(0, batch_energy_node_idx, local_energy)

    # Now we need to add the interaction energy to the correct batch, so we need to find the correct batch for each edge
    batch_interaction_energies = torch.zeros_like(batch_local_energies, dtype=torch.float64)
    # First we need to find the batch for each node in the edge
    batch_edge_index = batch_index[edge_index][0, :]
    # Now we need to find the unique batch for each edge, we know that no graph will connect edges between two batches
    unique_batches, batch_edge_idx = torch.unique(batch_edge_index, return_inverse=True)

    interaction_energy = interaction_energy.real.to(torch.float64)

    batch_interaction_energies.scatter_add_(0, batch_edge_idx, interaction_energy)

    batch_energies = batch_local_energies + batch_interaction_energies

    return batch_energies, unique_batches



# def calculate_energy_batched(one_rdms, two_rdms, node_params, edge_params, batch_index, edge_index):
#     # Imaginary part is 0
#     pauli_z = torch.Tensor([[1.0, 0.0], [0.0, -1.0]]).to(dtype=torch.float64).to(one_rdms.device)
#     pauli_x = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).to(dtype=torch.float64).to(one_rdms.device)
    

#     one_rdms = torch.Tensor(one_rdms).to(dtype=torch.float64)
#     two_rdms = torch.Tensor(two_rdms).to(dtype=torch.float64)
#     node_params = torch.Tensor(node_params).to(dtype=torch.float64)
#     edge_params = torch.Tensor(edge_params).to(dtype=torch.float64)
    
#     # Multiply each 1-RDM by pauli_z and pauli_x multplying with the correct parameter
#     one_rdms_pauli_z = torch.einsum('bij,jk->bik', one_rdms, pauli_z).to(one_rdms.device)
#     one_rdms_pauli_x = torch.einsum('bij,jk->bik', one_rdms, pauli_x).to(one_rdms.device)

#     # Perform trace and multiply with correct parameter
#     local_energy = torch.einsum('bii,b->b', one_rdms_pauli_x, node_params[:, 1]) \
#         + torch.einsum('bii,b->b', one_rdms_pauli_z, node_params[:, 0])
    
#     two_rdms_kron_pauli_z = torch.einsum('bij,jk->bik', two_rdms, torch.kron(pauli_z, pauli_z))

#     if len(edge_params.squeeze().size()) == 0:
#         interaction_energy = torch.einsum('bii,b->b', two_rdms_kron_pauli_z, edge_params.squeeze(0))
#     else:
#         interaction_energy = torch.einsum('bii,b->b', two_rdms_kron_pauli_z, edge_params.squeeze())
    

#     unique_batches, batch_energy_node_idx = torch.unique(batch_index, return_inverse=True)
#     batch_local_energies = torch.zeros_like(unique_batches, dtype=torch.float64)

#     batch_local_energies.scatter_add_(0, batch_energy_node_idx, local_energy)

#     # Now we need to add the interaction energy to the correct batch, so we need to find the correct batch for each edge
#     batch_interaction_energies = torch.zeros_like(batch_local_energies, dtype=torch.float64)
#     # First we need to find the batch for each node in the edge
#     batch_edge_index = batch_index[edge_index][0, :]
#     # Now we need to find the unique batch for each edge, we know that no graph will connect edges between two batches
#     unique_batches, batch_edge_idx = torch.unique(batch_edge_index, return_inverse=True)

#     batch_interaction_energies.scatter_add_(0, batch_edge_idx, interaction_energy)

#     batch_energies = batch_local_energies + batch_interaction_energies

#     return batch_energies, unique_batches




# same as above, but allow for batched RDMs
def fidelity(rdms1, rdms2):
    """ Compute the fidelity between two RDMs. """
    # using 3x svd is more numerically stable, but leads to exploding gradients (because singular values are not bounded by the trace constraint!)
    # rdms1_sqrt = torch_func_svd(rdms1, torch.sqrt)
    # rdms2_sqrt = torch_func_svd(rdms2, torch.sqrt)
    # fidelities = torch.linalg.norm(rdms1_sqrt @ rdms2_sqrt, ord='nuc', dim=(-2,-1))**2
    # # print(rdms1.shape, torch.max(fidelities))
    # return torch.mean(fidelities)

    # rdms might be non-PSD -> nan in sqrt -> gradient can't be computed
    # rdms1_sqrt = torch_func_eigh(rdms1, torch.sqrt)
    # evs = torch.linalg.eigvalsh(rdms1_sqrt @ rdms2 @ rdms1_sqrt)

    evs = torch.linalg.eigvals(rdms1 @ rdms2)  # this is correct, see https://arxiv.org/abs/2309.10565
    fidelities = torch.sum(torch.sqrt(evs), dim=-1).abs()**2
    #print("evs:",evs,"\nfids:", fidelities)
    return torch.sum(fidelities)
    # check there is no nan in the RDMs
    assert torch.all(torch.isfinite(rdms1)), f"{torch.sum(torch.isnan(rdms1))}/{torch.numel(rdms1)} elements of rdm1 and {torch.sum(torch.isinf(rdms1))}/{torch.numel(rdms1)} elements of rdm2 are nan/inf"

if __name__ == '__main__':
    # Test su and pauli_basis
    assert len(su(2**2, include_identity=True)) == len(pauli_basis(2, include_identity=True))
    I,X,Y,Z = su(2, include_identity=True)
    assert len(su(2)) == 3
    assert len(pauli_basis(1, include_identity=False)) == 3
    assert len(su(2, skip_complex=True)) == 2
    assert len(pauli_basis(1, skip_complex=True, include_identity=False)) == 2
    assert len(su(4)) == 15
    assert len(pauli_basis(2, include_identity=False)) == 15
    assert len(su(4, skip_complex=True)) == 9
    assert len(pauli_basis(2, skip_complex=True, include_identity=False)) == 9

    # Test rdm_from_bloch_vec
    vec = 0.1*torch.rand(10, 2)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))
    vec = 0.1*torch.rand(10, 3)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))
    vec = 0.1*torch.rand(10, 9)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))
    vec = 0.1*torch.rand(10, 15)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))

    # other batch sizes
    vec = 0.1*torch.rand(15)
    assert is_rdm(rdm_from_bloch_vec(vec, basis='pauli'))
    vec = 0.1*torch.rand(1, 2)
    assert is_rdm(rdm_from_bloch_vec(vec, basis='su')[0])
    vec = 0.1*torch.rand(42, 10, 3)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli').reshape(-1, 2, 2)))

    # custom basis
    basis = pauli_basis(5)
    vec = 0.01*torch.rand(10, len(basis)-1)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis=basis)))

    print("All tests passed.")
