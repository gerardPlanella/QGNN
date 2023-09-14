import functools
import itertools
import numpy as np

def su(n, include_identity=False, skip_complex=False, normalize=False):
    """ The Lie algebra associated with the Lie group SU(n). By default, it returns the n^2-1 generators (traceless Hermitian matrices) of the group. Use `include_identity = True` and `skip_complex = False` to return a complete orthogonal basis of hermitian `n x n` matrices.

    Parameters
        n (int): The dimension of the matrices.
        include_identity (bool, optional): If True, include the identity matrix in the basis (default: False).
        skip_complex (bool, optional): If True, skip the complex matrices (default: False).
        normalize (bool, optional): If True, normalize the matrices to have norm 1 (default: False).

    Returns
        list[ np.ndarray ]: A list of `n^2-1` matrices that form a basis of the Lie algebra.
    """
    base = np.zeros((n,n), dtype=complex)

    basis = []

    if include_identity:
        identity = base.copy()
        for i in range(n):
            identity[i,i] = 1
        if normalize:
            # factor 2 to get norm sqrt(2), too
            identity = np.sqrt(2/n) * identity
        basis.append(identity)

    # Generate the off-diagonal matrices
    for i in range(n):
        for j in range(i+1, n):
            m = base.copy()
            m[i,j] = 1
            m[j,i] = 1
            basis.append(m)

            if skip_complex:
                continue
            m = base.copy()
            m[i, j] = -1j
            m[j, i] = 1j
            basis.append(m)

    # Generate the diagonal matrices
    for i in range(1,n):
        m = base.copy()
        for j in range(i):
            m[j,j] = 1
        m[i,i] = -i
        if i > 1:
            m = np.sqrt(2/(i*(i+1))) * m
        basis.append(m)

    if normalize:
        # su have norm sqrt(2) by default
        basis = [m/np.sqrt(2) for m in basis]
    return np.array(basis)

def pauli_basis(n, include_identity=True, skip_complex=False, normalize=False):
    """ Generate the pauli basis of hermitian 2**n x 2**n matrices. This basis is orthonormal and, except for the identity, traceless.

    E.g. for n = 2, the basis is [II, IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ]

    Parameters
        n (int): Number of qubits
        include_identity (bool): Whether to include the identity matrix (default False)
        skip_complex (bool): Whether to skip the complex matrices (default False)
        normalize (bool): Whether to normalize the basis elements (default False)

    Returns
        list[ np.ndarray ]: The pauli basis
    """
    norm_factor = 1/np.sqrt(2**n) if normalize else 1
    def reduce_norm(l):
        # apply normalization to the first element, and reduce the rest
        first = norm_factor * l[0]
        if len(l) == 1:
            return first
        rest = functools.reduce(np.kron, l[1:])
        return np.kron(first, rest)

    basis = []
    I,X,Y,Z = su(2, include_identity=True)
    for i in itertools.product([I,X,Y,Z], repeat=n):
        # if skip_complex, and Y appears an odd number of times, skip
        if skip_complex and sum([m is Y for m in i]) % 2 == 1:
            continue
        basis_el = reduce_norm(i)
        basis.append(basis_el)

    # if not include_identity, remove the identity
    if not include_identity:
        basis = basis[1:]

    return np.array(basis)

basis_dict = {
    ('pauli', 2): pauli_basis(1, skip_complex=True, include_identity=True),
    ('pauli', 3): pauli_basis(1, skip_complex=False, include_identity=True),
    ('pauli', 9): pauli_basis(2, skip_complex=True, include_identity=True),
    ('pauli', 15): pauli_basis(2, skip_complex=False, include_identity=True),
    ('su', 2): su(2, skip_complex=True, include_identity=True),
    ('su', 3): su(2, skip_complex=False, include_identity=True),
    ('su', 9): su(4, skip_complex=True, include_identity=True),
    ('su', 15): su(4, skip_complex=False, include_identity=True)
}

def rdm_from_bloch_vec(bloch_vecs, basis='pauli'):
    """ This function converts a bloch vector of 1-RDMs or 2-RDMs into the corresponding RDM.

    Parameters
        bloch_vec (np.ndarray): The bloch vector of the RDM. The first dimension is the batch dimension.
        basis (str): The basis of the bloch vector. Either 'su' or 'pauli'.

    Returns
        np.ndarray: The RDM corresponding to the given bloch vector.
    """
    if len(bloch_vecs.shape) == 1:
        bloch_vecs = bloch_vecs[None,:]  # add a batch dimension

    if type(basis) == str:
        basis = basis_dict[(basis, bloch_vecs.shape[-1])]
    elif not (type(basis) == np.ndarray and len(basis)-1 == bloch_vecs.shape[-1]):
        raise ValueError(f"Invalid basis: {type(basis)}")

    # Create the RDM: 1/2 * (I + sum_i vec_i * basis_i)
    n = int(np.log2(basis[0].shape[0]))
    rdms = 2**(-n) * (basis[0] + np.einsum('...j,jkl->...kl', bloch_vecs, basis[1:]))
    return rdms.squeeze()

def is_rdm(rdm):
    """ Check whether the given object is a valid RDM. """
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
    vec = 0.1*np.random.rand(10, 2)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))
    vec = 0.1*np.random.rand(10, 3)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))
    vec = 0.1*np.random.rand(10, 9)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))
    vec = 0.1*np.random.rand(10, 15)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli')))
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='su')))

    # other batch sizes
    vec = 0.1*np.random.rand(15)
    assert is_rdm(rdm_from_bloch_vec(vec, basis='pauli'))
    vec = 0.1*np.random.rand(1, 2)
    assert is_rdm(rdm_from_bloch_vec(vec, basis='su'))
    vec = 0.1*np.random.rand(42, 10, 3)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis='pauli').reshape(-1, 2, 2)))

    # custom basis
    basis = pauli_basis(5)
    vec = 0.01*np.random.rand(10, len(basis)-1)
    assert all(map(is_rdm, rdm_from_bloch_vec(vec, basis=basis)))

    print("All tests passed.")
