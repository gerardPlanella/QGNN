import torch
import math
import numpy as np
import scipy.sparse as sp
from functools import reduce



matmap_np, matmap_sp = None, None  # global variables for the matrix map, so we don't have to recompute it every time and can use it outside of the function

def parse_hamiltonian(hamiltonian, sparse=False, scaling=1, buffer=None, max_buffer_n=0, dtype=float): # I'd usually default to complex, but because we're only dealing with Ising models here, float is more handy
    """Parse a string representation of a Hamiltonian into a matrix representation. The result is guaranteed to be Hermitian.

    Parameters:
        hamiltonian (str): The Hamiltonian to parse.
        sparse (bool): Whether to use sparse matrices (csr_matrix) or dense matrices (numpy.array).
        scaling (float): A constant factor to scale the Hamiltonian by.
        buffer (dict): A dictionary to store calculated chunks in. If `None`, it defaults to the global `matmap_np` (or `matmap_sp` if `sparse == True`). Give `buffer={}` and leave `max_buffer_n == 0` (default) to disable the buffer.
        max_buffer_n (int): The maximum length (number of qubits) for new chunks to store in the buffer (default: 0). If `0`, no new chunks will be stored in the buffer.

    Returns:
        numpy.ndarray | scipy.sparse.csr_matrix: The matrix representation of the Hamiltonian.

    Example:
    >>> parse_hamiltonian('0.5*(XX + YY + ZZ + II)') # SWAP
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]
           [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('-(XX + YY + .5*ZZ) + 1.5')
    array([[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
           [ 0.+0.j  2.+0.j -2.+0.j  0.+0.j]
           [ 0.+0.j -2.+0.j  2.+0.j  0.+0.j]
           [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]])
    >>> parse_hamiltonian('0.5*(II + ZI - ZX + IX)') # CNOT

    """
    kron = sp.kron if sparse else np.kron

    # Initialize the matrix map
    global matmap_np, matmap_sp
    if matmap_np is None or matmap_sp is None or matmap_np["I"].dtype != dtype:
        # numpy versions
        matmap_np = {
            "H": np.array([[1, 1], [1, -1]], dtype=dtype) / np.sqrt(2),
            "X": np.array([[0, 1], [1, 0]], dtype=dtype),
            "Z": np.array([[1, 0], [0, -1]], dtype=dtype),
            "I": np.array([[1, 0], [0, 1]], dtype=dtype),
        }
        # composites
        matmap_np.update({
            "ZZ": np.kron(matmap_np['Z'], matmap_np['Z']),
            "IX": np.kron(matmap_np['I'], matmap_np['X']),
            "XI": np.kron(matmap_np['X'], matmap_np['I']),
            "YY": np.array([[ 0,  0,  0, -1],  # to avoid complex numbers
                            [ 0,  0,  1,  0],
                            [ 0,  1,  0,  0],
                            [-1,  0,  0,  0]], dtype=dtype)
        })
        for i in range(2, 11):
            matmap_np["I"*i] = np.eye(2**i, dtype=dtype)
        # add 'Y' only if dtype supports imaginary numbers
        if np.issubdtype(dtype, np.complexfloating):
            matmap_np["Y"] = np.array([[0, -1j], [1j, 0]], dtype=dtype)

        # sparse versions
        matmap_sp = {k: sp.csr_array(v) for k, v in matmap_np.items()}
    
    if not np.issubdtype(dtype, np.complexfloating) and "Y" in hamiltonian:
        raise ValueError(f"The Pauli matrix Y is not supported for dtype {dtype.__name__}.")

    matmap = matmap_sp if sparse else matmap_np

    # only use buffer if pre-computed chunks are available or if new chunks are allowed to be stored
    use_buffer = buffer is None or len(buffer) > 0 or max_buffer_n > 0
    if use_buffer and buffer is None:
        buffer = matmap

    def calculate_chunk_matrix(chunk, sparse=False, scaling=1):
        # if scaling != 1:  # only relevant for int dtype
            # scaling = np.array(scaling, dtype=dtype)
        if use_buffer:
            if chunk in buffer:
                return buffer[chunk] if scaling == 1 else scaling * buffer[chunk]
            if len(chunk) == 1:
                return matmap[chunk[0]] if scaling == 1 else scaling * matmap[chunk[0]]
            # Check if a part of the chunk has already been calculated
            for i in range(len(chunk)-1, 1, -1):
                for j in range(len(chunk)-i+1):
                    subchunk = chunk[j:j+i]
                    if subchunk in buffer:
                        # If so, calculate the rest of the chunk recursively
                        parts = [chunk[:j], subchunk, chunk[j+i:]]
                        # remove empty chunks
                        parts = [c for c in parts if c != ""]
                        # See where to apply the scaling
                        shortest = min(parts, key=len)
                        # Calculate each part recursively
                        for i, c in enumerate(parts):
                            if c == subchunk:
                                if c == shortest:
                                    parts[i] = scaling * buffer[c]
                                    shortest = ""
                                else:
                                    parts[i] = buffer[c]
                            else:
                                if c == shortest:
                                    parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=scaling)
                                    shortest = ""
                                else:
                                    parts[i] = calculate_chunk_matrix(c, sparse=sparse, scaling=1)
                        return reduce(kron, parts)

        # Calculate the chunk matrix gate by gate
        if use_buffer and len(chunk) <= max_buffer_n:
            gates = [matmap[gate] for gate in chunk]
            chunk_matrix = reduce(kron, gates)
            buffer[chunk] = chunk_matrix
            if scaling != 1:
                chunk_matrix = scaling * chunk_matrix
        else:
            gates = [scaling * matmap[chunk[0]]] + [matmap[gate] for gate in chunk[1:]]
            chunk_matrix = reduce(kron, gates)

        return chunk_matrix

    # Remove whitespace
    hamiltonian = hamiltonian.replace(" ", "")
    # replace - with +-, except before e
    hamiltonian = hamiltonian \
                    .replace("-", "+-") \
                    .replace("e+-", "e-") \
                    .replace("(+-", "(-")

    # print("parse_hamiltonian: Pre-processed Hamiltonian:", hamiltonian)

    # Find parts in parentheses
    part = ""
    parts = []
    depth = 0
    current_part_weight = ""
    for i, c in enumerate(hamiltonian):
        if c == "(":
            if depth == 0:
                # for top-level parts search backwards for the weight
                weight = ""
                for j in range(i-1, -1, -1):
                    if hamiltonian[j] in ["("]:
                        break
                    weight += hamiltonian[j]
                    if hamiltonian[j] in ["+", "-"]:
                        break
                weight = weight[::-1]
                if weight != "":
                    current_part_weight = weight
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                part += c
                parts.append((current_part_weight, part))
                part = ""
                current_part_weight = ""
        if depth > 0: 
            part += c

    # print("Parts found:", parts)

    # Replace parts in parentheses with a placeholder
    for i, (weight, part) in enumerate(parts):
        hamiltonian = hamiltonian.replace(weight+part, f"+part{i}", 1)
        # remove * at the end of the weight
        if weight != "" and weight[-1] == "*":
            weight = weight[:-1]
        if weight in ["", "+", "-"]:
            weight += "1"
        # Calculate the part recursively
        part = part[1:-1] # remove parentheses
        parts[i] = parse_hamiltonian(part, sparse=sparse, scaling=float(weight), buffer=buffer, max_buffer_n=max_buffer_n, dtype=dtype)

    # print("Parts replaced:", parts)

    # Parse the rest of the Hamiltonian
    chunks = hamiltonian.split("+")
    # Remove empty chunks
    chunks = [c for c in chunks if c != ""]
    # If parts are present, use them to determine the number of qubits
    if parts:
        n = int(np.log2(parts[0].shape[0]))
    else: # Use chunks to determine the number of qubits
        n = 0
        for c in chunks:
            if c[0] in ["-", "+"]:
                c = c[1:]
            if "*" in c:
                c = c.split("*")[1]
            if c.startswith("part"):
                continue
            try:
                float(c)
                continue
            except ValueError:
                n = len(c)
                break
        if n == 0:
            print("Warning: Hamiltonian is a scalar!")

    if not sparse and n > 10:
        # check if we would blow up the memory
        mem_required = 2**(2*n) * np.array(1, dtype=dtype).nbytes
        mem_available = psutil.virtual_memory().available
        if mem_required > mem_available:
            raise MemoryError(f"This would blow up you memory ({duh(mem_required)} required)! Try using `sparse=True`.")

    if sparse:
        H = sp.csr_array((2**n, 2**n), dtype=dtype)
    else:
        if n > 10:
            print(f"Warning: Using a dense matrix for a {n}-qubit Hamiltonian is not recommended. Use sparse=True.")
        H = np.zeros((2**n, 2**n), dtype=dtype)

    for chunk in chunks:
        # print("Processing chunk:", chunk)
        chunk_matrix = None
        if chunk == "":
            continue
        # Parse the weight of the chunk
        
        if chunk.startswith("part"):
            weight = 1  # parts are already scaled
            chunk_matrix = parts[int(chunk.split("part")[1])]
        elif "*" in chunk:
            weight = float(chunk.split("*")[0])
            chunk = chunk.split("*")[1]
        elif len(chunk) == n+1 and chunk[0] in ["-", "+"] and n >= 1 and chunk[1] in matmap:
            weight = float(chunk[0] + "1")
            chunk = chunk[1:]
        elif (chunk[0] in ["-", "+", "."] or chunk[0].isdigit()) and all([c not in matmap for c in chunk[1:]]):
            if len(chunk) == 1 and chunk[0] in ["-", "."]:
                chunk = 0
            weight = complex(chunk)
            if np.iscomplex(weight):
                raise ValueError("Complex scalars would make the Hamiltonian non-Hermitian!")
            weight = weight.real
            # weight = np.array(weight, dtype=dtype)  # only relevant for int dtype
            chunk_matrix = np.eye(2**n, dtype=dtype)
        elif len(chunk) != n:
            raise ValueError(f"Gate count must be {n} but was {len(chunk)} for chunk \"{chunk}\"")
        else:
            weight = 1

        if chunk_matrix is None:
            chunk_matrix = calculate_chunk_matrix(chunk, sparse=sparse, scaling = scaling * weight)
        elif scaling * weight != 1:
            chunk_matrix = scaling * weight * chunk_matrix

        # Add the chunk to the Hamiltonian
        # print("Adding chunk", weight, chunk, "for hamiltonian", scaling, hamiltonian)
        # print(type(H), H.dtype, type(chunk_matrix), chunk_matrix.dtype)
        if len(chunks) == 1:
            H = chunk_matrix
        else:
            H += chunk_matrix

    if sparse:
        assert np.allclose(H.data, H.conj().T.data), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H.data}"
    else:
        assert np.allclose(H, H.conj().T), f"The given Hamiltonian {hamiltonian} is not Hermitian: {H}"

    return H

def create_random_rdm_torch(n, complex=True, tol=0, min_eig=1e-10):
    """
    Create a random NxN Reduced Density Matrix (RDM) using PyTorch, with adjustments for complex number handling.
    """
    # Generate a random complex matrix
    A = torch.rand(n, n, dtype=torch.double, requires_grad=True) 

    
    mat_type = torch.float64
    if complex == True:
        mat_type = torch.cdouble
        # Convert to cfloat
        A = A.type(mat_type)
        A = A + 1j*torch.rand(n, n, dtype=mat_type)

    # Create a Hermitian matrix
    A_herm = (A + A.conj().t()) / 2

    # Diagonalize and make positive semi-definite
    eigenvalues, eigenvectors = torch.linalg.eigh(A_herm)
    eigenvalues = torch.clamp(eigenvalues, min=min_eig)
    diag_eigenvalues = torch.diag(eigenvalues.type(mat_type))
    A_pos_semi_def = eigenvectors @ diag_eigenvalues @ eigenvectors.conj().t()

    # Normalize to have trace 1 
    A_rdm = A_pos_semi_def / torch.trace(A_pos_semi_def)

    return A_rdm

def create_random_rdm_torch_batched(n, batch_size, complex=True, tol=0, min_eig=1e-10):
    rdms = []
    for i in range(batch_size):
        rdms.append(create_random_rdm_torch(n, complex=complex, tol=tol, min_eig=min_eig))
    return torch.stack(rdms)


def is_valid_rdm(rdm, tol=0):
    """
    Check if a matrix is a valid Reduced Density Matrix (RDM) with a tolerance for numerical precision.
    """
    # Check Hermitian
    if not torch.allclose(rdm, rdm.conj().t()):
        print("Hermitian check failed")
        return False

    # Check positive semi-definite with a tolerance
    if not torch.all(torch.linalg.eigvalsh(rdm) >= -tol):
        print("Positive semi-definite check failed")
        print(torch.linalg.eigvalsh(rdm))
        return False

    if rdm.dtype == torch.cdouble:
        trace = torch.tensor(1.0 + 0.0j, dtype=torch.cdouble)
    else:
        trace = torch.tensor(1.0, dtype=torch.double)

    # Check unit trace
    if not torch.isclose(torch.trace(rdm), trace):
        print("Unit trace check failed")
        return False
    
    return True


degs = [1, 2, 4, 8, 12, 18]

thetas_dict = {"single": [1.192092800768788e-07,  # m_vals = 1
                          5.978858893805233e-04,  # m_vals = 2
                         #1.123386473528671e-02,
                          5.116619363445086e-02,  # m_vals = 4
                         #1.308487164599470e-01,
                         #2.495289322846698e-01,
                         #4.014582423510481e-01,
                          5.800524627688768e-01,  # m_vals = 8
                         #7.795113374358031e-01,
                         #9.951840790004457e-01,
                         #1.223479542424143e+00,
                          1.461661507209034e+00,  # m_vals = 12
                         #1.707648529608701e+00,
                         #1.959850585959898e+00,
                         #2.217044394974720e+00,
                         #2.478280877521971e+00,
                         #2.742817112698780e+00,
                          3.010066362817634e+00], # m_vals = 18
               "double": [
                          2.220446049250313e-16,  # m_vals = 1
                          2.580956802971767e-08,  # m_vals = 2
                         #1.386347866119121e-05,
                          3.397168839976962e-04,  # m_vals = 4
                         #2.400876357887274e-03,
                         #9.065656407595102e-03,
                         #2.384455532500274e-02,
                          4.991228871115323e-02,  # m_vals = 8
                         #8.957760203223343e-02,
                         #1.441829761614378e-01,
                         #2.142358068451711e-01,
                          2.996158913811580e-01,  # m_vals = 12
                         #3.997775336316795e-01,
                         #5.139146936124294e-01,
                         #6.410835233041199e-01,
                         #7.802874256626574e-01,
                         #9.305328460786568e-01,
                          1.090863719290036e+00]  # m_vals = 18
               }

def matrix_power_two_batch(A, k):
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2**ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2**new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def expm_taylor(A):
    if A.ndimension() < 2 or A.size(-2) != A.size(-1):
        raise ValueError('Expected a square matrix or a batch of square matrices')

    if A.ndimension() == 2:
        # Just one matrix

        # Trivial case
        if A.size() == (1, 1):
            return torch.exp(A)

        if A.element_size() > 4:
            thetas = thetas_dict["double"]
        else:
            thetas = thetas_dict["single"]

        # No scale-square needed
        # This could be done marginally faster if iterated in reverse
        normA = torch.max(torch.sum(torch.abs(A), axis=0)).item()
        for deg, theta in zip(degs, thetas):
            if normA <= theta:
                return taylor_approx(A, deg)

        # Scale square
        s = int(math.ceil(math.log2(normA) - math.log2(thetas[-1])))
        A = A * (2**-s)
        X = taylor_approx(A, degs[-1])
        return torch.matrix_power(X, 2**s)
    else:
        # Batching

        # Trivial case
        if A.size()[-2:] == (1, 1):
            return torch.exp(A)

        if A.element_size() > 4:
            thetas = thetas_dict["double"]
        else:
            thetas = thetas_dict["single"]

        normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values

        # Handle trivial case
        if (normA == 0.).all():
            I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)
            I = I.expand_as(A)
            return I

        # Handle small normA
        more = normA > thetas[-1]
        k = normA.new_zeros(normA.size(), dtype=torch.long)
        k[more] = torch.ceil(torch.log2(normA[more]) - math.log2(thetas[-1])).long()

        # A = A * 2**(-s)
        A = torch.pow(.5, k.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
        X = taylor_approx(A, degs[-1])
        return matrix_power_two_batch(X, k)


def taylor_approx(A, deg):
    batched = A.ndimension() > 2
    I = torch.eye(A.size(-2), A.size(-1), dtype=A.dtype, device=A.device)
    if batched:
        I = I.expand_as(A)

    if deg >= 2:
        A2 = A @ A
    if deg > 8:
        A3 = A @ A2
    if deg == 18:
        A6 = A3 @ A3

    if deg == 1:
        return I + A
    elif deg == 2:
        return I + A + .5*A2
    elif deg == 4:
        return I + A + A2 @ (.5*I + A/6. + A2/24.)
    elif deg == 8:
        # Minor: Precompute
        SQRT = math.sqrt(177.)
        x3 = 2./3.
        a1 = (1.+SQRT)*x3
        x1 = a1/88.
        x2 = a1/352.
        c0 = (-271.+29.*SQRT)/(315.*x3)
        c1 = (11.*(-1.+SQRT))/(1260.*x3)
        c2 = (11.*(-9.+SQRT))/(5040.*x3)
        c4 = (89.-SQRT)/(5040.*x3*x3)
        y2 = ((857.-58.*SQRT))/630.
        # Matrix products
        A4 = A2 @ (x1*A + x2*A2)
        A8 = (x3*A2 + A4) @ (c0*I + c1*A + c2*A2 + c4*A4)
        return I + A + y2*A2 + A8
    elif deg == 12:
        b = torch.tensor(
		[[-1.86023205146205530824e-02,
		  -5.00702322573317714499e-03,
		  -5.73420122960522249400e-01,
		  -1.33399693943892061476e-01],
		  [ 4.6,
		    9.92875103538486847299e-01,
		   -1.32445561052799642976e-01,
		    1.72990000000000000000e-03],
		  [ 2.11693118299809440730e-01,
		    1.58224384715726723583e-01,
		    1.65635169436727403003e-01,
		    1.07862779315792429308e-02],
		  [ 0.,
		   -1.31810610138301836924e-01,
		   -2.02785554058925905629e-02,
		   -6.75951846863086323186e-03]],
        dtype=A.dtype, device=A.device)

        # We implement the following allowing for batches
        #q31 = a01*I+a11*A+a21*A2+a31*A3
        #q32 = a02*I+a12*A+a22*A2+a32*A3
        #q33 = a03*I+a13*A+a23*A2+a33*A3
        #q34 = a04*I+a14*A+a24*A2+a34*A3
        # Matrix products
        #q61 = q33 + q34 @ q34
        #return (q31 + (q32 + q61) @ q61)

        # Example of non-batched version for reference
        #q = torch.stack([I, A, A2, A3]).repeat(4, 1, 1, 1)
        #b = b.unsqueeze(-1).unsqueeze(-1).expand_as(q)
        #q = (b * q).sum(dim=1)
        #qaux = q[2] + q[3] @ q[3]
        #return q[0] + (q[1] + qaux) @ qaux

        q = torch.stack([I, A, A2, A3], dim=-3).unsqueeze_(-4)
        len_batch = A.ndimension() - 2
        # Expand first dimension
        q_size =  [-1 for _ in range(len_batch)] + [4, -1, -1, -1]
        q = q.expand(*q_size)
        b = b.unsqueeze_(-1).unsqueeze_(-1).expand_as(q)
        q = (b * q).sum(dim=-3)
        if batched:
            # Indexing the third to last dimension, because otherwise we
            # would have to prepend as many 1's as the batch shape for the
            # previous expand_as to work
            qaux = q[..., 2,:,:] + q[..., 3,:,:] @ q[..., 3,:,:]
            return q[..., 0,:,:] + (q[..., 1,:,:] + qaux) @ qaux
        else:
            qaux = q[2] + q[3] @ q[3]
            return q[0] + (q[1] + qaux) @ qaux

    elif deg == 18:
        b = torch.tensor(
	[[0.,
	 -1.00365581030144618291e-01,
         -8.02924648241156932449e-03,
	 -8.92138498045729985177e-04,
          0.],
        [ 0.,
	  3.97849749499645077844e-01,
          1.36783778460411720168e+00,
	  4.98289622525382669416e-01,
         -6.37898194594723280150e-04],
        [-1.09676396052962061844e+01,
	  1.68015813878906206114e+00,
          5.71779846478865511061e-02,
	 -6.98210122488052056106e-03,
          3.34975017086070470649e-05],
        [-9.04316832390810593223e-02,
	 -6.76404519071381882256e-02,
          6.75961301770459654925e-02,
	  2.95552570429315521194e-02,
         -1.39180257516060693404e-05],
        [ 0.,
	  0.,
         -9.23364619367118555360e-02,
	 -1.69364939002081722752e-02,
         -1.40086798182036094347e-05]],
        dtype=A.dtype, device=A.device)

        # We implement the following allowing for batches
        #q31 = a01*I + a11*A + a21*A2 + a31*A3
        #q61 = b01*I + b11*A + b21*A2 + b31*A3 + b61*A6
        #q62 = b02*I + b12*A + b22*A2 + b32*A3 + b62*A6
        #q63 = b03*I + b13*A + b23*A2 + b33*A3 + b63*A6
        #q64 = b04*I + b14*A + b24*A2 + b34*A3 + b64*A6
        #q91 = q31 @ q64 + q63
        #return q61 + (q62 + q91) @ q91
        q = torch.stack([I, A, A2, A3, A6], dim=-3).unsqueeze_(-4)
        len_batch = A.ndimension() - 2
        q_size =  [-1 for _ in range(len_batch)] + [5, -1, -1, -1]
        q = q.expand(*q_size)
        b = b.unsqueeze_(-1).unsqueeze_(-1).expand_as(q)
        q = (b * q).sum(dim=-3)
        if batched:
            # Indexing the third to last dimension, because otherwise we
            # would have to prepend as many 1's as the batch shape for the
            # previous expand_as to work
            qaux = q[..., 0,:,:] @ q[..., 4,:,:] + q[..., 3,:,:]
            return q[..., 1,:,:] + (q[..., 2,:,:] + qaux) @ qaux
        else:
            qaux = q[0] @ q[4] + q[3]
            return q[1] + (q[2] + qaux) @ qaux


def differential(A, E, f):
    n = A.size(-1)
    size_M = list(A.size()[:-2]) + [2*n, 2*n]
    M = A.new_zeros(size_M)
    M[..., :n, :n] = A
    M[..., n:, n:] = A
    M[..., :n, n:] = E
    return f(M)[..., :n, n:]


class expm_taylor_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        return expm_taylor(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        # Handle tipical case separately as (dexp)_0 = Id
        if (A == 0).all():
            return G
        else:
            return differential(A.transpose(-2, -1), G, expm_taylor)


expm = expm_taylor_class.apply




if __name__ == "__main__":

    # Test expm with complex matrix of 4x4 and compare to pytorch expm

    A = torch.rand(4, 4, dtype=torch.double)\
        + 1j*torch.rand(4, 4, dtype=torch.double)

    A = A.type(torch.cdouble)
    A.requires_grad = True

    expm_taylor_val = expm(A)
    expm_val = torch.matrix_exp(A)

    expm_taylor_val.retain_grad()
    expm_val.retain_grad()

    print("expm_taylor_val")
    print(expm_taylor_val)

    print("expm_val")
    print(expm_val)

    assert torch.allclose(expm_taylor_val, expm_val)

    # Create a target tensor (you can modify this as needed)
    target = torch.rand_like(expm_val, requires_grad=True)

    # Calculate a loss (e.g., Mean Squared Error) between expm_taylor_val and target
    loss = torch.mean(torch.abs(expm_taylor_val - target))

    # Backpropagation
    loss.backward()

    loss2 = torch.mean(torch.abs(expm_val - target))
    loss2.backward()

    # Check if gradients are computed
    print("Gradient for expm_taylor_val:", expm_taylor_val.grad)
    print("Gradient for expm_val:", expm_val.grad)






    # tol = 1e-5
    # create_complex = False
    # for i in range(1000):
    #     print(f"i = {i}")

    #     for j in [2, 4]:
    #         rdm = create_random_rdm_torch(j, tol=tol, complex=create_complex)
    #         if not is_valid_rdm(rdm, tol=tol):
    #             print(f"Invalid {'1' if j == 2 else '2'}-RDM!")
    #             print(rdm)
    #             exit(1)

            
        
                 
        