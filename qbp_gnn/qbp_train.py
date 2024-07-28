import torch
import torch.nn as nn
from itertools import combinations
import quimb.tensor as qtn
from tqdm import tqdm
from ordered_set import OrderedSet
import quimb as qu
from tensor_network import tn_tensors_to_torch, build_dual_tn, unbuild_dual_tn, generate_tensor_network, visualize_tensor_network, dynamic_tensor_mult, compute_partial_trace
from tensor_network import contract_physical_indices as contract_phys_ind
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'


import sys
sys.path.append(".")
from lib.rdm import calculate_energy_batched
from dataset.ising.isingModel import IsingModelDataset
from baselines.tensorNetworks import TensorNetworkRunner, DMRG_QUIMB, SimpleUpdate
from lib.utils import parse_hamiltonian
from qbp_gnn import QBP_QGNN
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg



def hvp(model, v):
    """Compute the Hessian-vector product for a given vector v."""
    output, _, _ = model()
    first_grads = torch.autograd.grad(output, model.parameters(), create_graph=True)
    flat_first_grads = torch.cat([g.contiguous().view(-1) for g in first_grads])
    hvp = torch.autograd.grad(flat_first_grads, model.parameters(), grad_outputs=v, retain_graph=True)
    # Compute product of Hessian and v
    return torch.cat([g.contiguous().view(-1) for g in hvp])
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    contract_physical_indices = False
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    params = {
        'tensor_dtype': dtype,
        'contract_physical_indices': contract_physical_indices,
        'max_iter': 200,
        'tol': 1e-10,
        'improvement_rate_threshold': 1e-3,
        'stable_convergence_required': 1,
        'show_progress_bar': True,
        'device': device
    }

    defaults_SU = {
        'chi': 15,
        'bond_dim': 2,
        'num_iters': 100,
        'tau': [0.1, 0.01, 0.001]
    }

    defaults_DMRG_QUIMB = {
    'max_bond': 30,
    'cutoff': 1.e-10,
    'tol': 1.e-6,
    'verbosity': 0
    }

    # baseline = TensorNetworkRunner(DMRG_QUIMB, defaults_DMRG_QUIMB)
    baseline = TensorNetworkRunner(SimpleUpdate, defaults_SU)

    model = QBP_QGNN(params=params)

    defaults_SU = {
        'chi': 15,
        'bond_dim': 2,
        'num_iters': 100,
        'tau': [0.1, 0.01, 0.001]
    }    


    data_file = "dataset\ising\data\PEPS_5x4_N2000_PBCFalse.pt"
    # data_file = "dataset\ising\data\MPS_10x1_N2000_PBCFalse.pt"
    dataset = IsingModelDataset.load(data_file)

    idx = 80
    data_point = dataset[idx]
    print(f"Data Point {idx}")
    print(f"Shape {data_point.grid_extent}")
    print(f"PBC {data_point.pbc}")
    true_energy, rdms = baseline.run(data_point)
    ground_state = baseline.algorithm.getGroundState()
    tn_tensors_to_torch(ground_state, dtype=dtype)
    one_rdms = rdms[0]
    two_rdms = rdms[1]
    shape = data_point.grid_extent
    if len(shape) == 1:
        shape = (shape[0], 1)
    Lx, Ly = shape

    

    tn_rand, _, tn_type = generate_tensor_network(Lx, Ly, bond_dim=4, pbc=data_point.pbc, dtype=dtype, normalize = False)
    model.set_datapoint(tn_rand, tn_type=baseline.tn_type, datapoint=data_point.to(device))
    model.to(device)
    model.train()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, nesterov=True, momentum=0.999995)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, history_size=50, tolerance_change=0, tolerance_grad=0, line_search_fn="strong_wolfe")
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1.1, patience=2, verbose=True, threshold=1e-12)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.9, total_iters =200)

    energies = []
    
    print("SU/DMRG Energy: ", true_energy)
    print("True Energy: ", data_point.y_energy.item())
    diff = torch.inf
    iter = 0
    one_rdms_dmrg = {}
    one_rdms_labels = {}
    two_rdms_dmrg = {}
    two_rdms_labels = {}
    energy_trend = []

    difs = []

    with torch.autograd.set_detect_anomaly(True):
    
        for i in tqdm(range(100)):
            iter += 1
            def closure():
                optimizer.zero_grad()
                energy, _, _ = model()  # Obtain outputs from model
                loss = energy  # In this case, the 'energy' itself is used as the loss
                loss.backward(retain_graph=True)  # Compute gradients

                return loss  # Return the computed loss to the optimizer
            
            # Clear gradients at each step
            optimizer.step(closure)
            with torch.no_grad():
                energy, one_rdms_bp, two_rdms_bp = model()

            # optimizer.zero_grad()
            # energy, one_rdms_bp, two_rdms_bp = model()
            # energy.backward(retain_graph=True)
            # optimizer.step() 
            
            with torch.no_grad():
                for tensor in optimizer.param_groups[0]['params']:
                    tensor = tensor / torch.norm(tensor)
  
            if iter % 1 == 0:
                #Calculate fidelity with one and two rdms
                if len(one_rdms_dmrg) == 0 or len(one_rdms_labels) == 0:
                    for key in one_rdms_bp.keys():
                        one_rdms_dmrg[key] = torch.tensor(one_rdms[int(key.split('_')[0])], dtype=dtype)
                        one_rdms_labels[key] = torch.tensor(data_point.y_node_rdms[int(key.split('_')[0])], dtype=dtype)
                if len(two_rdms_dmrg) == 0 or len(two_rdms_labels) == 0:
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

            
                for key in one_rdms_bp.keys():
                    error = QBP_QGNN.calculate_distance(one_rdms_labels[key], one_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
                    print(f"[BP-Label]Error for {key}: {abs(1.0 - error)}")
                for key in two_rdms_bp.keys():
                    error = QBP_QGNN.calculate_distance(two_rdms_labels[key], two_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
                    print(f"[BP-Label]Error for {key}: {abs(1.0 - error)}")

            energies.append(energy.item())
            print(f"Energy: {energy.item()}")
            diff = abs(energy.item() - true_energy)
            difs.append(diff)  
            print(f"Difference: {diff}") 



    print("Ground State Energy: ", data_point.y_energy.item())
    print("Lowest Energy: ", energies[-1])

    print("One RDMs")
    for key in one_rdms_bp.keys():
        error = QBP_QGNN.calculate_distance(one_rdms_dmrg[key], one_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
        print(f"Error for {key}: {error}")
    
    print("Two RDMs")
    for key in two_rdms_bp.keys():
        error = QBP_QGNN.calculate_distance(two_rdms_dmrg[key], two_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
        print(f"Error for {key}: {error}")


    #Matplotlib print energy with line of true energy
    #Label both axes, title, legend, label the lines too
    import matplotlib.pyplot as plt
    plt.plot(energies)
    plt.axhline(y=true_energy, color='r', linestyle='--')
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.title("Energy vs Iterations")
    plt.legend(["Energy", "True Energy"])


    plt.show()



    




