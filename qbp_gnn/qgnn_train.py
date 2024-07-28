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
from torchviz import make_dot

import sys
sys.path.append(".")
from lib.rdm import calculate_energy_batched
from dataset.ising.isingModel import IsingModelDataset
from baselines.tensorNetworks import TensorNetworkRunner, DMRG_QUIMB, SimpleUpdate
from lib.utils import parse_hamiltonian
from qbp_gnn.qgnn_bp import QGNN, QBP


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    contract_physical_indices = False
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    params_qbp = {
        'tensor_dtype': dtype,
        'contract_physical_indices': contract_physical_indices,
        'max_iter': 10,
        'tol': 1e-10,
        'improvement_rate_threshold': 1e-3,
        'stable_convergence_required': 1,
        'show_progress_bar': False,
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

    baseline = TensorNetworkRunner(DMRG_QUIMB, defaults_DMRG_QUIMB)
    # baseline = TensorNetworkRunner(SimpleUpdate, defaults_SU)

  

    defaults_SU = {
        'chi': 15,
        'bond_dim': 2,
        'num_iters': 100,
        'tau': [0.1, 0.01, 0.001]
    }    


    # data_file = "dataset\ising\data\PEPS_2x2_1000_N1000_4_PBCFalse.pt"
    data_file = "dataset\ising\data\MPS_180_N20_10_PBCFalse.pt"
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

    params_model = {
        'num_layers': 6,
        'grid_extent': shape,
        'bond_dim': 3,
        'phys_dim': 2,
        'z_dim': 16,
        'pbc': data_point.pbc,
        'hidden_dim': 32,
        'hidden_dim_z':32,
        'hidden_dim_msg': 32,
        'max_iter':1,
        'normalize_tn_init': False,
    }

    model = QGNN(params={**params_qbp, **params_model})
    model.to(device)
    model.train()
    datapoint = data_point.to(device)

    opt_tensor_indices = []
    #Show model named parameters
    i = 0
    for name, param in model.named_parameters():
        print(name, param.shape)
        if "tn_tensor" in name:
            opt_tensor_indices.append(i)
        i += 1


    print("Initiialized!")

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, nesterov=True, momentum=0.999995)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
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

    datapoint.y_energy = datapoint.y_energy.to(params_qbp['tensor_dtype'])

    with torch.autograd.set_detect_anomaly(True):
    
        for i in range(100):
            iter += 1
            def closure():
                optimizer.zero_grad()
                energy, _, _ = model(datapoint)  # Obtain outputs from model
                loss = torch.nn.MSELoss()(energy[0], data_point.y_energy) 
                loss.backward(retain_graph=True)  # Compute gradients

                return loss  # Return the computed loss to the optimizer
            
            # Clear gradients at each step
            optimizer.step(closure)
            with torch.no_grad():
                energy, one_rdms_bp, two_rdms_bp = model(data_point)

            # optimizer.zero_grad()
            # energy, one_rdms_bp, two_rdms_bp = model(data_point)
            # loss = torch.nn.MSELoss()(energy[0], data_point.y_energy)  
            # loss.backward(retain_graph=True)
            # optimizer.step() 
            
            with torch.no_grad():
                for idx, param in enumerate(optimizer.param_groups[0]['params']):
                        if idx in opt_tensor_indices:
                            param /= torch.norm(param)
            
            # #Print Gradients
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, torch.norm(param.grad).item())

  
            if iter % 2 == 0:
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
                    error = QBP.calculate_distance(one_rdms_labels[key], one_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
                    print(f"[BP-Label]Error for {key}: {abs(1.0 - error)}")
                for key in two_rdms_bp.keys():
                    error = QBP.calculate_distance(two_rdms_labels[key], two_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
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
        error = QBP.calculate_distance(one_rdms_dmrg[key], one_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
        print(f"Error for {key}: {error}")
    
    print("Two RDMs")
    for key in two_rdms_bp.keys():
        error = QBP.calculate_distance(two_rdms_dmrg[key], two_rdms_bp[key][0].cpu(), method="fidelity", dtype=dtype)
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



    




