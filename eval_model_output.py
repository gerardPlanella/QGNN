from datetime import datetime
import os
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import copy
import time
import json
import numpy as np
import argparse
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from torch_geometric.loader import DataLoader
from dataset import IsingModelDataset
from train import train, evaluate, evaluate_qbp, train_qgnn


from baselines import GNN, TensorNetworkRunner, DMRG, SimpleUpdate, FullUpdate, SimpleUpdateGen, DMRG_QUIMB
from qgnn import QGNN, QGNN2, QGNN_EM
from lib import rdm_from_bloch_vec, fidelity, conditional_fidelity_loss, squared_hilbert_schmidt_distance, trace_distance, quantum_loss
from qbp_gnn import QBP_QGNN
from qbp_gnn import QGNN as QGNN4


def criterion(pred, batch):
    
    x_nodes, x_edges, x_global = pred

    #For batch size 1, x_global is a scalar, but batch.y_energy is a tensor of size 1
    if x_global.dim() == 0 and batch.y_energy.dim() == 1:
        x_global = x_global.unsqueeze(0)


    loss_energy = F.l1_loss(x_global, batch.y_energy, reduce='sum')

    # QGNN_EM returns the rdms directly
    x_node_rdms = x_nodes
    x_edge_rdms = x_edges

    # # convert batch to tensors
    # # y_node_rdms is a list of lists of different sizes, so merge the two first dimensions (out of 4)
    y_node_rdms = np.concatenate(batch.y_node_rdms, axis=0)
    y_node_rdms = torch.tensor(y_node_rdms, dtype=x_node_rdms.dtype, device=x_node_rdms.device)
    # fid_node = squared_hilbert_schmidt_distance(x_node_rdms, y_node_rdms)
    # fid_node = fidelity(x_node_rdms, y_node_rdms)
    # fid_node = conditional_fidelity_loss(x_node_rdms, y_node_rdms)
    # fid_node = trace_distance(x_node_rdms, y_node_rdms)
    # fid_node = F.l1_loss(x_node_rdms, y_node_rdms, reduction='sum')
    loss_node = quantum_loss(x_node_rdms, y_node_rdms, trace_distance)
    # print("loss node",fid_node.item())

    
    y_edge_rdms = np.concatenate(batch.y_edge_rdms, axis=0)
    y_edge_rdms = torch.tensor(y_edge_rdms, dtype=x_edge_rdms.dtype, device=x_edge_rdms.device)
    #For all interaction terms with value 0, the 2rdm should not contribute to the loss
    interaction_terms = batch.x_edges
    #Remove all rdms that correspond to interaction terms with value 0, use torch.where to get the indices
    #of the interaction terms with value 0
    indices = torch.where(interaction_terms != 0)[0]
    x_edge_rdms = torch.index_select(x_edge_rdms, 0, indices)
    y_edge_rdms = torch.index_select(y_edge_rdms, 0, indices)

    # fid_edge = squared_hilbert_schmidt_distance(x_edge_rdms, y_edge_rdms)
    # fid_edge = fidelity(x_edge_rdms, y_edge_rdms)
    # fid_edge = conditional_fidelity_loss(x_edge_rdms, y_edge_rdms)
    # fid_edge = trace_distance(x_edge_rdms, y_edge_rdms)
    # fid_edge = F.l1_loss(x_edge_rdms, y_edge_rdms, reduction='sum')
    loss_edge = quantum_loss(x_edge_rdms, y_edge_rdms, trace_distance)
    # print("loss edge",fid_edge.item())

    loss = loss_energy + (loss_node + loss_edge)

    # print(f"loss energy: {loss_energy[0].item()} | loss_node: {loss_node[0].item()} | loss_edge: {loss_edge.item()[0]}")
    # construct a string of length 100 with + for energy, o for node and x for edge, representing the respective percentage of the loss
    # energy_len = int(np.round(100 * loss_energy.item() / loss.item()))
    # node_len = int(np.round(100 * loss_node.item() / loss.item()))
    # edge_len = 100 - energy_len - node_len
    # loss_str = "+" * energy_len + "o" * node_len + "x" * edge_len
    # print(f"loss: {loss.item():8.4f} | {loss_str}")
    

    return loss, loss_energy, loss_node, loss_edge

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate model output')
  parser.add_argument('--data_path', type=str, default='output\model_outputs\qbp_PEPS_5x4_N2000_PBCFalse.pt', help='Path to the data directory')
  args = parser.parse_args()

  dataset_folder = 'output\\model_outputs\\'

  datasets = ['qbp_MPS_10x1_N2000_PBCFalse_S42_2024-07-06_14-12-05.pt', 'qbp_MPS_20x1_N2000_PBCFalse_S42_2024-07-08_20-52-06.pt', 'qbp_PEPS_3x3_N2000_PBCFalse_S42_2024-07-08_20-26-49.pt', 'qbp_PEPS_5x4_N2000_PBCFalse.pt']

  data_path = args.data_path

  

  for i, dataset_file in enumerate(datasets):
    fig_boxplots, ax_boxplots = plt.subplots(1,1, figsize=(10, 5))
    # fig_hist, ax_hist = plt.subplots(1, 3, figsize=(10,5))

    dataset_path = os.path.join(dataset_folder, dataset_file)
    dataset = IsingModelDataset.load(dataset_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    node_loss = []
    edge_loss = []
    energy_loss = []

    system_type = dataset_file.split('_')[1]
    system_dims = dataset_file.split('_')[2]
    dataset_name = f"{system_type} {system_dims}"

    for data_point in loader:
      pred = data_point.x_node_rdms, data_point.x_edge_rdms, data_point.x_energy
      _, loss_energy, loss_node, loss_edge = criterion(pred, data_point)
      print(f"loss energy: {loss_energy.item()} | loss_node: {loss_node.item()} | loss_edge: {loss_edge.item()}")
      node_loss.append(loss_node.item())
      edge_loss.append(loss_edge.item())
      energy_loss.append(loss_energy.item())

    print(f"Average node loss: {np.mean(node_loss)}")
    print(f"Average edge loss: {np.mean(edge_loss)}")
    print(f"Average energy loss: {np.mean(energy_loss)}")

    # Boxplot showing anomalies for each loss, in one same figure, aligned horizontally
    ax_boxplots.boxplot([node_loss, edge_loss, energy_loss], manage_ticks=True, showmeans=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax_boxplots.set_yscale('log')  # Set y-axis scale to log
    ax_boxplots.set_ylabel('Loss', fontsize=20)
    ax_boxplots.set_xticklabels(['1RDM', '2RDM', 'Energy'], fontsize=20)
    ax_boxplots.tick_params(axis='both', which='major', labelsize=20)
    ax_boxplots.tick_params(axis='y', labelsize=15)  # Change y-axis tick label size

    # Add mean and median labels per loss
    for i, loss in enumerate([node_loss, edge_loss, energy_loss]):
      ax_boxplots.text(i + 1.04, np.mean(loss), f"{np.mean(loss):.1e}", ha='left', va='bottom', color='darkgreen', fontsize=15)
      ax_boxplots.text(i + 1.2, np.median(loss), f"{np.median(loss):.1e}", ha='left', va='top', color='darkorange', fontsize=15)

    # Calculate average without outliers
    node_loss = np.array(node_loss)
    edge_loss = np.array(edge_loss)
    energy_loss = np.array(energy_loss)
    node_loss = node_loss[np.abs(node_loss - np.mean(node_loss)) < 2 * np.std(node_loss)]
    edge_loss = edge_loss[np.abs(edge_loss - np.mean(edge_loss)) < 2 * np.std(edge_loss)]
    energy_loss = energy_loss[np.abs(energy_loss - np.mean(energy_loss)) < 2 * np.std(energy_loss)]
    print(f"Average node loss without outliers: {np.mean(node_loss)}")
    print(f"Average edge loss without outliers: {np.mean(edge_loss)}")
    print(f"Average energy loss without outliers: {np.mean(energy_loss)}")

    # # Histogram with log x scale for each loss
    # ax_hist[0].hist(node_loss, bins=200)
    # ax_hist[0].set_title(f'{dataset_name} - 1RDM Loss')
    # ax_hist[0].set_xscale('log')
    # ax_hist[1].hist(edge_loss, bins=200)
    # ax_hist[1].set_title(f'{dataset_name} - 2RDM Loss')
    # ax_hist[1].set_xscale('log')
    # ax_hist[2].hist(energy_loss, bins=20)
    # ax_hist[2].set_title(f'{dataset_name} - Energy Loss')
    # ax_hist[2].set_xscale('log')

    plt.tight_layout()
    plt.show()


    #Wait for user input to either continue to the next dataset or exit
    if i < len(datasets) - 1:
      input("Press Enter to continue...")

    else:
      break
