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


def get_dataset(dataset_type, dataset_path):
    if dataset_type == "ising":
        return IsingModelDataset.load(os.path.dirname(os.path.abspath(__file__)) + "/" + dataset_path)
    else:
        raise NotImplementedError()

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    dataset_size = len(dataset)
    assert (test_ratio + val_ratio + train_ratio == 1.0)

    # Calculate the number of samples for each split
    num_train = int(dataset_size * train_ratio)
    num_val = int(dataset_size * val_ratio)
    num_test = dataset_size - num_train - num_val

    # Create a list of indices to shuffle the dataset
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    # Split the dataset using the shuffled indices
    train_indices, remaining_indices = indices[:num_train], indices[num_train:]
    val_indices, test_indices = remaining_indices[:num_val], remaining_indices[num_val:]

    # Create train, val, and test subsets using list indexing
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]

    return train_dataset, val_dataset, test_dataset


def set_seed(seed):
    """Function for setting the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    dataset_name = "PEPS_5x4_N2000_PBCFalse.pt"
    dataset_folder = "dataset\ising\data"
    output_folder = "dataset\ising\data"
    max_split_size = 50
    seed = 42

    data_file = os.path.join(dataset_folder, dataset_name)
    set_seed(seed)

    dataset = get_dataset("ising", data_file)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    #Only split test dataset
    for i in range(0, len(test_dataset), max_split_size):
        test_dataset_split = test_dataset[i:i + max_split_size]
        test_dataset_split_name = f"{dataset_name}_test_{i}_{i + max_split_size}.pt"
        torch.save(test_dataset_split, os.path.join(output_folder, test_dataset_split_name))
        print(f"Saved test dataset split to {test_dataset_split_name}")

