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

script_dir = os.path.dirname(__file__)
dataset_path = 'dataset/ising/data/PEPS_3x3_N2000_PBCFalse.pt'
# dataset_path = 'dataset/ising/data/MPS_10x1_N2000_PBCFalse.pt'
#Check if windows or linux to change the path
if os.name == 'nt':
    dataset_path = dataset_path.replace('/', '\\')

qbp_by_parts = False

defaults = {
    "model":"qbp",
    "rdm_dataset" : False,
    "use_gpu": True,
    "version": 1,
    "dataset_type": "ising",
    "dataset_path": dataset_path,
    "seed": 42,
    "batch_size": 1,
    "use_wandb": True,
    "save_model_outputs": False,
    "model_output_folder_name": "model_outputs",
}

defaults_parametrised = {
    "epochs": 60,
    "learning_rate": 5e-4,
    "weight_decay": 1e-16,
    "num_layers": 7,
    "hidden_channels": 128,
    "use_pbc": False
}

defaults_qgnn_EM = {    
    "one_rdm_dims" : [2, 2],
    "one_rdm_hidden_rot_dim" : 128, 
    "one_rdm_hidden_mixing_dim" : 128,
    "two_rdm_dims": [4, 4], 
    "two_rdm_hidden_rot_dim": 256, 
    "two_rdm_hidden_mixing_dim": 256,
    "use_simplified": False, # Replaces 1-RDM Rotation Layers with a simplified Euler Angle Rotation Layer
    "complex_init": False, # Initializes the 1-RDM hidden states to complex values
    "calculate_energy": True, # Calculates the energy from the 1-RDMs and 2-RDMs or predicts the value from these
    "use_eigenvector_phase" : False, # Uses the eigenvector phase in the Mixing Layers
    "use_complex": False
}

defaults_QBP_Grad = {
    'tensor_dtype': torch.float64,
    'contract_physical_indices': False,
    'max_iter': 200,
    'tol': 1e-5,
    'improvement_rate_threshold': 1e-5,
    'stable_convergence_required': 2,
    'show_progress_bar': True,
    'use_lbfgs': True,
}

default_LBFGS_optimizer_params = {
    'lr_tensor':1,
    'history_size':50,
    'tolerance_change':0,
    'tolerance_grad':0,
    'line_search_fn':"strong_wolfe"
}

#FIXME: Defaults not working between models
defaults_QBP_QGNN = {
    'bond_dim': 4,
    'phys_dim': 2,
    'z_dim': 32,
    'pbc': False, 
    'hidden_dim': 64,
    'hidden_dim_z':64,
    'hidden_dim_msg': 64,
    'normalize_tn_init': False,
    'use_residual': True,
    'use_rdms_loss':False
}

defaults_SU = {
    'chi': 15,
    'bond_dim': 4,
    'num_iters': 100,
    'tau': [0.1, 0.01, 0.001]
}

defaults_DMRG = {
    'max_E_err': 1.e-10,
    'chi_max': 30,
    'svd_min': 1.e-10
}

defaults_DMRG_QUIMB = {
    'max_bond': 60,
    'cutoff': 1.e-10,
    'tol': 1.e-6,
    'verbosity': 0
}


defaults_SU_gen = {
    'max_bond': 15,
}




def setup_gpu():
    """Function for setting up the GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA available. Setting device to CUDA:", device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS available. Setting device to MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU or MPS available. Setting device to CPU.")
    return device


def set_seed(seed):
    """Function for setting the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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



def get_model(model_name, kwargs, device, grid_extent=None):
    """Gets the corresponding model according to the specified run args."""
    rdm_dataset = kwargs['rdm_dataset']

    def to_double(model):
        for child in model.children():
            if isinstance(child, torch.nn.Linear):
                child.weight = torch.nn.Parameter(child.weight.double())
                if child.bias is not None:
                    child.bias = torch.nn.Parameter(child.bias.double())
            else:
                to_double(child)  # Apply recursively to children

    shape = None
    if grid_extent is not None:
        shape = grid_extent
        if len(shape) == 1:
            shape = (shape[0], 1)

    if model_name == 'gnn':
        model = GNN(in_channels=2, out_channels=1, edge_channels=1, **kwargs)
    elif model_name == 'dmrg':
        model = TensorNetworkRunner(DMRG, kwargs)
    elif model_name == 'su':
        model = TensorNetworkRunner(SimpleUpdate, kwargs)
    elif model_name == 'fu':
        model = TensorNetworkRunner(FullUpdate, kwargs)
    elif model_name == 'dmrg_quimb':
        model = TensorNetworkRunner(DMRG_QUIMB, kwargs)
    elif model_name == 'su_gen':
        assert kwargs['bond_dim'] == 2, "Bond dimension must be 2 for SU_gen, make sure to pass it as argument"
        model = TensorNetworkRunner(SimpleUpdateGen, kwargs)
    
    elif model_name == 'qbp':
        kwargs['device'] = device
        model = QBP_QGNN(params=kwargs)

    elif model_name == 'qgnn':
        version = kwargs['version']

        if isinstance(version, int):
            if version == 1:
                model = QGNN(node_in_channels=2, node_channels=kwargs['hidden_channels'], node_out_channels=2,
                            edge_in_channels=1, edge_channels=kwargs['hidden_channels'], edge_out_channels=9,
                            num_layers=kwargs['num_layers'], global_out_channels=1, use_pbc=kwargs['use_pbc'])
            elif version == 2:
                if not rdm_dataset:
                    node_in_channels = 2
                    node_out_channels = 2
                    edge_in_channels = 1
                    edge_out_channels = 9
                else:
                    node_in_channels = 2 + 2**2 + 1
                    edge_in_channels = 1 + 2**4 + 1
                    edge_out_channels = 9
                    node_out_channels = 2

                model = QGNN2(node_in_channels=node_in_channels, node_channels=kwargs['hidden_channels'], node_out_channels=node_out_channels,
                            edge_in_channels=edge_in_channels, edge_channels=kwargs['hidden_channels'], edge_out_channels=edge_out_channels,
                            num_layers=kwargs['num_layers'], global_out_channels=1, use_pbc=kwargs['use_pbc'], rdm_dataset = rdm_dataset)
            
                if rdm_dataset:
                    to_double(model)

            elif version == 3:    

                

                model = QGNN_EM(one_rdm_dims=kwargs['one_rdm_dims'], one_rdm_hidden_rot_dim=kwargs['one_rdm_hidden_rot_dim'], one_rdm_hidden_mixing_dim=kwargs['one_rdm_hidden_mixing_dim'],
                                two_rdm_dims=kwargs['two_rdm_dims'], two_rdm_hidden_rot_dim=kwargs['two_rdm_hidden_rot_dim'], two_rdm_hidden_mixing_dim=kwargs['two_rdm_hidden_mixing_dim'], 
                                num_layers=kwargs["num_layers"], device=device,
                                kwargs=kwargs)
                
                to_double(model)
            
            elif version == 4:
                assert shape is not None, "Grid extent must be passed as argument for QGNN4"
                kwargs['grid_extent'] = shape
                kwargs['pbc'] = kwargs['use_pbc']
                kwargs['device'] = device
                model = QGNN4(params=kwargs)

                
            else:
                raise NotImplementedError(f'Version {version} not recognized for {model_name}.')
    else:
        raise NotImplementedError(f'Model name {model_name} not recognized.')
    return model.to(device)


def filter_model_params(model_name, args):
    # Dictionary mapping model names to their respective default parameter sets
    model_defaults = {
        'gnn': defaults_parametrised,
        'dmrg': defaults_DMRG,
        'dmrg_quimb': defaults_DMRG_QUIMB,
        'su': defaults_SU,
        'fu': defaults_SU,
        'su_gen': defaults_SU_gen,
        'qgnn': defaults_parametrised,  # Default case for QGNN,
        'qbp': defaults_QBP_Grad,
        'qgnn4': defaults_QBP_QGNN,
        'lbfgs_opt': default_LBFGS_optimizer_params
    }

    relevant_defaults = model_defaults.get(model_name, {})

    if model_name == 'su_gen':
        relevant_defaults = {**defaults_SU, **defaults_SU_gen}

    # Special handling for QGNN version 3
    if model_name == 'qgnn':
        if args.version == 3:
            relevant_defaults = {**relevant_defaults, **defaults_qgnn_EM, "version": args.version, **defaults}
        elif args.version < 3:
            relevant_defaults = {**relevant_defaults, "version": args.version, **defaults}
        elif args.version == 4:
            relevant_defaults = {**relevant_defaults, **defaults_QBP_Grad, **defaults_QBP_QGNN,  "version": args.version}
    else:
        relevant_defaults = {**relevant_defaults, **defaults}

    if model_name == 'qbp'or (model_name == 'qgnn' and args.version == 4):
        if relevant_defaults['use_lbfgs']:
            relevant_defaults = {**relevant_defaults, **default_LBFGS_optimizer_params}
            #remove learning_rate
            relevant_defaults.pop('learning_rate', None)
        elif model_name != 'qbp':
            #Add the tensor_lr
            relevant_defaults = {**relevant_defaults, 'lr_tensor': default_LBFGS_optimizer_params['lr_tensor']}
        

    # Filter the args to include only those present in the relevant defaults
    filtered_args = {k: v for k, v in vars(args).items() if k in relevant_defaults}
    return filtered_args


def parse_options():
    """Function for parsing command line arguments."""
    parser = argparse.ArgumentParser("Model runner.")

    """
        QGNN Versions:
        1. Simple GNN, predicts RDM Bloch Vectors and Energy directly
        2. Simple GNN, predicts RDM Bloch Vectors and Calculates Energy from these
        3. EM Inspired GNN, Rotation and Mixing Layers, directly prefdicts RDMs and either 
           calculates Energy from these or predicts it, no communication between one and two body RDM layers
        4. QBP based version 
    """


    torch.set_printoptions(linewidth=135, sci_mode=False)
    

    # Config matters
    parser.add_argument('--config', type=str, default=None, metavar='S',
                    help='Config file for parsing arguments. '
                        'Command line arguments will be overriden.')
    parser.add_argument('--write_config_to', type=str, default=None, metavar='S',
                    help='Writes the current arguments as a json file for '
                        'config with the specified filename.')
    parser.add_argument('--evaluate', type=str, default=None, metavar='S',
                    help='Directly evaluates the model with the model weights'
                        'of the path specified here. No need to specify the directory.')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb', default=defaults["use_wandb"],
                    help='Whether or not to use wandb for logging.')
    parser.add_argument('--wandb_group', type=str, default=None, metavar='S',
                    help='Wandb group for logging.')
    parser.add_argument('--use_gpu', action='store_true', default=defaults["use_gpu"],
                    help='Whether or not to use GPU.')
    parser.add_argument('--save_model_outputs', action='store_true', default=defaults["save_model_outputs"],
                    help='Whether or not to save model outputs.')
    parser.add_argument('--model_output_folder_name', type=str, default=defaults["model_output_folder_name"], metavar='S',
                    help='Model output folder name.')
    parser.add_argument('--rdm_dataset', action='store_true', default=defaults["rdm_dataset"],
                    help='Whether or not to use the RDM dataset.')
    

    # General Training parameters
    parser.add_argument('--model', type=str, default=defaults["model"], metavar='S',
                    help='Available models: qgnn | gnn | dmrg (MPS) | su (PEPS) | fu (PEPS) | su_gen (PEPS & MPS) | dmrg_quimb (MPS OBC)')
    parser.add_argument('--version', type=int, default=defaults["version"], metavar='S',
                    help='Available versions (qgnn): 1 | 2 | 3(EM) | 4(QBP)')
    parser.add_argument('--dataset_type', type=str, default=defaults["dataset_type"], metavar='S',
                    help='Available datasets: ising')
    parser.add_argument('--dataset_path', type=str, default=defaults["dataset_path"], metavar='S',
                    help='Available datasets: ising')
    parser.add_argument('--seed', type=int, default=defaults["seed"], metavar='N',
                    help='Random seed')
    parser.add_argument('--epochs', type=int, default=defaults_parametrised["epochs"], metavar='N',
                    help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=defaults["batch_size"], metavar='N',
                    help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=defaults_parametrised["learning_rate"], metavar='N',
                    help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=defaults_parametrised["weight_decay"], metavar='N',
                    help='clamp the output of the coords function if get too large')
    parser.add_argument('--use_pbc', action='store_true', default=defaults_parametrised["use_pbc"], 
                    help='Whether or not to use periodic boundary conditions in the models')
    
    # Specific QGNN EM parameters

    parser.add_argument('--one_rdm_dims', type=int, nargs='+', default=defaults_qgnn_EM["one_rdm_dims"], metavar='N',
                    help='Dimensions of the one-body reduced density matrix')
    parser.add_argument('--one_rdm_hidden_rot_dim', type=int, default=defaults_qgnn_EM["one_rdm_hidden_rot_dim"], metavar='N',
                    help='Hidden dimensions for the rotation network of the one-body reduced density matrix')
    parser.add_argument('--one_rdm_hidden_mixing_dim', type=int, default=defaults_qgnn_EM["one_rdm_hidden_mixing_dim"], metavar='N',
                    help='Hidden dimensions for the mixing network of the one-body reduced density matrix')
    parser.add_argument('--two_rdm_dims', type=int, nargs='+', default=defaults_qgnn_EM["two_rdm_dims"], metavar='N',
                    help='Dimensions of the two-body reduced density matrix')
    parser.add_argument('--two_rdm_hidden_rot_dim', type=int, default=defaults_qgnn_EM["two_rdm_hidden_rot_dim"], metavar='N',
                    help='Hidden dimensions for the rotation network of the two-body reduced density matrix')
    parser.add_argument('--two_rdm_hidden_mixing_dim', type=int, default=defaults_qgnn_EM["two_rdm_hidden_mixing_dim"], metavar='N',
                    help='Hidden dimensions for the mixing network of the two-body reduced density matrix')
    parser.add_argument('--use_simplified', action='store_true', default=defaults_qgnn_EM["use_simplified"],
                    help='Whether or not to use the simplified Euler Angle Rotation Layer')
    parser.add_argument('--complex_init', action='store_true', default=defaults_qgnn_EM["complex_init"],
                    help='Whether or not to initialize the 1-RDM hidden states to complex values')
    parser.add_argument('--calculate_energy', action='store_true', default=defaults_qgnn_EM["calculate_energy"],
                    help='Whether or not to calculate the energy from the 1-RDMs and 2-RDMs or predict the value from these')
    parser.add_argument('--use_eigenvector_phase', action='store_true', default=defaults_qgnn_EM["use_eigenvector_phase"],
                    help='Whether or not to use the eigenvector phase in the Mixing Layers')
    parser.add_argument('--use_complex', action='store_true', default=defaults_qgnn_EM["use_complex"],
                        help='Whether or not to use complex numbers in the model')

    # Network specific parameters
    parser.add_argument('--hidden_channels', type=int, default=defaults_parametrised["hidden_channels"], metavar='N',
                    help='Hidden dimensions')
    parser.add_argument('--num_layers', type=int, default=defaults_parametrised["num_layers"], metavar='N',
                    help='Number of model layers')

    # Simple Update / Full Update parameters
    parser.add_argument('--chi', type=int, default=defaults_SU["chi"], metavar='N',
                    help='Virtual Bond dimension')
    parser.add_argument('--bond_dim', type=int, default=defaults_SU["bond_dim"], metavar='N',
                    help='Physical Bond dimension')
    parser.add_argument('--num_iters', type=int, default=defaults_SU["num_iters"], metavar='N',
                    help='Number of iterations')
    parser.add_argument('--tau', type=float, nargs='+', default=defaults_SU["tau"], metavar='N',
                    help='Time steps for Imaginary time evolution')
    parser.add_argument('--max_bond', type=int, default=defaults_SU_gen["max_bond"], metavar='N',
                    help='Maximum virtual bond dimension')
    
    # DMRG parameters
    parser.add_argument('--max_E_err', type=float, default=defaults_DMRG["max_E_err"], metavar='N',
                    help='Maximum energy error')
    parser.add_argument('--chi_max', type=int, default=defaults_DMRG["chi_max"], metavar='N',
                    help='Maximum virtual bond dimension')
    parser.add_argument('--svd_min', type=float, default=defaults_DMRG["svd_min"], metavar='N',
                    help='Minimum singular value')
    
    #QBP/QGNN4 parameters
    parser.add_argument('--tensor_dtype', type=torch.dtype, default=defaults_QBP_Grad["tensor_dtype"], metavar='S',
                    help='Tensor data type')
    parser.add_argument('--contract_physical_indices', action='store_true', default=defaults_QBP_Grad["contract_physical_indices"],
                    help='Contract physical indices')
    parser.add_argument('--max_iter', type=int, default=defaults_QBP_Grad["max_iter"], metavar='N',
                    help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=defaults_QBP_Grad["tol"], metavar='N',
                    help='Tolerance')
    parser.add_argument('--improvement_rate_threshold', type=float, default=defaults_QBP_Grad["improvement_rate_threshold"], metavar='N',
                    help='Improvement rate threshold')
    parser.add_argument('--stable_convergence_required', type=int, default=defaults_QBP_Grad["stable_convergence_required"], metavar='N',
                    help='Stable convergence required')
    parser.add_argument('--show_progress_bar', action='store_true', default=defaults_QBP_Grad["show_progress_bar"],
                    help='Show progress bar')
    
    #QBP/QGNN4 parameters
    parser.add_argument('--use_lbfgs', action='store_true', default=defaults_QBP_Grad["use_lbfgs"],
                    help='Use LBFGS optimizer')
    parser.add_argument('--z_dim', type=int, default=defaults_QBP_QGNN["z_dim"], metavar='N',
                    help='Z dimension')
    parser.add_argument('--hidden_dim', type=int, default=defaults_QBP_QGNN["hidden_dim"], metavar='N',
                    help='Hidden dimension')
    parser.add_argument('--hidden_dim_z', type=int, default=defaults_QBP_QGNN["hidden_dim_z"], metavar='N',
                    help='Hidden dimension for Z')
    parser.add_argument('--hidden_dim_msg', type=int, default=defaults_QBP_QGNN["hidden_dim_msg"], metavar='N',
                    help='Hidden dimension for MSG')
    parser.add_argument('--normalize_tn_init', action='store_true', default=defaults_QBP_QGNN["normalize_tn_init"],
                    help='Normalize the tensor network initialization')
    parser.add_argument('--phys_dim', type=int, default=defaults_QBP_QGNN["phys_dim"], metavar='N',
                    help='Physical dimension, 2 for spin 1/2 systems')
    parser.add_argument('--use_residual', action='store_true', default=defaults_QBP_QGNN["use_residual"],
                    help='Use residual connections')
    parser.add_argument('--use_rdms_loss', action='store_true', default=defaults_QBP_QGNN["use_rdms_loss"],
                    help='Use RDMs loss')
    
    # LBFGS Optimizer parameters
    parser.add_argument('--lr_tensor', type=float, default=default_LBFGS_optimizer_params['lr_tensor'], metavar='N',
                    help='Learning rate for the tensor parameters')
    parser.add_argument('--history_size', type=int, default=default_LBFGS_optimizer_params['history_size'], metavar='N',
                    help='History size')
    parser.add_argument('--tolerance_change', type=float, default=default_LBFGS_optimizer_params['tolerance_change'], metavar='N',
                    help='Tolerance change')
    parser.add_argument('--tolerance_grad', type=float, default=default_LBFGS_optimizer_params['tolerance_grad'], metavar='N',
                    help='Tolerance gradient')
    parser.add_argument('--line_search_fn', type=str, default=default_LBFGS_optimizer_params['line_search_fn'], metavar='S',
                    help='Line search function')
    

    
    args = parser.parse_args()

    # If config file is specified, parse it and override the command line arguments
    if args.config is not None:
        config_dir_path = os.path.join(script_dir, 'config')
        with open(os.path.join(config_dir_path, args.config), 'r') as cf:
            parser.set_defaults(**json.load(cf))
            print(f'Successfully parsed the arguments from config/{args.config}')
        args = parser.parse_args()

    # If evaluate is specified, override the command line arguments
    if args.write_config_to is not None:
        # If no config directory, make it
        config_dir_path = os.path.join(script_dir, 'config')
        if not os.path.exists(config_dir_path):
            os.makedirs(config_dir_path)

        # If no file, make it
        args.write_config_to += '.json' if args.write_config_to[-5:] != '.json' else ""
        with open(os.path.join(config_dir_path, args.write_config_to), 'w') as cf:
            json_args = copy.deepcopy(vars(args))
            del json_args['config']
            del json_args['write_config_to']
            json.dump(json_args, cf, indent=4)
            print(f'Successfully wrote the config to config/{args.write_config_to}')

    # Delete the config and write_config_to arguments to avoid confusion
    del args.config
    del args.write_config_to
    return args

if __name__ == '__main__':
    args = parse_options()

    model_output_path = None
    if args.save_model_outputs:
        model_folder = "output/" + args.model_output_folder_name
        # If it doesn't exist, create it
        model_output_path = os.path.join(script_dir, model_folder)
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)
    

        dataset_name = os.path.basename(args.dataset_path)[:-3]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        output_dataset_name = f'{args.model}_{dataset_name}_S{args.seed}_{datetime_str}.pt'
        model_output_path = os.path.join(model_output_path, output_dataset_name)
        if os.name == 'nt':
            model_output_path = model_output_path.replace('/', '\\')
    

    # Set the hardware accelerator
    device = setup_gpu() if args.use_gpu else torch.device('cpu')

    # Set seed for reproducibility
    set_seed(args.seed)

    # Get the dataset object
    dataset = get_dataset(args.dataset_type, args.dataset_path)

    grid_extent = None
    if ((args.model == 'qgnn' and args.version == 4) or args.model == 'qbp'):
        if not args.rdm_dataset:
            shape_name = args.dataset_path.split('\\')[-1].split('_')[1]
        else:
            shape_name = args.dataset_path.split('\\')[-1].split('_')[2]
        grid_extent = tuple(map(int, shape_name.split('x')))
        print(f"Grid extent: {grid_extent}")

    # Initialize the model
    model = get_model(args.model, vars(args), device=device, grid_extent=grid_extent)
    print(model)

    if qbp_by_parts and isinstance(model, QBP_QGNN):
        test_dataset = dataset
        test_loader = test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # Split the dataset into train, val and test
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        # Initialize the dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Number of parameters of the model

    if not isinstance(model, (TensorNetworkRunner, QBP_QGNN)):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of parameters: {num_params}\n')
    else:
        num_params = 0

    dataset_name = os.path.basename(args.dataset_path)[:-3]
    run_name = f'{args.model}_{args.num_layers}_{dataset_name}_{args.seed}'
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.model == 'qgnn':
        run_name = f'{args.model}_V{args.version}_L{args.num_layers}_{dataset_name}_S{args.seed}_{datetime_str}'
    else:
        run_name = f'{args.model}_{dataset_name}_S{args.seed}_H{args.hidden_channels}_{datetime_str}'


    if args.use_wandb:
        import wandb

        filtered_params = filter_model_params(args.model, args)

        # Setting the WandB parameters
        config = {
            **filtered_params,
            'num_params': num_params,
            'num_samples': len(dataset),
            'dataset_name': dataset_name,
            'model': args.model,
        }

        if (args.model == 'qgnn' and args.version == 4) or args.model == 'qbp':
            config['shape'] = 'x'.join(map(str,grid_extent)) 

        pprint(config)

        # Initialize the wandb run
        wandb.init(project="qgnn_experiments_v3", config=config, reinit=True, group=args.wandb_group,
                name=run_name)
        
        if not isinstance(model, TensorNetworkRunner):
            wandb.watch(model)
    else:
        print("Not using wandb. Skipping logging.")
        train_maes = []
        valid_maes = []
        train_energy_maes = []
        train_node_maes = []
        train_edge_maes = []
        valid_energy_maes = []
        valid_node_maes = []
        valid_edge_maes = []


    # Declare the training criterion, optimizer and scheduler
    if isinstance(model, GNN):
        # l1 loss with reduction sum between pred and batch.y_energy
        criterion = lambda pred, batch: F.l1_loss(pred, batch.y_energy, reduction='sum')

    elif isinstance(model, (QGNN, QGNN2, QGNN_EM, QGNN4, QBP_QGNN)):
        # criterion should involve both, the energies and the rdms
        # `pred` returns x_nodes, x_edges, and x_global
        # use `rdm_from_bloch_vec` to get the rdms from x_nodes and x_edges
        # compare `x_node_rdms` to `batch.y_node_rdms`, `x_edge_rdms` to `batch.y_edge_rdms`, and `x_global` to `batch.y_energy`

        def criterion(pred, batch):
            
            if isinstance(model, (QGNN4, QBP_QGNN)):
                x_global, x_nodes, x_edges = pred
            else:
                x_nodes, x_edges, x_global = pred

            #For batch size 1, x_global is a scalar, but batch.y_energy is a tensor of size 1
            if x_global.dim() == 0 and batch.y_energy.dim() == 1:
                x_global = x_global.unsqueeze(0)


            loss_energy = F.l1_loss(x_global, batch.y_energy, reduce='sum')


            if not isinstance(model, (QGNN_EM, QGNN4, QBP_QGNN)):
                x_node_rdms = rdm_from_bloch_vec(x_nodes)
                x_edge_rdms = rdm_from_bloch_vec(x_edges)
            else:
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
            energy_len = int(np.round(100 * loss_energy.item() / loss.item()))
            node_len = int(np.round(100 * loss_node.item() / loss.item()))
            edge_len = 100 - energy_len - node_len
            loss_str = "+" * energy_len + "o" * node_len + "x" * edge_len
            # print(f"loss: {loss.item():8.4f} | {loss_str}")
            

            return loss, loss_energy, loss_node, loss_edge
        
    elif isinstance(model, TensorNetworkRunner):
        def criterion(pred, batch):
            x_nodes, x_edges, x_global = pred

            loss_energy = F.l1_loss(x_global, batch.y_energy, reduce='sum')
        
            x_node_rdms = np.concatenate(x_nodes, axis=0)
            x_node_rdms = torch.tensor(x_node_rdms, dtype=torch.float32)
            y_node_rdms = np.concatenate(batch.y_node_rdms, axis=0)
            y_node_rdms = torch.tensor(y_node_rdms, dtype=torch.float32)

            loss_node = quantum_loss(x_node_rdms, y_node_rdms, trace_distance)
            # print("loss node",fid_node.item())

            x_edge_rdms = np.concatenate(x_edges, axis=0)
            x_edge_rdms = torch.tensor(x_edge_rdms, dtype=torch.float32)    
            y_edge_rdms = np.concatenate(batch.y_edge_rdms, axis=0)
            y_edge_rdms = torch.tensor(y_edge_rdms, dtype=torch.float32)

            #For all interaction terms with value 0, the 2rdm should not contribute to the loss
            interaction_terms = batch.x_edges
            #Remove all rdms that correspond to interaction terms with value 0, use torch.where to get the indices
            #of the interaction terms with value 0
            indices = torch.where(interaction_terms != 0)[0]
            x_edge_rdms = torch.index_select(x_edge_rdms, 0, indices)
            y_edge_rdms = torch.index_select(y_edge_rdms, 0, indices)

            loss_edge = quantum_loss(x_edge_rdms, y_edge_rdms, trace_distance)
            # print("loss edge",fid_edge.item())

            loss = loss_energy + (loss_node + loss_edge)

            # print(f"loss energy: {loss_energy.item():8.4f} | fid_node: {fid_node.item():8.4f} | fid_edge: {fid_edge.item():8.4f}")
            # construct a string of length 100 with + for energy, o for node and x for edge, representing the respective percentage of the loss
            energy_len = int(np.round(100 * loss_energy.item() / loss.item()))
            node_len = int(np.round(100 * loss_node.item() / loss.item()))
            edge_len = 100 - energy_len - node_len
            loss_str = "+" * energy_len + "o" * node_len + "x" * edge_len
#            print(f"loss: {loss.item():8.4f} | {loss_str}")

            return loss, loss_energy, loss_node, loss_edge

    else:
        raise NotImplementedError(f'WTF-Error: Model type {type(model)}')
    
    if not isinstance(model, TensorNetworkRunner) and not isinstance(model, (QBP_QGNN, QGNN4)):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif isinstance(model, (QGNN4)):
        scheduler = None
        if args.use_lbfgs:
            optimizer = torch.optim.LBFGS(model.parameters(), lr=default_LBFGS_optimizer_params['lr_tensor'], 
                                               history_size=default_LBFGS_optimizer_params['history_size'], 
                                               tolerance_change=default_LBFGS_optimizer_params['tolerance_change'],
                                               tolerance_grad=default_LBFGS_optimizer_params['tolerance_grad'], 
                                               line_search_fn=default_LBFGS_optimizer_params['line_search_fn'])
        else:
            #We want two learning rates for different parameters of the optimizer, the params with tensor in their name and the rest
            # We can use named_parameters() to get the names of the parameters
            model_parameters = []
            tensor_parameters = []
            for name, param in model.named_parameters():
                if 'tensor' in name:
                    tensor_parameters.append(param)
                else:
                    model_parameters.append(param)
            optimizer = torch.optim.Adam([{'params': model_parameters, 'lr': args.learning_rate}, 
                                        {'params': tensor_parameters, 'lr': args.lr_tensor}])


    print(isinstance(model, (QBP_QGNN, QGNN4)))
    # Saving the best model instance based on validation MAE
    best_train_mae = np.inf
    best_val_mae = np.inf
    model_path = None
    ckpt_path = None
    ckpt = None

    # Skip training if the evaluate parameter is set.
    skip_train = args.evaluate is not None
    if isinstance(model, (TensorNetworkRunner, QBP_QGNN)):  
         skip_train = True

    # If evaluate is set, load the model and evaluate it.
    if not skip_train:
        print('Beginning training...')
        try:
            
            with tqdm(range(args.epochs)) as t:
                start_time = time.time()
                for epoch in t:
                    t.set_description(f'Epoch {epoch}')
                    start = time.time()
                    if not isinstance(model, QGNN4):
                        epoch_train_mae, energy_mae, node_mae, edge_mae = train(model, train_loader, criterion, optimizer, device, unroll_batch=isinstance(model, (QBP_QGNN, QGNN4)))
                    else:
                        epoch_train_mae, energy_mae, node_mae, edge_mae = train_qgnn(model, train_loader, criterion, device, optimizer, use_lbfgs=isinstance(optimizer, torch.optim.LBFGS), LBFGS_params = default_LBFGS_optimizer_params,   use_rdms_loss = args.use_rdms_loss)

                    epoch_val_mae, energy_mae_val, node_mae_val, edge_mae_val, _ , _= evaluate(model, val_loader, criterion, device, unroll_batch=isinstance(model, (QBP_QGNN, QGNN4)))
                    print(f'Epoch {epoch} | Train MAE: {round(epoch_train_mae, 3)} | ' + 
                            f'Val MAE: {round(epoch_val_mae, 3)}' + 
                            f'Train Energy MAE: {round(energy_mae, 3)} | ' + 
                            f'Val Energy MAE: {round(energy_mae_val, 3)} | ')
                    

                    if args.use_wandb:
                        wandb.log({'Train MAE': epoch_train_mae,
                                    'Validation MAE': epoch_val_mae,
                                    'Train Energy MAE': energy_mae,
                                    'Train Node MAE': node_mae,
                                    'Train Edge MAE': edge_mae,
                                    'Validation Energy MAE': energy_mae_val,
                                    'Validation Node MAE': node_mae_val,
                                    'Validation Edge MAE': edge_mae_val,
                                    })
                    else:
                        train_maes.append(epoch_train_mae)
                        valid_maes.append(epoch_val_mae)
                        train_energy_maes.append(energy_mae)
                        train_node_maes.append(node_mae)
                        train_edge_maes.append(edge_mae)
                        valid_energy_maes.append(energy_mae_val)
                        valid_node_maes.append(node_mae_val)    
                        valid_edge_maes.append(edge_mae_val)



                    # Best model based on validation MAE
                    if epoch_val_mae < best_val_mae:
                        best_val_mae = epoch_val_mae
                        if args.use_wandb:
                            wandb.run.summary["best_val_mae"] = best_val_mae
                        ckpt = {"state_dict": copy.deepcopy(model.state_dict()),
                                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                                "best_mae": best_val_mae,
                                "best_epoch": epoch}

                        # model path appends run_name with other details
                        model_path = f'{run_name}' \
                                     f'_nl-{args.num_layers}' \
                                     f'_hc-{args.hidden_channels}' \
                                     f'_bs-{args.batch_size}' \
                                     f'_lr-{args.learning_rate}' \
                                     f'_seed-{args.seed}.pt'

                        # Save the model to the saved_models directory
                        saved_models_dir = os.path.join(script_dir, 'output/saved_models')

                        # If the directory does not exist, create it
                        if not os.path.exists(saved_models_dir):
                            os.makedirs(saved_models_dir)
                        ckpt_path = os.path.join(saved_models_dir, model_path)

                    if scheduler is not None:
                        # Perform LR step
                        scheduler.step()

                    # Update the postfix of tqdm with every iteration
                    t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_mae, val_loss=epoch_val_mae)
                
                training_time_epoch = time.time() - start_time / args.epochs
                if args.use_wandb:
                    wandb.log({'Training Time per Epoch': training_time_epoch})
        except KeyboardInterrupt:
            # Training can be safely interrupted with Ctrl+C
            torch.save(ckpt, ckpt_path)
            print('Exiting training early because of keyboard interrupt.')

        if ckpt_path is not None and ckpt is not None:
            torch.save(ckpt, ckpt_path)

    # Plot if not using wandb
    if not args.use_wandb and not skip_train:
        # Make 4 plots for the 4 different MAEs
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(train_maes, label='Train MAE')
        axs[0, 0].plot(valid_maes, label='Validation MAE')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('MAE')
        axs[0, 0].set_title('Total MAE')
        axs[0, 0].legend()

        axs[0, 1].plot(train_energy_maes, label='Train Energy MAE')
        axs[0, 1].plot(valid_energy_maes, label='Validation Energy MAE')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('MAE')
        axs[0, 1].set_title('Energy MAE')
        axs[0, 1].legend()

        axs[1, 0].plot(train_node_maes, label='Train Node MAE')
        axs[1, 0].plot(valid_node_maes, label='Validation Node MAE')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('MAE')
        axs[1, 0].set_title('Node MAE')
        axs[1, 0].legend()

        axs[1, 1].plot(train_edge_maes, label='Train Edge MAE')
        axs[1, 1].plot(valid_edge_maes, label='Validation Edge MAE')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('MAE')
        axs[1, 1].set_title('Edge MAE')
        axs[1, 1].legend()

        plt.show()


    saved_models_dir = os.path.join(script_dir, 'output/saved_models')
    if not skip_train:
        # If the training is skipped, load the model from the saved_models directory
        print('Loading best model...')
        if not isinstance(model, (TensorNetworkRunner, QBP_QGNN)) and args.use_wandb:
            # Log this checkpoint as an artifact in wandb
            if ckpt_path is None:
                raise TypeError('Checkpoint path not found.')
            artifact = wandb.Artifact('model_checkpoint', type='model')
            artifact.add_file(ckpt_path)
            artifact.metadata['epoch'] = ckpt['best_epoch']  # Add epoch info as metadata
            artifact.metadata['mae'] = ckpt['best_mae']  # Add best mae info as metadata
            wandb.run.log_artifact(artifact)
    elif not isinstance(model, (TensorNetworkRunner, QBP_QGNN)):
        # Otherwise, load the model from training from the saved_models directory
        model_path = os.path.join(saved_models_dir, args.evaluate)
        if not os.path.exists(model_path):
            raise TypeError(f'Model path not recognized: {model_path}')
        print(f'Loading model with weights stored at {model_path}...')

    if not isinstance(model, (TensorNetworkRunner, QBP_QGNN)):
        # Load the model
        ckpt = torch.load(os.path.join(saved_models_dir, model_path), map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    # Perform evaluation on test set
    print('\nBeginning evaluation...')
    if not isinstance(model, QBP_QGNN):
        test_mae, test_mae_energy, test_mae_node, test_mae_edge, evaluation_time, model_outputs = evaluate(model, test_loader, criterion, device, unroll_batch=isinstance(model, (QGNN4)), return_outputs = model_output_path != None)
    else:
        #Args to dict
        model_args = vars(args)
        test_mae, test_mae_energy, test_mae_node, test_mae_edge, evaluation_time, model_outputs = evaluate_qbp(model, test_loader, criterion, device, model_args, max_iter=20, return_outputs = model_output_path != None, tol=args.tol)

    if args.save_model_outputs:
        print(f'Saving model outputs to {model_output_path}')
        if not model_output_path.endswith('.pt'):
             model_output_path += '.pt'
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        torch.save(model_outputs, model_output_path)
    
    if args.use_wandb:
        wandb.log({'Avg. Inference time': evaluation_time})
        wandb.run.summary["Test MAE"] = test_mae
        wandb.run.summary["Test Energy MAE"] = test_mae_energy
        wandb.run.summary["Test Node MAE"] = test_mae_node
        wandb.run.summary["Test Edge MAE"] = test_mae_edge
        wandb.finish()

    print(f'\nTest MAE: {round(test_mae, 3)}')
    print(f'Test Energy MAE: {round(test_mae_energy, 3)}')
    print(f'Test Node MAE: {round(test_mae_node, 3)}')
    print(f'Test Edge MAE: {round(test_mae_edge, 3)}')
    print(f'Evaluation time: {evaluation_time:8.4f}')
    print('Evaluation finished. Exiting...')
