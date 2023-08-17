import os
import re

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import copy
import time
import json
import numpy as np
import argparse

from tqdm import tqdm
from pprint import pprint
from torch_geometric.loader import DataLoader
from dataset.isingModel import IsingModelDataset
from train import train, evaluate

# Plotting via wandb
import wandb

script_dir = os.path.dirname(__file__)

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


def parse_options():
    """Function for parsing command line arguments."""
    parser = argparse.ArgumentParser("Model runner.")

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

    # General Training parameters
    parser.add_argument('--model', type=str, default='egnn', metavar='S',
                        help='Available models: egnn | mpnn ')
    parser.add_argument('--dataset', type=str, default='IsingModel', metavar='S',
                        help='Available datasets: IsingModel')
    parser.add_argument('--dataset_path', type=str, default='C:\\Users\\gerar_0ev1q4m\\OneDrive\\Documents\\AI\\QGNN\\src\\QGNN\\data\\nk_(12,)_False.npy', metavar='S',
                        help='Available datasets: IsingModel')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Random seed')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='N',
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='clamp the output of the coords function if get too large')
    

    # Network specific parameters
    parser.add_argument('--in_channels', type=int, default=11, metavar='N',
                        help='Input dimension of features')
    parser.add_argument('--hidden_channels', type=int, default=128, metavar='N',
                        help='Hidden dimensions')
    parser.add_argument('--num_layers', type=int, default=7, metavar='N',
                        help='Number of model layers')
    parser.add_argument('--out_channels', type=int, default=1, metavar='N',
                        help='Output dimensions')

    # Funky experimentation with a lot of abstraction headache
    parser.add_argument('--include_dist', action='store_true',
                        help='Whether or not to include distance '
                             'in the message state. (default: False)')

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



def get_dataset(dataset_name, dataset_path):
    if dataset_name == "IsingModel":
        return IsingModelDataset(root=os.path.dirname(dataset_path), data_file=os.path.basename(dataset_path), flatten=True)
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



def get_model(model_name):
    """Gets the corresponding model according to the specified run args."""
    if model_name == 'egnn':
        from models.egnn import EGNN
        return EGNN
    else:
        raise NotImplementedError(f'Model name {model_name} not recognized.')

def main(args):
    """Main function."""
    # Display run arguments
    pprint(args)
    print()

    # Set the hardware accelerator
    device = setup_gpu()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Get the dataset object
    dataset = get_dataset(args.dataset, args.dataset_path)

    # Split the dataset into train, val and test
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Initialize the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    model = get_model(args.model)
    model = model(**vars(args)).to(device)

    # Number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}\n')

    # Setting the WandB parameters
    config = {
        **vars(args),
        'num_params': num_params
    }
    run_name = f'{args.model}_{args.dataset}' \
               f'_epochs-{args.epochs}_num_layers-{args.num_layers}'

    # Initialize the wandb run
    wandb.init(project="qgnn-benchmark-exp", config=config, reinit=True,
               name=run_name)
    wandb.watch(model)

    # Declare the training criterion, optimizer and scheduler
    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Saving the best model instance based on validation MAE
    best_train_mae, best_val_mae, model_path = float('inf'), float('inf'), ""

    # Calculate the mean and mad of the dataset
    values = [torch.squeeze(graph.y[:, 1]) for graph in train_loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    mean, mad = mean.to(device), mad.to(device)

    # Skip training if the evaluate parameter is set.
    skip_train = args.evaluate is not None

    # If evaluate is set, load the model and evaluate it.
    if not skip_train:
        print('Beginning training...')
        try:
            with tqdm(range(args.epochs)) as t:
                for epoch in t:
                    t.set_description(f'Epoch {epoch}')
                    start = time.time()
                    epoch_train_mae = train(model, train_loader, criterion, optimizer, device, mean, mad)
                    epoch_val_mae = evaluate(model, val_loader, criterion, device, mean, mad)

                    wandb.log({'Train MAE': epoch_train_mae,
                               'Validation MAE': epoch_val_mae})

                    # Best model based on validation MAE
                    if epoch_val_mae < best_val_mae:
                        best_val_mae = epoch_val_mae
                        wandb.run.summary["best_val_mae"] = best_val_mae
                        ckpt = {"state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_mae": best_val_mae,
                                "best_epoch": epoch}

                        # model path appends run_name with other details
                        model_path = f'{run_name}' \
                                     f'_in_c-{args.in_channels}' \
                                     f'_h_c-{args.hidden_channels}' \
                                     f'_o_c-{args.out_channels}' \
                                     f'_bs-{args.batch_size}' \
                                     f'_lr-{args.learning_rate}' \
                                     f'_seed-{args.seed}.pt'

                        # Save the model to the saved_models directory
                        saved_models_dir = os.path.join(script_dir, 'saved_models')

                        # If the directory does not exist, create it
                        if not os.path.exists(saved_models_dir):
                            os.makedirs(saved_models_dir)
                        torch.save(ckpt, os.path.join(saved_models_dir, model_path))

                    # Perform LR step
                    scheduler.step()

                    # Update the postfix of tqdm with every iteration
                    t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_mae, val_loss=epoch_val_mae)
        except KeyboardInterrupt:
            # Training can be safely interrupted with Ctrl+C
            print('Exiting training early because of keyboard interrupt.')

    saved_models_dir = os.path.join(script_dir, 'saved_models')
    if not skip_train:
        # If the training is skipped, load the model from the saved_models directory
        print('Loading best model...')
    else:
        # Otherwise, load the model from training from the saved_models directory
        model_path = os.path.join(saved_models_dir, args.evaluate)
        if not os.path.exists(model_path):
            raise TypeError(f'Model path not recognized: {model_path}')
        print(f'Loading model with weights stored at {model_path}...')

    # Load the model
    ckpt = torch.load(os.path.join(saved_models_dir, model_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # Perform evaluation on test set
    print('\nBeginning evaluation...')
    test_mae = evaluate(model, test_loader, criterion, device, mean, mad)
    wandb.run.summary["test_mae"] = test_mae
    print(f'\nTest MAE: {round(test_mae, 3)}')
    print('Evaluation finished. Exiting...')


if __name__ == '__main__':
    args = parse_options()
    main(args)
