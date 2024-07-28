import time
import torch
from tqdm import tqdm
from baselines import DataPointError
from qbp_gnn import generate_tensor_network
from torchviz import make_dot
from baselines import GNN, TensorNetworkRunner, DMRG, SimpleUpdate, FullUpdate, SimpleUpdateGen, DMRG_QUIMB
from qgnn import QGNN, QGNN2, QGNN_EM
from qbp_gnn import QBP_QGNN
from qbp_gnn import QGNN as QGNN4
import numpy as np


def extract_data_points(model, pred, batch):
    x_nodes, x_edges, x_global = pred
    data = []

    if isinstance(model, TensorNetworkRunner):
        for i in range(len(batch)):
            data_point = batch[i].clone()
            x_node_rdms = x_nodes[i] 
            x_edge_rdms = x_edges[i] 
            x_energy = x_global[i].unsqueeze(0) 
            data_point.x_energy = x_energy
            data_point.x_node_rdms = x_node_rdms
            data_point.x_edge_rdms = x_edge_rdms
            data.append(data_point)
            assert data_point.x_energy.shape == data_point.y_energy.shape, f"Energy shapes do not match: {data_point.x_energy.shape} vs {data_point.y_energy.shape}"
            assert data_point.x_node_rdms.shape == data_point.y_node_rdms.shape, f"Node RDM shapes do not match: {data_point.x_node_rdms.shape} vs {data_point.y_node_rdms.shape}"
            assert data_point.x_edge_rdms.shape == data_point.y_edge_rdms.shape, f"Edge RDM shapes do not match: {data_point.x_edge_rdms.shape} vs {data_point.y_edge_rdms.shape}"
    elif isinstance(model, QBP_QGNN):
        assert len(batch) == 1, "QBP only works for batch size 1"
        data_point = batch[0].clone()
        data_point.x_energy = x_global
        data_point.x_node_rdms = x_nodes
        data_point.x_edge_rdms = x_edges
        data.append(data_point)
        assert data_point.x_energy.shape == data_point.y_energy.shape, f"Energy shapes do not match: {data_point.x_energy.shape} vs {data_point.y_energy.shape}"
        assert data_point.x_node_rdms.shape == data_point.y_node_rdms.shape, f"Node RDM shapes do not match: {data_point.x_node_rdms.shape} vs {data_point.y_node_rdms.shape}"
        assert data_point.x_edge_rdms.shape == data_point.y_edge_rdms.shape, f"Edge RDM shapes do not match: {data_point.x_edge_rdms.shape} vs {data_point.y_edge_rdms.shape}"
    else:
        raise ValueError(f"Model {model} not recognized")
    
    return data
    
    

def train_qgnn(model, loader, criterion, device, optimizer, use_lbfgs=True, LBFGS_params = None, use_rdms_loss = True):
    """Train the model on the training set.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The training set loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use.

    Returns:
        float: The mean absolute error on the training set.
    """
    mae = 0.0
    mae_energy = 0.0
    mae_node = 0.0
    mae_edge = 0.0
    model.train()

    opt_tensor_indices = []
    #Show model named parameters
    i = 0
    for name, param in model.named_parameters():
        if "tn_tensor" in name:
            opt_tensor_indices.append(i)
        i += 1

    for _, batch in enumerate(tqdm(loader)):
        assert len(batch) == 1, "QGNN only works for batch size 1"
        batch = batch.to(device)
        data_point = batch[0]
        

        if not use_lbfgs:
            pred = model(data_point, format_output=True)
            # make_dot(pred[0], show_attrs=True , show_saved=True).render("pred", format="png")
        else:
            def closure():
                optimizer.zero_grad()
                energy, _, _ = model(data_point)  # Obtain outputs from model
                loss = torch.nn.MSELoss()(energy[0], data_point.y_energy[0].to(energy[0].dtype)) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                loss.backward(retain_graph=True)  # Compute gradients#Clip gradients
                
                
                return loss  # Return the computed loss to the optimizer

            optimizer.step(closure)
            with torch.no_grad():
                pred = model(data_point, format_output=True)
            
        # Calculate train loss
        loss = criterion(pred, batch)
        
        if not isinstance(loss, tuple):
            loss_total = loss
            loss_energy = loss
            loss_node = 0
            loss_edge = 0
        else:
            loss_total, loss_energy, loss_node, loss_edge = loss

        mae += loss_total.item()
        mae_energy += loss_energy.item()
        mae_node += loss_node.item()
        mae_edge += loss_edge.item()

        if not use_lbfgs:
            # Delete info on previous gradients
            optimizer.zero_grad()
            if use_rdms_loss:
                loss_total.backward(retain_graph=True)
            else:
                # Propagate & optimizer step
                loss_energy.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
            optimizer.step()
        
        ##Print gradients of all parameters with names
        # for name, param in model.named_parameters():
        #     print(f"Gradients for {name}: {param.grad}")
        #     if param.grad is not None:
        #         if torch.any(torch.isnan(param.grad)):
        #             raise ValueError(f"{name} contains nan: {param.grad}")
        
        # norms = 0
        # num_norms = 0

        # norms_tensor = 0
        # num_norms_tensor = 0
        # with torch.no_grad():
        #     for idx, param in enumerate(optimizer.param_groups[1]['params']):
        #             if param.grad is not None:
        #                 norms_tensor += torch.norm(param.grad)
        #                 num_norms_tensor += 1
            
        #     for idx, param in enumerate(optimizer.param_groups[0]['params']):
        #         if param.grad is not None:
        #             norms += torch.norm(param.grad)
        #             num_norms += 1

        # print(f"Average norm of gradients: {norms/num_norms}")
        # print(f"Average norm of tensor gradients: {norms_tensor/num_norms_tensor}")
    return mae / len(loader.dataset), mae_energy / len(loader.dataset), mae_node / len(loader.dataset), mae_edge / len(loader.dataset)


def evaluate_qbp(model, loader, criterion, device, opt_params, tol = 1e-6, max_iter = 200, bond_dim = 4, return_outputs = False):
    mae = 0.0
    mae_energy = 0.0
    mae_node = 0.0
    mae_edge = 0.0
    model_outputs = []
    model.train()
    start_time = time.time()
    for idx, batch in enumerate(tqdm(loader)):
        assert len(batch) == 1, "QBP only works for batch size 1"
        batch = batch.to(device)
        energies = []
        one_rdms = []
        two_rdms = []
        try:
            data_point = batch[0]
            shape = data_point.grid_extent
            if len(shape) == 1:
                shape = (shape[0], 1)
            Lx, Ly = shape
            tn_rand, _, tn_type = generate_tensor_network(Lx, Ly, bond_dim=bond_dim, pbc=data_point.pbc, dtype=model.tensor_dtype, normalize = False)
            model.set_datapoint(tn_rand, tn_type=tn_type, datapoint=data_point)
            model = model.to(device)


            optimizer = torch.optim.LBFGS(model.parameters(), lr=opt_params['lr_tensor'], 
                                history_size=opt_params['history_size'],
                                    tolerance_change=opt_params['tolerance_change'],
                                    tolerance_grad=opt_params['tolerance_grad'],
                                    line_search_fn=opt_params['line_search_fn'], max_iter=10)
            curr_change = torch.inf
            iter = 0

            def closure():
                optimizer.zero_grad()
                energy, _, _ = model()  # Obtain outputs from model
                loss = energy  # In this case, the 'energy' itself is used as the loss
                loss.backward(retain_graph=True)  # Compute gradients

                return loss  # Return the computed loss to the optimizer
            
            
            curr_energy = 0
            while curr_change > tol:
                # Clear gradients at each step
                optimizer.step(closure)
                with torch.no_grad():
                    energy, one_rdms, two_rdms = model(format_output=True)

                # optimizer.zero_grad()
                # energy, one_rdms_bp, two_rdms_bp = model()
                # energy.backward(retain_graph=True)
                # optimizer.step() 
                
                with torch.no_grad():
                    for tensor in optimizer.param_groups[0]['params']:
                        tensor = tensor / torch.norm(tensor)

                curr_change = torch.abs(energy - curr_energy)
                # print(f"Energy: {energy[0].item()}, Change: {curr_change[0].item()}")
                curr_energy = energy
                iter += 1
                if iter > max_iter:
                    break                

        except DataPointError as e:
            print(f"DataPointError: in batch {idx}, datapoint {e.index} with message {e.message}")
            break

        loss = criterion((energy, one_rdms, two_rdms), batch)
        if return_outputs:
            model_outputs += extract_data_points(model, (one_rdms, two_rdms, energy), batch)

        
        if not isinstance(loss, tuple):
            loss_total = loss
            loss_energy = loss
            loss_node = 0
            loss_edge = 0
        else:
            loss_total, loss_energy, loss_node, loss_edge = loss


        mae += loss_total.item()
        mae_energy += loss_energy.item()
        mae_node += loss_node.item()
        mae_edge += loss_edge.item()
        
    end_time = time.time()
    average_time = (end_time - start_time) / len(loader.dataset)


    return mae / len(loader.dataset), mae_energy / len(loader.dataset), mae_node / len(loader.dataset), mae_edge / len(loader.dataset), average_time, model_outputs

        
    

def evaluate(model, loader, criterion, device, unroll_batch = False, return_outputs = False):

    """Evaluate the model on the validation/test set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The validation set loader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use.

    Returns:
        float: The mean absolute error on the validation set.
    """
    mae = 0.0
    mae_energy = 0.0
    mae_node = 0.0
    mae_edge = 0.0

    model_outputs = []

    model.eval()
    start_time = time.time()
    for idx, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)

        try:
            # Perform forward pass
            if unroll_batch:
                assert len(batch) == 1, "Unrolling batch only works for batch size 1"
                batch_unroll = batch[0]
                pred = model(batch_unroll, format_output=True)
            else:
                pred = model(batch)
        except DataPointError as e:
            print(f"DataPointError: in batch {idx}, datapoint {e.index} with message {e.message}")
            break

        loss = criterion(pred, batch)

        if return_outputs:
            model_outputs += extract_data_points(model, pred, batch)


        if not isinstance(loss, tuple):
            loss_total = loss
            loss_energy = loss
            loss_node = 0
            loss_edge = 0
        else:
            loss_total, loss_energy, loss_node, loss_edge = loss


        mae += loss_total.item()
        mae_energy += loss_energy.item()
        mae_node += loss_node.item()
        mae_edge += loss_edge.item()
    
    end_time = time.time()
    average_time = (end_time - start_time) / len(loader.dataset)


    return mae / len(loader.dataset), mae_energy / len(loader.dataset), mae_node / len(loader.dataset), mae_edge / len(loader.dataset), average_time, model_outputs

def train(model, loader, criterion, optimizer, device, unroll_batch = False):
    """Train the model on the training set.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The training set loader.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use.

    Returns:
        float: The mean absolute error on the training set.
    """
    mae = 0.0
    mae_energy = 0.0
    mae_node = 0.0
    mae_edge = 0.0
    model.train()
    for _, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)

        # Perform forward pass
        if unroll_batch:
            assert len(batch) == 1, "Unrolling batch only works for batch size 1"
            batch_unroll = batch[0]
            pred = model(batch_unroll, format_output=True)
        else:
            pred = model(batch)

        # Calculate train loss
        loss = criterion(pred, batch)
         
        if not isinstance(loss, tuple):
            loss_total = loss
            loss_energy = loss
            loss_node = 0
            loss_edge = 0
        else:
            loss_total, loss_energy, loss_node, loss_edge = loss

        mae += loss_total.item()
        mae_energy += loss_energy.item()
        mae_node += loss_node.item()
        mae_edge += loss_edge.item()
        
        # Delete info on previous gradients
        optimizer.zero_grad()

        # Propagate & optimizer step
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Print gradients
        # Check if model has embed_nodes
        if hasattr(model, "embed_nodes"):
            for name, param in model.embed_nodes.named_parameters():
                if param.grad is not None:
                    # print(f"Gradients for embed_nodes.{name}: {param.grad}")
                    # check for nan
                    if torch.any(torch.isnan(param.grad)):
                        raise ValueError(f"embed_nodes.{name} contains nan: {param.grad}")

        optimizer.step()

    return mae / len(loader.dataset), mae_energy / len(loader.dataset), mae_node / len(loader.dataset), mae_edge / len(loader.dataset)
