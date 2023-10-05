import time
import torch

from tqdm import tqdm

def evaluate(model, loader, criterion, device):
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
    model.eval()
    for _, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch)
        mae += loss.item()

    return mae / len(loader.dataset)

def train(model, loader, criterion, optimizer, device):
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
    model.train()
    for _, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)

        # Perform forward pass
        pred = model(batch)

        # Calculate train loss
        loss = criterion(pred, batch)
        mae += loss.item()

        # Delete info on previous gradients
        optimizer.zero_grad()

        # Propagate & optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return mae / len(loader.dataset)
