# src/engine.py
#
# Contains the core functions for training and evaluating the PyTorch model.
# This modular approach keeps our main training script clean and readable.

import torch
from torch import nn
from tqdm.auto import tqdm # For a nice progress bar
from typing import Tuple

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Performs a single training step for one epoch.

    Iterates through the dataloader, performs forward and backward passes,
    and updates the model's weights.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the training data.
        loss_fn (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The target device (e.g., "cuda" or "cpu").

    Returns:
        A tuple containing the average training loss and training accuracy.
    """
    # Put model in training mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_logits = model(X).squeeze() # Squeeze to remove extra dimension
        loss = loss_fn(y_logits, y.float()) # Loss function expects float labels
        train_loss += loss.item()

        # 2. Optimizer zero grad
        optimizer.zero_grad()

        # 3. Loss backward
        loss.backward()

        # 4. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.round(torch.sigmoid(y_logits))
        train_acc += (y_pred_class == y).sum().item() / len(y_logits)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Performs a single evaluation step on a testing or validation dataset.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the test/validation data.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The target device.

    Returns:
        A tuple containing the average test loss and test accuracy.
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X).squeeze()

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y.float())
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: torch.device) -> dict:
    """
    The main training function that orchestrates the training process.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing/validation set.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn (nn.Module): The loss function.
        epochs (int): The number of epochs to train for.
        device (torch.device): The target device.

    Returns:
        A dictionary containing the training and testing loss and accuracy for each epoch.
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
