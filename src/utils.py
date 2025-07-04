# src/utils.py
#
# Contains utility functions for the project, such as saving the model.
# Keeping these functions separate helps to keep other scripts clean.

import torch
from pathlib import Path
import logging

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: The PyTorch model to save.
        target_dir: The directory to save the model to.
        model_name: The filename for the saved model. Should include ".pth" or ".pt".
    """
    # Create target directory if it doesn't exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict
    logging.info(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

