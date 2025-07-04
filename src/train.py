# src/train.py
#
# This is the main script to start the training process.
# It brings together the data setup, model architecture, and training engine
# to train our custom Tuberculosis detection model.
#
# To run: python -m src.train

import os
import torch
from torch import nn
import logging
import multiprocessing

# Use relative imports for custom modules within the same package
from .data_setup import create_dataloaders, get_data_transforms
from .model import HybridTBNet
from .engine import train
from .utils import save_model

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Training Hyperparameters
NUM_EPOCHS = 10 # Start with a smaller number of epochs to test the pipeline
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4 # Regularization parameter

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")

def main():
    """
    Main function to orchestrate the training process.
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

    # --- 1. Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 2. Create DataLoaders ---
    logging.info("Creating DataLoaders...")
    train_transform, test_transform = get_data_transforms()
    train_dataloader, val_dataloader, test_dataloader, class_names = create_dataloaders(
        data_dir=DATA_DIR,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=32 # BATCH_SIZE is defined in data_setup, but we can override here if needed
    )
    logging.info(f"DataLoaders created. Classes: {class_names}")

    # --- 3. Initialize Model ---
    logging.info("Initializing the model...")
    model = HybridTBNet(
        input_channels=1,
        cnn_output_channels=128,
        transformer_heads=8,
        transformer_dropout=0.1,
        mlp_dropout=0.5
    ).to(device)

    # --- 4. Define Loss Function and Optimizer ---
    # BCEWithLogitsLoss is numerically stable and recommended for binary classification
    loss_fn = nn.BCEWithLogitsLoss()

    # AdamW is a good choice for models with transformers and includes weight decay
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    # --- 5. Start Training ---
    logging.info(f"Starting training for {NUM_EPOCHS} epochs...")
    results = train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=val_dataloader, # Use validation set for monitoring during training
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)

    logging.info("Training complete.")

    # --- 6. Save the Trained Model ---
    logging.info(f"Saving model to: {MODEL_SAVE_PATH}")
    save_model(model=model,
               target_dir=os.path.dirname(MODEL_SAVE_PATH),
               model_name=os.path.basename(MODEL_SAVE_PATH))

    logging.info("Script finished successfully.")


if __name__ == '__main__':
    # --- FIX: Protect the main execution logic ---
    # This is crucial for multiprocessing on Windows. It prevents child processes
    # from re-running the main script, which would cause a RuntimeError.
    multiprocessing.freeze_support() # Good practice for Windows
    main()
