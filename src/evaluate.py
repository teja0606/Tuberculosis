# src/evaluate.py
#
# This script loads the trained model and evaluates its performance on the
# unseen test set. It calculates key metrics and saves a classification
# report and a confusion matrix plot.
#
# To run: python -m src.evaluate

import os
import torch
from torch import nn
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Import our custom modules
from .data_setup import create_dataloaders, get_data_transforms
from .model import HybridTBNet

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create output directories if they don't exist
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

def evaluate_model():
    """
    Loads a trained model and evaluates it on the test dataset.
    """
    # --- 1. Setup Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # --- 2. Load the Test DataLoader ---
    # We only need the test set for final evaluation
    logging.info("Creating Test DataLoader...")
    _, test_transform = get_data_transforms()
    _, _, test_dataloader, class_names = create_dataloaders(
        data_dir=DATA_DIR,
        train_transform=None, # Not needed
        test_transform=test_transform,
        batch_size=32
    )
    logging.info(f"Test DataLoader created. Classes: {class_names}")

    # --- 3. Load the Trained Model ---
    logging.info(f"Loading model from: {MODEL_PATH}")
    # Instantiate the model architecture
    model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)
    # Load the saved weights (state_dict)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # --- 4. Evaluate the Model ---
    model.eval()
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            logits = model(X).squeeze()
            
            # Convert logits to predictions (0 or 1)
            preds = torch.round(torch.sigmoid(logits))
            
            # Store true labels and predictions
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    logging.info("Evaluation complete.")

    # --- 5. Generate and Save Classification Report ---
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    report_path = os.path.join(OUTPUT_DIR, "reports", "classification_report.csv")
    report_df.to_csv(report_path)
    
    logging.info("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    logging.info(f"Classification report saved to {report_path}")

    # --- 6. Generate and Save Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    cm_path = os.path.join(OUTPUT_DIR, "plots", "confusion_matrix.png")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix plot saved to {cm_path}")

if __name__ == '__main__':
    evaluate_model()
