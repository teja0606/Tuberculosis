# src/visualize_model.py
#
# This script uses Grad-CAM to create a heatmap visualization, showing which parts
# of an image our model focused on to make a prediction. This is crucial for
# model interpretability.
#
# To run: python -m src.visualize_model

import os
import torch
import logging
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Use the correct import path for the pytorch-grad-cam library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .model import HybridTBNet
from .data_setup import get_data_transforms

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_tb_net_v1.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "grad_cam")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_grad_cam():
    """
    Loads the trained model, picks a random test image, and generates
    a Grad-CAM visualization.
    """
    # --- 1. Setup Device and Load Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    logging.info(f"Loading model from: {MODEL_PATH}")
    model = HybridTBNet(input_channels=1, cnn_output_channels=128).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- 2. Select a Random Image from the Test Set ---
    # We'll pick a random image from the Tuberculosis class to see what the model finds.
    tb_image_dir = os.path.join(DATA_DIR, "Tuberculosis")
    random_image_name = random.choice(os.listdir(tb_image_dir))
    image_path = os.path.join(tb_image_dir, random_image_name)
    logging.info(f"Selected random image for visualization: {image_path}")

    # --- 3. Preprocess the Image ---
    # We need to prepare the image exactly as we did for training/evaluation.
    img = Image.open(image_path).convert("RGB") # Grad-CAM library expects RGB
    
    # Get the standard validation/test transformation pipeline
    _, test_transform_raw = get_data_transforms()
    
    # Create a tensor for the model
    input_tensor = test_transform_raw(img).unsqueeze(0).to(device)

    # --- 4. Set up Grad-CAM ---
    # The target layer is the last convolutional layer in our model's backbone.
    # This is where the most high-level spatial features are located.
    target_layer = model.cnn_backbone[-1] 
    
    # Initialize Grad-CAM
    # --- FIX: Removed the deprecated 'use_cuda' argument ---
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=input_tensor)
    
    # Get the first (and only) heatmap from the batch
    grayscale_cam = grayscale_cam[0, :]
    
    # --- 5. Create and Save the Visualization ---
    # Convert the original image to a format suitable for visualization
    vis_img = img.resize((224, 224))
    # Normalize the image to be in the range [0, 1] for show_cam_on_image
    vis_img_float = np.array(vis_img) / 255.0

    # Overlay the heatmap on the original image
    visualization = show_cam_on_image(vis_img_float, grayscale_cam, use_rgb=True)

    # Plot and save the result
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(vis_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"grad_cam_{random_image_name}")
    plt.savefig(save_path)
    logging.info(f"Grad-CAM visualization saved to: {save_path}")

if __name__ == '__main__':
    visualize_grad_cam()
