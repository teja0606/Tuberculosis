# src/data_setup.py
#
# Contains PyTorch Dataset and DataLoader logic for loading and transforming
# the Tuberculosis image data. This is where we define our data augmentation
# and preprocessing pipelines.

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import logging

# --- Configuration ---
# Define image size and batch size, which can be imported by other scripts
IMG_SIZE = 224
BATCH_SIZE = 32 # A good starting point, can be tuned based on VRAM

def get_data_transforms():
    """
    Defines and returns the transformation pipelines for training and validation/testing.
    
    The training pipeline includes aggressive data augmentation to help the model
    generalize better and prevent overfitting.
    
    The validation/test pipeline only performs the necessary preprocessing steps
    to ensure a consistent evaluation.
    
    Returns:
        tuple: A tuple containing the training transforms and validation/test transforms.
    """
    
    # --- Transformation for Validation and Testing Sets ---
    # Only includes essential preprocessing
    valid_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # Convert to grayscale
        transforms.Resize((IMG_SIZE, IMG_SIZE)),     # Resize to a uniform size
        transforms.ToTensor(),                       # Convert image to a PyTorch Tensor (values from 0 to 1)
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize tensor values to [-1, 1]
    ])
    
    # --- Transformation for Training Set ---
    # Includes aggressive data augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # --- Augmentation Layers ---
        transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images horizontally
        transforms.RandomRotation(degrees=10),  # Randomly rotate images by up to 10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Randomly shift and zoom
        # --- End Augmentation ---
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, valid_test_transform

def create_dataloaders(data_dir: str, train_transform: transforms.Compose, test_transform: transforms.Compose, batch_size: int):
    """
    Creates training, validation, and testing DataLoaders.

    This function takes the main data directory, splits the data into
    training, validation, and testing sets, and creates PyTorch DataLoaders
    for each.

    Args:
        data_dir (str): The path to the root data directory (e.g., "data/").
        train_transform (transforms.Compose): The transformation pipeline for the training data.
        test_transform (transforms.Compose): The transformation pipeline for the validation/testing data.
        batch_size (int): The number of samples per batch in each DataLoader.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders, and class names.
    """
    
    # --- 1. Load the full dataset with ImageFolder ---
    # ImageFolder automatically finds class folders and assigns labels
    full_dataset = datasets.ImageFolder(data_dir)
    class_names = full_dataset.classes
    
    # --- 2. Define split sizes ---
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.15 * total_size)   # 15% for validation
    test_size = total_size - train_size - val_size # Remaining 15% for testing
    
    # --- 3. Split the dataset ---
    # Use a fixed generator for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # --- 4. Assign the correct transforms to each subset ---
    # This is a crucial step. The training subset gets the augmentation pipeline,
    # while validation and test subsets get the simple preprocessing pipeline.
    # We achieve this by accessing the 'dataset' attribute of the Subset object.
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = test_transform
    test_subset.dataset.transform = test_transform

    # --- 5. Create the DataLoaders ---
    train_dataloader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True, # Shuffle training data to ensure model sees varied batches
        num_workers=os.cpu_count() // 2 # Use multiple CPU cores to load data faster
    )
    
    val_dataloader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=os.cpu_count() // 2
    )
    
    test_dataloader = DataLoader(
        dataset=test_subset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=os.cpu_count() // 2
    )

    logging.info(f"Created DataLoaders with {len(train_dataloader)} training batches, "
                 f"{len(val_dataloader)} validation batches, and {len(test_dataloader)} testing batches.")
    
    return train_dataloader, val_dataloader, test_dataloader, class_names

if __name__ == '__main__':
    # This block allows you to test the script directly.
    # Best practice is to run this from the ROOT of your project directory:
    # python -m src.data_setup
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- FIX: Create a robust path to the data directory ---
    # This finds the project's root directory and joins it with 'data'
    # It will work no matter where you run the script from.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    
    logging.info(f"Looking for data in: {DATA_PATH}")

    train_transform, test_transform = get_data_transforms()
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir=DATA_PATH,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE
    )
    
    # Print some info to verify
    print(f"Class names: {classes}")
    img, label = next(iter(train_loader))
    print(f"Image batch shape: {img.shape}")
    print(f"Label batch shape: {label.shape}")
    print(f"Image data type: {img.dtype}")
    print(f"Label data type: {label.dtype}")
