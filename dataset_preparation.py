import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from medmnist import PneumoniaMNIST, BloodMNIST
import os
import numpy as np

# --- Step 1: Define a Custom Combined Dataset Class ---
class CombinedImageDataset(Dataset):
    """
    A custom PyTorch Dataset that combines two datasets:
    1. A non-medical dataset (e.g., CIFAR-10), labeled as 0.
    2. A medical dataset (e.g., MedMNIST combination), labeled as 1.
    """
    def __init__(self, non_medical_dataset, medical_dataset):
        self.non_medical_dataset = non_medical_dataset
        self.medical_dataset = medical_dataset
        
        # Calculate the total size of the combined dataset
        self.total_size = len(self.non_medical_dataset) + len(self.medical_dataset)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Determine which dataset to sample from based on the index
        non_medical_len = len(self.non_medical_dataset)
        
        if idx < non_medical_len:
            # It's a non-medical image from the CIFAR-10 dataset
            image, _ = self.non_medical_dataset[idx]
            label = 0  # 0 for non-medical
        else:
            # It's a medical image from the combined MedMNIST dataset
            medical_idx = idx - non_medical_len
            image, _ = self.medical_dataset[medical_idx]
            label = 1  # 1 for medical

        return image, label

# --- Step 2: Prepare and Combine the Datasets ---
def prepare_datasets():
    """
    Downloads and prepares all datasets, then combines them into a
    single training and validation dataset for the model.
    """
    # Define a standard image transformation pipeline
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    medical_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channels for ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 1. Non-medical images (CIFAR-10)
    cifar10_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=data_transform)
    cifar10_val_dataset = CIFAR10(root='./data', train=False, download=True, transform=data_transform)

    # 2. Medical images (Combining multiple MedMNIST datasets)
    print("Downloading and preparing MedMNIST datasets...")
    pneumonia_train = PneumoniaMNIST(split='train', transform=medical_transform, download=True)
    pneumonia_val = PneumoniaMNIST(split='test', transform=medical_transform, download=True)

    blood_train = BloodMNIST(split='train', transform=medical_transform, download=True)
    blood_val = BloodMNIST(split='test', transform=medical_transform, download=True)
    
    # Concatenate the medical datasets
    medical_train_dataset = ConcatDataset([pneumonia_train, blood_train])
    medical_val_dataset = ConcatDataset([pneumonia_val, blood_val])

    # --- Step 3: Combine all datasets into final training and validation sets ---
    print("Combining medical and non-medical datasets...")
    combined_train_dataset = CombinedImageDataset(cifar10_train_dataset, medical_train_dataset)
    combined_val_dataset = CombinedImageDataset(cifar10_val_dataset, medical_val_dataset)
    
    return combined_train_dataset, combined_val_dataset

if __name__ == "__main__":
    try:
        import torch, torchvision, medmnist
    except ImportError:
        print("Please install required libraries: pip install torch torchvision medmnist")
        exit()

    train_dataset, val_dataset = prepare_datasets()
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n--- Dataset Summary ---")
    print(f"Total Combined Training Dataset Size: {len(train_dataset)} samples")
    print(f"Total Combined Validation Dataset Size: {len(val_dataset)} samples")
    print(f"Total number of batches in training loader: {len(train_loader)}")
    print(f"Total number of batches in validation loader: {len(val_loader)}")

    first_image, first_label = train_dataset[0]
    print(f"\nFirst image tensor shape: {first_image.shape}")
    print(f"First image label (0=non-medical, 1=medical): {first_label}")
