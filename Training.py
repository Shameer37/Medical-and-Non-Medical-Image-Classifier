import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import os
import numpy as np
# This script assumes dataset_preparation.py is in the same directory.
from dataset_preparation import prepare_datasets

# --- Step 1: Set up the Device and Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- Step 2: Load the Pretrained Model and Modify it ---
def get_model():
    print("Loading pretrained ResNet50 model...")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)
    
    return model

# --- Step 3: Define the Training and Validation Loops ---
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)
        
        val_loss = val_running_loss / val_total_samples
        val_acc = val_running_corrects.double() / val_total_samples
        print(f"Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model!")

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return model

# --- Step 4: Main function to orchestrate the process ---
def main():
    train_dataset, val_dataset = prepare_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model()
    
    trained_model = train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()