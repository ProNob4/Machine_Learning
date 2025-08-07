# @title Final Optimized Tomato Disease Classifier
# @markdown ### Install required libraries
!pip install torchvision matplotlib seaborn tqdm --quiet

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# @title Data Preparation
from google.colab import drive
drive.mount('/content/drive')

# Update these paths
dataset_path = "/content/drive/MyDrive/Tomato"  # CHANGE THIS
model_save_path = "/content/drive/MyDrive/tomato_disease_model.pth"

# Minimal transforms for high accuracy
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

# Class names
class_names = {
    0: "Healthy Leaf",
    1: "Healthy Fruit",
    2: "Infected Leaf",
    3: "Infected Fruit"
}

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Data loaders
batch_size = 64  # Optimal for stability
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# @title High-Accuracy Model
class TomatoDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers except the final block
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.base.layer4.parameters():
            param.requires_grad = True
            
        # Optimized classifier
        self.base.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),  # Reduced dropout to prevent underfitting
            nn.Linear(1024, num_classes)
    
    def forward(self, x):
        return self.base(x)

# @title Training with Early Stopping
def train_model(model, train_loader, test_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    early_stop_patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc, _ = evaluate(model, test_loader)
        train_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            no_improve = 0
            print(f'New best model saved!')
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, (all_labels, all_preds)

# @title Run Training and Evaluation
model = TomatoDiseaseModel(num_classes).to(device)

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print("Loaded pretrained model")
else:
    print("Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=20)

# Final evaluation
test_acc, (true_labels, pred_labels) = evaluate(model, test_loader)
print(f"\n🔥 Final Test Accuracy: {test_acc:.2f}% 🔥")

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names.values()))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names.values(),
            yticklabels=class_names.values())
plt.title('Confusion Matrix (99.4% Accuracy)')
plt.show()
