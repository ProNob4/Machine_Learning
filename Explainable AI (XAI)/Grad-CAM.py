# Tomato Disease XAI Analysis in Colab
# @title Setup and Installation
!pip install shap lime
!pip install --upgrade scikit-image

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch.nn.functional as F
import shap
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
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

# Update these paths to point to your dataset in Google Drive
dataset_path = "/content/drive/MyDrive/path_to_your_tomato_dataset"  # CHANGE THIS
model_save_path = "/content/drive/MyDrive/tomato_model.pth"

# Verify dataset structure
try:
    classes = os.listdir(dataset_path)
    print(f"Found {len(classes)} classes: {classes}")
except FileNotFoundError:
    print("Dataset directory not found! Please update dataset_path")

# Data transformations with augmentation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(full_dataset.classes)

# Class names mapping
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

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# @title Model Definition
class TomatoDiseaseXAI(nn.Module):
    def __init__(self, num_classes):
        super(TomatoDiseaseXAI, self).__init__()
        # Use pretrained ResNet50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Feature extractor (remove final layers)
        self.features = nn.Sequential(*list(self.base_model.children())[:-2])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        attention_map = self.attention(features)
        weighted_features = features * attention_map
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(weighted_features, (1, 1)).squeeze()
        
        # Handle batch dimension
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
            
        return self.classifier(pooled)

# @title Training Function
def train_model(model, train_loader, test_loader, epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
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
        
        scheduler.step()
        train_acc = 100 * correct / total
        
        # Validation
        test_acc, _ = evaluate(model, test_loader)
        
        print(f'Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'New best model saved with accuracy {best_acc:.2f}%')
    
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

# @title XAI Visualization Functions
def grad_cam(model, input_image, target_class):
    model.eval()
    
    # Register hooks
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Get last conv layer
    target_layer = model.features[-1]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_image)
    model.zero_grad()
    
    # Target for backprop
    target = output[:, target_class].sum() if output.dim() == 2 else output[target_class]
    target.backward()
    
    # Process gradients and activations
    grads = gradients[0].cpu().data.numpy()
    acts = activations[0].cpu().data.numpy()
    
    # Compute weights
    weights = np.mean(grads, axis=(2, 3))
    cam = np.sum(weights * acts, axis=1)[0]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    return cam

def visualize_explanations(model, dataset, num_samples=2):
    model.eval()
    selected_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in selected_indices:
        image, label = dataset[idx]
        input_image = image.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_image)
            _, predicted = torch.max(output, 1)
        
        # Generate Grad-CAM
        cam = grad_cam(model, input_image, label)
        
        # Prepare images
        original_image = image.permute(1, 2, 0).cpu().numpy()
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = 0.6 * heatmap / 255 + 0.4 * original_image
        
        # Plot
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original\nTrue: {class_names[label]}\nPredicted: {class_names[predicted.item()]}")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# @title Main Execution
# Initialize model
model = TomatoDiseaseXAI(num_classes=num_classes).to(device)

# Check if pretrained model exists
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print("Loaded pretrained model")
else:
    print("Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=15)

# Evaluate
test_acc, (true_labels, pred_labels) = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names.values()))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names.values(), 
            yticklabels=class_names.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualize explanations
print("\nGenerating explanations for sample images...")
visualize_explanations(model, test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset)
