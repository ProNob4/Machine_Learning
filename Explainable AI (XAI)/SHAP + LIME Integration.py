# @title Install Required Libraries
!pip install shap lime albumentations --quiet
!pip install --upgrade scikit-image numpy torchvision --quiet

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
from tqdm import tqdm

# @title Setup and Configuration
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# @title Data Loading (Adapt to Your Dataset)
from google.colab import drive
drive.mount('/content/drive')

# Update this path
dataset_path = "/content/drive/MyDrive/your_tomato_dataset"  # CHANGE THIS
model_path = "/content/drive/MyDrive/tomato_model.pth"  # Your trained model

# Class names (update based on your folder structure)
class_names = {
    0: "Healthy Leaf",
    1: "Healthy Fruit",
    2: "Infected Leaf",
    3: "Infected Fruit"
}

# Data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
num_classes = len(dataset.classes)

# @title Model Loading (Replace with Your Model)
class TomatoModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)

model = TomatoModel(num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# @title SHAP Implementation
def shap_explanation(input_image, model, class_idx):
    """Generate SHAP values for a single image"""
    # Define prediction function
    def predict(img):
        img = torch.tensor(img).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            output = model(img)
        return output.cpu().numpy()

    # Prepare image
    input_np = input_image.cpu().numpy().transpose(1, 2, 0)
    
    # Create SHAP explainer
    explainer = shap.Explainer(
        predict, 
        masker=shap.maskers.Image("inpaint_telea", input_np.shape),
        output_names=list(class_names.values())
    
    # Compute SHAP values
    shap_values = explainer(input_np[np.newaxis, ...])
    
    # Process for visualization
    shap_heatmap = np.abs(shap_values.values[0, ..., class_idx])
    shap_heatmap = (shap_heatmap - shap_heatmap.min()) / (shap_heatmap.max() - shap_heatmap.min())
    
    return shap_heatmap, shap_values

# @title LIME Implementation
def lime_explanation(input_image, model, class_idx, num_samples=1000):
    """Generate LIME explanation for a single image"""
    # Prediction function
    def predict(images):
        tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            outputs = model(tensor)
        return outputs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    
    # Convert tensor to numpy
    input_np = input_image.cpu().numpy().transpose(1, 2, 0)
    
    # Get explanation
    explanation = explainer.explain_instance(
        input_np,
        predict,
        top_labels=num_classes,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Get LIME mask
    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    return mask, explanation

# @title Visualization Functions
def plot_shap(shap_values, input_image, class_name):
    """Visualize SHAP explanations"""
    input_np = input_image.cpu().numpy().transpose(1, 2, 0)
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
    
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(input_np)
    plt.title(f"Original: {class_name}")
    plt.axis('off')
    
    # SHAP Heatmap
    plt.subplot(1, 2, 2)
    shap.image_plot(
        [shap_values.values[0, ..., :]], 
        [input_np],
        show=False
    )
    plt.title("SHAP Explanation")
    plt.tight_layout()
    plt.show()

def plot_lime(mask, input_image, class_name):
    """Visualize LIME explanations"""
    input_np = input_image.cpu().numpy().transpose(1, 2, 0)
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
    
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(input_np)
    plt.title(f"Original: {class_name}")
    plt.axis('off')
    
    # LIME Explanation
    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(input_np, mask))
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# @title Generate Explanations for Sample Images
def analyze_samples(num_samples=2):
    """Analyze random samples from dataset"""
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in sample_indices:
        image, label = dataset[idx]
        input_image = image.unsqueeze(0).to(device)
        class_name = class_names[label]
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_image)
            pred_class = torch.argmax(output).item()
        
        print(f"\nAnalyzing: {class_name} (Predicted: {class_names[pred_class]})")
        
        # SHAP
        print("Generating SHAP explanation...")
        shap_heatmap, shap_values = shap_explanation(input_image[0], model, label)
        plot_shap(shap_values, input_image[0], class_name)
        
        # LIME
        print("Generating LIME explanation...")
        lime_mask, lime_exp = lime_explanation(input_image[0], model, label)
        plot_lime(lime_mask, input_image[0], class_name)

# @title Run the Analysis
analyze_samples(num_samples=2)  # Change number of samples as needed
