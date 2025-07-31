# @title SHAP Analysis for Tomato Disease Classification
!pip install shap --quiet
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from torchvision import transforms
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @title 1. Load Your Model and Dataset
# Replace with your actual model loading code
model = ... # Your trained PyTorch model
model.eval()

# Class names (update according to your dataset)
class_names = ["Healthy Leaf", "Healthy Fruit", "Infected Leaf", "Infected Fruit"]

# @title 2. SHAP Explanation Function
def shap_analyze(image_path, model, class_idx=None, n_samples=50):
    """
    Generate SHAP explanations for a single image
    Args:
        image_path: Path to input image
        model: Your trained PyTorch model
        class_idx: Specific class to explain (None for auto-detect)
        n_samples: Number of SHAP samples (more=accurate but slower)
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Prediction function for SHAP
    def predict(img_np):
        # Convert numpy array to tensor
        img_t = torch.tensor(img_np, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            outputs = model(img_t)
        return outputs.cpu().numpy()

    # Convert image to numpy for SHAP
    img_np = img_tensor.cpu().numpy()[0].transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean  # Reverse normalization
    img_np = np.clip(img_np, 0, 1)

    # If no class specified, use model's prediction
    if class_idx is None:
        with torch.no_grad():
            preds = model(img_tensor)
            class_idx = torch.argmax(preds).item()
    
    print(f"Explaining for class: {class_names[class_idx]}")

    # Create SHAP explainer
    explainer = shap.Explainer(
        predict,
        masker=shap.maskers.Image("inpaint_telea", img_np.shape),
        output_names=class_names
    )

    # Compute SHAP values
    shap_values = explainer(
        img_np[np.newaxis, ...], 
        max_evals=n_samples * 100,  # Increased evaluations for better accuracy
        outputs=[class_idx]  # Focus on the target class
    )

    # @title 3. Visualization
    plt.figure(figsize=(16, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title(f"Original: {class_names[class_idx]}")
    plt.axis('off')

    # SHAP heatmap
    plt.subplot(1, 3, 2)
    shap.image_plot(
        [shap_values.values[0, ..., class_idx:class_idx+1]], 
        [img_np],
        show=False
    )
    plt.title("SHAP Values (Absolute)")
    plt.axis('off')

    # Overlay visualization
    plt.subplot(1, 3, 3)
    abs_shap = np.abs(shap_values.values[0, ..., class_idx])
    overlay = img_np * 0.7 + abs_shap[..., np.newaxis] * 0.3
    plt.imshow(overlay)
    plt.title("Image + SHAP Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    return shap_values

# @title 4. Run SHAP Analysis (Example)
# Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Example usage
image_path = "/content/drive/MyDrive/your_dataset/Infected_Leaf/image123.jpg"  # CHANGE THIS
shap_values = shap_analyze(image_path, model, n_samples=100)  # Increase samples for better accuracy
