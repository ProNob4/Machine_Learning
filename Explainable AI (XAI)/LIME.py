# @title LIME Analysis for Tomato Disease Classification
!pip install lime --quiet
import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @title 1. Load Your Model and Dataset
# Replace with your actual model (example shown)
class TomatoDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.model.fc = torch.nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return self.model(x)

model = TomatoDiseaseModel().to(device)
model.load_state_dict(torch.load("your_model_path.pth"))  # CHANGE THIS
model.eval()

# Class names (update according to your dataset)
class_names = ["Healthy Leaf", "Healthy Fruit", "Infected Leaf", "Infected Fruit"]

# @title 2. LIME Explanation Function
def lime_explain(image_path, model, top_labels=2, num_samples=1000):
    """
    Generate LIME explanations for a single image
    Args:
        image_path: Path to input image
        model: Your trained PyTorch model
        top_labels: Number of top classes to explain
        num_samples: Number of LIME samples (more=accurate but slower)
    """
    # Preprocess function for LIME
    def batch_predict(images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = torch.stack([transform(Image.fromarray(img)) for img in images]).to(device)
        with torch.no_grad():
            logits = model(tensor)
        return logits.cpu().numpy()

    # Load and prepare image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img.resize((256, 256))) / 255.0
    
    # Create explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get explanation
    explanation = explainer.explain_instance(
        img_np,
        batch_predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
        batch_size=32  # Faster processing
    )

    # @title 3. Visualization
    plt.figure(figsize=(15, 5))
    
    # Original image with predictions
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    pred_probs = batch_predict(np.expand_dims(img_np, 0))[0]
    pred_class = np.argmax(pred_probs)
    plt.title(f"Original\nPredicted: {class_names[pred_class]} ({pred_probs[pred_class]:.2f})")
    plt.axis('off')

    # LIME explanation for top class
    plt.subplot(1, 3, 2)
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"LIME Explanation\n{class_names[pred_class]}")
    plt.axis('off')

    # LIME heatmap
    plt.subplot(1, 3, 3)
    seg = explanation.segments
    heatmap = np.zeros(seg.shape)
    for i in range(seg.max() + 1):
        heatmap[seg == i] = explanation.local_exp[pred_class].get(i, 0)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("LIME Heatmap (Red=Positive, Blue=Negative)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    return explanation

# @title 4. Run LIME Analysis (Example)
from google.colab import drive
drive.mount('/content/drive')

# Example usage
image_path = "/content/drive/MyDrive/your_dataset/Infected_Leaf/image123.jpg"  # CHANGE THIS
lime_explanation = lime_explain(image_path, model, num_samples=1000)
