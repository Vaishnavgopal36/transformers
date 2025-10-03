
import torch
import torch.nn as nn
import timm
import argparse
from PIL import Image
from torchvision import transforms

# --- Configuration ---
MODEL_PATH = 'vision_encoder/models/vit_cifar10_finetuned.pth'
# Image size and normalization values must match the training notebook
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
# CIFAR-10 class names
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_inference_transforms():
    """Returns the transformations for an inference image."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

def predict(image_path: str, model: nn.Module, device: torch.device):
    """Loads an image, preprocesses it, and returns the predicted class name."""
    try:
        # 1. Load and transform the image
        img = Image.open(image_path).convert("RGB")
        transform = get_inference_transforms()
        # Add a batch dimension (B, C, H, W) and send to device
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 2. Get prediction
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
        # 3. Get the top prediction
        top_prob, top_idx = torch.max(probabilities, 1)
        pred_class_name = CIFAR10_CLASSES[top_idx.item()]
        confidence = top_prob.item()

        return pred_class_name, confidence

    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}", None
    except Exception as e:
        return f"An error occurred: {e}", None

if __name__ == "__main__":
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Classify an image using a fine-tuned ViT model.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()
    
    # --- Setup Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Re-create the model structure (same as in the notebook)
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(CIFAR10_CLASSES))

    # 2. Load the fine-tuned weights from your saved .pth file
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Make sure you've run the training notebook first.")
        exit()
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()

    # --- Get Prediction ---
    predicted_class, confidence_score = predict(args.image, model, device)

    if confidence_score is not None:
        print(f"\n--> Prediction: {predicted_class}")
        print(f"    Confidence: {confidence_score:.2%}")
    else:
        # Print the error message from the predict function
        print(predicted_class)