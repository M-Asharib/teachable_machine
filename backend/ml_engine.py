import os
import io
import pickle
from typing import Dict, Tuple, List
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.linear_model import LogisticRegression

# Set device to CPU strictly for lightweight and portable execution
device = torch.device("cpu")

# Initialize and cache the pretrained MobileNetV3 Small model as a feature extractor
def get_feature_extractor():
    # Load MobileNetV3 Small with default pretrained weights
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # Freeze all parameters so they are not updated during training
    for param in model.parameters():
        param.requires_grad = False
        
    # Slicing the classifier: Replace the final classifier with Identity
    # The forward pass will return the 576-dimensional pooled & flattened embedding
    model.classifier = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

# Global singleton/cached instance of the feature extractor to save memory and startup time
FEATURE_EXTRACTOR = get_feature_extractor()

# Mandatory preprocessing pipeline (Uniformity constraint)
# Resizes to 224x224 and normalizes according to default ImageNet/channel parameters
preprocess_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Transforms raw image bytes into a preprocessed 4D tensor ready for the backbone.
    Ensures 224x224 resizing and channel normalization.
    """
    try:
        # Load image and force to standard RGB
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Apply transformation
        tensor = preprocess_transforms(image)
        # Add batch dimension: shape (1, 3, 224, 224)
        return tensor.unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def extract_features(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Passes a preprocessed image tensor through the MobileNetV3 backbone
    and returns a 1D numpy feature vector.
    """
    with torch.no_grad():
        embedding = FEATURE_EXTRACTOR(image_tensor)
        # Convert tensor to numpy and flatten it
        return embedding.squeeze().cpu().numpy()

# In-memory model cache to avoid disk reads on every prediction call
_MODEL_CACHE = None

def train_model(dataset_dir: str, model_path: str) -> Dict:
    """
    Scans the dataset directory, extracts features for all samples,
    trains a Scikit-Learn LogisticRegression classifier, and saves weights.
    """
    global _MODEL_CACHE
    _MODEL_CACHE = None
    
    # 1. Scan directory and map folders to classes
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d != ".gitkeep"
    ])
    
    if len(classes) < 2:
        raise ValueError("At least 2 distinct classes with images are required to train a model.")

    X = []
    y = []
    label_map = {} # Maps index -> class_name
    reverse_map = {} # Maps class_name -> index
    
    for idx, class_name in enumerate(classes):
        label_map[idx] = class_name
        reverse_map[class_name] = idx
        
        class_folder = os.path.join(dataset_dir, class_name)
        image_files = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ]
        
        if not image_files:
            raise ValueError(f"Class '{class_name}' folder exists but contains no valid images.")
            
        for img_file in image_files:
            img_path = os.path.join(class_folder, img_file)
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                
                # Preprocess and extract 576-dim feature vector
                tensor = preprocess_image(img_bytes)
                features = extract_features(tensor)
                
                X.append(features)
                y.append(idx)
            except Exception as e:
                # Log or handle corrupted images gracefully
                print(f"Skipping corrupted image {img_file} in class {class_name}: {str(e)}")
                continue

    if len(X) == 0:
        raise ValueError("No valid images could be processed for training.")
        
    X = np.array(X)
    y = np.array(y)

    # 2. Fit Scikit-Learn Logistic Regression Classifier
    # Use L2 penalty, default C=1.0, and liblinear/lbfgs solver (fast for small datasets)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X, y)

    # 3. Save serialized weights bundle
    model_bundle = {
        "classifier": classifier,
        "label_map": label_map,
        "features_dim": 576
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    # Compile training details for summary response
    class_distribution = {label_map[i]: int(np.sum(y == i)) for i in range(len(classes))}
    
    return {
        "classes": classes,
        "samples_trained": len(y),
        "class_distribution": class_distribution
    }

def predict_image(image_bytes: bytes, model_path: str) -> Dict:
    """
    Loads saved model.pkl, preprocesses incoming testing image,
    runs inference, and returns predicted class and percentage probabilities.
    """
    global _MODEL_CACHE
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train the model first.")

    # 1. Load model bundle from cache if available
    if _MODEL_CACHE is None:
        with open(model_path, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
        
    classifier = _MODEL_CACHE["classifier"]
    label_map = _MODEL_CACHE["label_map"]

    # 2. Preprocess identical to training
    tensor = preprocess_image(image_bytes)
    features = extract_features(tensor)
    
    # Reshape for single sample prediction
    features = features.reshape(1, -1)

    # 3. Predict class index and probabilities
    pred_idx = int(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]

    # 4. Format return values in percentage format
    probs_dict = {
        label_map[idx]: float(round(prob * 100, 2))
        for idx, prob in enumerate(probabilities)
    }

    return {
        "predicted_class": label_map[pred_idx],
        "probabilities": probs_dict
    }
