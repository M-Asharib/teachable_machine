import os
import io
import time
import base64
import pickle
from typing import Dict, Tuple, List
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.linear_model import LogisticRegression

# Set device to CPU strictly for lightweight and portable execution
device = torch.device("cpu")

# Global singleton/cached instance of feature extractors to save memory and startup time
_BACKBONES = {}

def get_backbone(name: str):
    global _BACKBONES
    if name not in _BACKBONES:
        if name == "ResNet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = torch.nn.Identity()
        else:  # MobileNetV3
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = torch.nn.Identity()
        
        model.to(device)
        model.eval()
        _BACKBONES[name] = model
    return _BACKBONES[name]

# Default backbone for initial startup/backwards compatibility
FEATURE_EXTRACTOR = get_backbone("MobileNetV3")

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

def extract_features(image_tensor: torch.Tensor, backbone_name: str = "MobileNetV3") -> np.ndarray:
    """
    Passes a preprocessed image tensor through the specified backbone
    and returns a 1D numpy feature vector.
    """
    backbone = get_backbone(backbone_name)
    with torch.no_grad():
        embedding = backbone(image_tensor)
        return embedding.squeeze().cpu().numpy()

# In-memory model cache to avoid disk reads on every prediction call
_MODEL_CACHE = None

def train_model(dataset_dir: str, model_path: str, backbone_name: str = "MobileNetV3", c_value: float = 1.0, penalty: str = "l2") -> Dict:
    """
    Scans the dataset directory, extracts features for all samples using the chosen backbone,
    splits data into train/val subsets, trains a Scikit-Learn LogisticRegression classifier
    with the selected penalty and regularizer strength, computes evaluation metrics, and saves weights.
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
                
                # Preprocess and extract features
                tensor = preprocess_image(img_bytes)
                features = extract_features(tensor, backbone_name=backbone_name)
                
                X.append(features)
                y.append(idx)
            except Exception as e:
                print(f"Skipping corrupted image {img_file} in class {class_name}: {str(e)}")
                continue

    if len(X) == 0:
        raise ValueError("No valid images could be processed for training.")
        
    X = np.array(X)
    y = np.array(y)

    # 2. Evaluate eligibility for train-validation split (need at least 2 samples per class and >= 6 total samples)
    unique_classes, counts = np.unique(y, return_counts=True)
    can_split = len(y) >= 6 and all(cnt >= 2 for cnt in counts)
    
    X_train, y_train = X, y
    validation_metrics = None
    
    if can_split:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
    
    # 3. Fit Scikit-Learn Logistic Regression Classifier
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    classifier = LogisticRegression(C=c_value, penalty=penalty, solver=solver, max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # Calculate validation metrics if validation set exists
    if can_split:
        y_pred = classifier.predict(X_val)
        accuracy = float(round(np.mean(y_val == y_pred) * 100, 2))
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        
        validation_metrics = {
            "split_executed": True,
            "accuracy": accuracy,
            "precision": float(round(precision * 100, 2)),
            "recall": float(round(recall * 100, 2)),
            "f1_score": float(round(f1 * 100, 2)),
            "confusion_matrix": {
                "labels": classes,
                "matrix": cm.tolist()
            }
        }
    else:
        validation_metrics = {
            "split_executed": False,
            "warning": "Low sample count: all images used for training (no validation split)."
        }

    # 4. Save serialized model bundle
    model_bundle = {
        "classifier": classifier,
        "label_map": label_map,
        "backbone_name": backbone_name,
        "features_dim": 512 if backbone_name == "ResNet18" else 576
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    class_distribution = {label_map[i]: int(np.sum(y == i)) for i in range(len(classes))}
    
    return {
        "classes": classes,
        "samples_trained": len(y_train),
        "samples_validated": len(y) - len(y_train) if can_split else 0,
        "class_distribution": class_distribution,
        "validation_metrics": validation_metrics
    }

def predict_image(image_bytes: bytes, model_path: str) -> Dict:
    """
    Loads saved model.pkl, preprocesses incoming testing image,
    runs inference using the saved backbone model, generates saliency visual
    attention maps and foreground bounding box overlays, and returns them.
    """
    global _MODEL_CACHE
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train the model first.")

    start_time = time.time()

    # 1. Load model bundle from cache if available
    if _MODEL_CACHE is None:
        with open(model_path, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
        
    classifier = _MODEL_CACHE["classifier"]
    label_map = _MODEL_CACHE["label_map"]
    backbone_name = _MODEL_CACHE.get("backbone_name", "MobileNetV3")

    # Fetch corresponding backbone model
    backbone_model = get_backbone(backbone_name)

    # 2. Preprocess image tensor (enable grad for saliency extraction)
    tensor = preprocess_image(image_bytes)
    tensor.requires_grad_()

    # 3. Extract features and track gradient flow
    backbone_model.zero_grad()
    features = backbone_model(tensor)
    
    # Detach features for classifier prediction
    features_np = features.squeeze(0).detach().cpu().numpy().reshape(1, -1)

    # 4. Predict class index and probabilities
    pred_idx = int(classifier.predict(features_np)[0])
    probabilities = classifier.predict_proba(features_np)[0]

    # 5. Compute Saliency Attention Map via backpropagation
    loss = features.sum()
    loss.backward()
    
    saliency, _ = torch.max(tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze(0).cpu().numpy()

    # Normalization helper
    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 1e-8:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    # 6. Build the visual bounding box and attention maps using PIL and NumPy
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize saliency map back to match original image dimensions
    saliency_img = Image.fromarray((saliency * 255).astype(np.uint8), mode='L')
    saliency_img = saliency_img.resize(original_image.size, resample=Image.BICUBIC)
    
    # Map normalized grayscale values to custom jet colors (blue-cyan-green-yellow-red)
    s_np = np.array(saliency_img) / 255.0
    r = np.clip(1.5 - np.abs(4 * (s_np - 0.75)), 0, 1)
    g = np.clip(1.5 - np.abs(4 * (s_np - 0.50)), 0, 1)
    b = np.clip(1.5 - np.abs(4 * (s_np - 0.25)), 0, 1)
    heatmap_np = np.stack([r, g, b], axis=-1)
    heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8))
    
    # Overlay attention map transparently on top of original image
    saliency_blend = Image.blend(original_image, heatmap_img, alpha=0.55)

    # 7. Create Bounding Box surrounding highly active saliency clusters
    mask = s_np > 0.35
    y_indices, x_indices = np.where(mask)
    
    bbox_img = original_image.copy()
    draw = ImageDraw.Draw(bbox_img)
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        
        pad = 12
        x_min = max(0, x_min - pad)
        x_max = min(original_image.width, x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(original_image.height, y_max + pad)
        
        # Draw high-contrast cyan scanning rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="#00f2fe", width=3)
        
        # Overlay premium neon-purple bounding brackets on corners
        corner_len = min(24, (x_max - x_min) // 4)
        corner_color = "#7f00ff"
        # Top-Left Bracket
        draw.line([(x_min, y_min), (x_min + corner_len, y_min)], fill=corner_color, width=5)
        draw.line([(x_min, y_min), (x_min, y_min + corner_len)], fill=corner_color, width=5)
        # Top-Right Bracket
        draw.line([(x_max, y_min), (x_max - corner_len, y_min)], fill=corner_color, width=5)
        draw.line([(x_max, y_min), (x_max, y_min + corner_len)], fill=corner_color, width=5)
        # Bottom-Left Bracket
        draw.line([(x_min, y_max), (x_min + corner_len, y_max)], fill=corner_color, width=5)
        draw.line([(x_min, y_max), (x_min, y_max - corner_len)], fill=corner_color, width=5)
        # Bottom-Right Bracket
        draw.line([(x_max, y_max), (x_max - corner_len, y_max)], fill=corner_color, width=5)
        draw.line([(x_max, y_max), (x_max, y_max - corner_len)], fill=corner_color, width=5)

    # 8. Encode final images as Base64 strings for REST API delivery
    def convert_to_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    bbox_b64 = convert_to_b64(bbox_img)
    saliency_b64 = convert_to_b64(saliency_blend)

    # 9. Format response variables
    probs_dict = {
        label_map[idx]: float(round(prob * 100, 2))
        for idx, prob in enumerate(probabilities)
    }

    inference_time_ms = float(round((time.time() - start_time) * 1000, 1))

    return {
        "predicted_class": label_map[pred_idx],
        "probabilities": probs_dict,
        "bounding_box_image": bbox_b64,
        "saliency_image": saliency_b64,
        "inference_time_ms": inference_time_ms,
        "backbone_used": backbone_name
    }

def get_dataset_pca(dataset_dir: str, backbone_name: str = "MobileNetV3") -> List[Dict]:
    """
    Extracts features for all images in dataset_dir using the specified backbone,
    projects them to 2D using PCA, and returns a list of dictionaries with coordinates.
    """
    from sklearn.decomposition import PCA
    
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d != ".gitkeep"
    ])
    
    X = []
    metadata = []
    
    for class_name in classes:
        class_folder = os.path.join(dataset_dir, class_name)
        image_files = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ]
        
        for img_name in image_files:
            img_path = os.path.join(class_folder, img_name)
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                tensor = preprocess_image(img_bytes)
                features = extract_features(tensor, backbone_name)
                X.append(features)
                metadata.append({
                    "class": class_name,
                    "filename": img_name
                })
            except Exception:
                continue
                
    if len(X) < 3:
        return []
        
    X_arr = np.array(X)
    # Fit PCA with 2 components
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(X_arr)
    
    results = []
    for i, meta in enumerate(metadata):
        results.append({
            "class": meta["class"],
            "filename": meta["filename"],
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1])
        })
        
    return results
