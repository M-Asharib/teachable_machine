import os
import io
import time
import base64
import pickle
import logging
import hashlib
from typing import Dict, Tuple, List
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ─── Logger ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("teachable_machine.ml")

# ─── Classifier Registry ────────────────────────────────────────────────────────
# Supported classifiers selectable from the UI and /train endpoint
CLASSIFIER_REGISTRY = {
    "LogisticRegression": lambda c, penalty: LogisticRegression(
        C=c, penalty=penalty,
        solver="liblinear" if penalty == "l1" else "lbfgs",
        max_iter=1000, random_state=42
    ),
    "SVM": lambda c, penalty: SVC(
        C=c, kernel="rbf", probability=True, random_state=42
    ),
    "RandomForest": lambda c, penalty: RandomForestClassifier(
        n_estimators=150, max_depth=None, random_state=42, n_jobs=-1
    ),
    "KNN": lambda c, penalty: KNeighborsClassifier(
        n_neighbors=5, metric="cosine"
    ),
}

# Set device to CPU strictly for lightweight and portable execution
device = torch.device("cpu")

# Global singleton/cached instance of feature extractors to save memory and startup time
_BACKBONES = {}

def get_backbone(name: str):
    global _BACKBONES
    if name not in _BACKBONES:
        logger.info(f"Loading backbone: {name}")
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
        logger.info(f"Backbone '{name}' loaded and cached.")
    return _BACKBONES[name]

# Default backbone for initial startup/backwards compatibility
FEATURE_EXTRACTOR = get_backbone("MobileNetV3")

# Mandatory preprocessing pipeline — used for BOTH training and inference (uniformity constraint)
# Resizes to 224x224 and normalizes according to default ImageNet channel parameters
preprocess_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Augmentation pipeline — applied only during training, not inference
# Adds variety to small datasets: flips, color jitter, small rotations
train_augmentation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10),
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

# ─── Feature Vector Cache (disk-based, keyed by MD5 hash) ───────────────────────
def _get_feature_cache_path(img_path: str, backbone_name: str) -> str:
    """Returns the cache .npy file path for a given image and backbone."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    with open(img_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return os.path.join(cache_dir, f"{file_hash}_{backbone_name}.npy")

def get_cached_features(img_path: str, backbone_name: str) -> np.ndarray:
    """
    Returns feature vector for an image, loading from disk cache if available.
    On a cache miss, extracts features and saves to cache for next time.
    This makes retraining near-instant on already-processed images.
    """
    cache_path = _get_feature_cache_path(img_path, backbone_name)
    if os.path.exists(cache_path):
        logger.debug(f"Cache HIT: {os.path.basename(img_path)}")
        return np.load(cache_path)
    # Cache MISS — extract and store
    logger.debug(f"Cache MISS: {os.path.basename(img_path)} — extracting features.")
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    tensor = preprocess_image(img_bytes)
    features = extract_features(tensor, backbone_name)
    np.save(cache_path, features)
    return features

def train_model(dataset_dir: str, model_path: str, backbone_name: str = "MobileNetV3",
                classifier_type: str = "LogisticRegression",
                c_value: float = 1.0, penalty: str = "l2") -> Dict:
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

    logger.info(f"Scanning dataset: {len(classes)} classes found — {classes}")

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
                # ── Use disk cache for original features — near-instant on repeat training
                features_orig = get_cached_features(img_path, backbone_name)
                X.append(features_orig)
                y.append(idx)

                # Augmented copy — always re-extracted (random transforms change each run)
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor_aug = train_augmentation(image).unsqueeze(0).to(device)
                features_aug = extract_features(tensor_aug, backbone_name=backbone_name)
                X.append(features_aug)
                y.append(idx)

            except Exception as e:
                logger.warning(f"Skipping corrupted image {img_file} in class {class_name}: {e}")
                continue

    if len(X) == 0:
        raise ValueError("No valid images could be processed for training.")

    logger.info(f"Feature extraction complete: {len(X)} vectors of dim {len(X[0])}")

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
    
    # 3. Build and fit classifier from registry
    if classifier_type not in CLASSIFIER_REGISTRY:
        logger.warning(f"Unknown classifier_type='{classifier_type}', falling back to LogisticRegression.")
        classifier_type = "LogisticRegression"

    logger.info(f"Fitting classifier='{classifier_type}' | C={c_value} | penalty={penalty} | train_samples={len(X_train)}")
    classifier = CLASSIFIER_REGISTRY[classifier_type](c_value, penalty)
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
        "classifier_type": classifier_type,
        "label_map": label_map,
        "backbone_name": backbone_name,
        "features_dim": 512 if backbone_name == "ResNet18" else 576
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    logger.info(f"Model saved to: {model_path}")

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
        logger.info(f"Loading model from disk: {model_path}")
        with open(model_path, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
        logger.info("Model loaded and cached in memory.")
        
    classifier = _MODEL_CACHE["classifier"]
    label_map = _MODEL_CACHE["label_map"]
    backbone_name = _MODEL_CACHE.get("backbone_name", "MobileNetV3")

    # Fetch corresponding backbone model
    backbone_model = get_backbone(backbone_name)

    # 2. Preprocess image tensor (enable grad for Grad-CAM)
    tensor = preprocess_image(image_bytes)
    tensor.requires_grad_()

    # 3. Run feature extraction with gradient hooks for Grad-CAM
    backbone_model.zero_grad()

    # Register a hook to capture the last convolutional feature map
    _gradcam_activations = {}
    _gradcam_gradients = {}

    def _save_activation(module, inp, out):
        _gradcam_activations["value"] = out.detach()

    def _save_gradient(module, grad_in, grad_out):
        _gradcam_gradients["value"] = grad_out[0].detach()

    # Hook into the last conv layer of the backbone
    # For MobileNetV3Small: features[-1] is the last conv block
    # For ResNet18: layer4[-1] is the last residual block
    if backbone_name == "ResNet18":
        target_layer = backbone_model.layer4[-1]
    else:  # MobileNetV3
        target_layer = backbone_model.features[-1]

    hook_fwd = target_layer.register_forward_hook(_save_activation)
    hook_bwd = target_layer.register_full_backward_hook(_save_gradient)

    features = backbone_model(tensor)
    features_np = features.squeeze(0).detach().cpu().numpy().reshape(1, -1)

    # 4. Predict class index and probabilities
    pred_idx = int(classifier.predict(features_np)[0])
    probabilities = classifier.predict_proba(features_np)[0]

    # 5. Compute Grad-CAM — backprop through the predicted class score
    # Use LogReg weights as the class score proxy for non-differentiable classifiers
    try:
        class_weights = torch.tensor(
            classifier.coef_[pred_idx], dtype=torch.float32
        ).to(device)
        score = (features.squeeze(0) * class_weights).sum()
        score.backward()

        # Global Average Pool the gradients over spatial dims
        gradients = _gradcam_gradients["value"]   # [1, C, H, W]
        activations = _gradcam_activations["value"] # [1, C, H, W]
        weights = gradients.mean(dim=[2, 3], keepdim=True)   # [1, C, 1, 1]
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))  # [1, 1, H, W]
        cam = cam.squeeze().cpu().numpy()

        # Normalize CAM to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        use_gradcam = True
    except Exception as e:
        logger.warning(f"Grad-CAM failed, falling back to gradient saliency: {e}")
        use_gradcam = False
        # Fallback — basic gradient saliency
        loss = features.sum()
        loss.backward()
        saliency_raw, _ = torch.max(tensor.grad.data.abs(), dim=1)
        cam = saliency_raw.squeeze(0).cpu().numpy()
        s_min, s_max = cam.min(), cam.max()
        cam = (cam - s_min) / (s_max - s_min) if s_max - s_min > 1e-8 else np.zeros_like(cam)

    hook_fwd.remove()
    hook_bwd.remove()

    attention_map = cam  # normalized 2D heatmap

    # 6. Build the visual bounding box and attention maps using PIL and NumPy
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize attention map (Grad-CAM or fallback) to original image dimensions
    attn_img = Image.fromarray((attention_map * 255).astype(np.uint8), mode='L')
    attn_img = attn_img.resize(original_image.size, resample=Image.BICUBIC)

    # Map normalized grayscale to jet colormap (blue-cyan-green-yellow-red)
    s_np = np.array(attn_img) / 255.0
    r = np.clip(1.5 - np.abs(4 * (s_np - 0.75)), 0, 1)
    g = np.clip(1.5 - np.abs(4 * (s_np - 0.50)), 0, 1)
    b = np.clip(1.5 - np.abs(4 * (s_np - 0.25)), 0, 1)
    heatmap_np = np.stack([r, g, b], axis=-1)
    heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8))

    # Overlay attention map transparently on top of original image
    saliency_blend = Image.blend(original_image, heatmap_img, alpha=0.55)

    # 7. Create Bounding Box surrounding highly active regions
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

    # Confidence threshold — if no class is above 60%, label result as "Unknown"
    CONFIDENCE_THRESHOLD = 60.0
    top_label = label_map[pred_idx]
    top_confidence = float(round(probabilities[pred_idx] * 100, 2))
    if top_confidence < CONFIDENCE_THRESHOLD:
        top_label = "Unknown (Low Confidence)"
        logger.info(f"Prediction below confidence threshold: best={label_map[pred_idx]} @ {top_confidence}%")
    else:
        logger.info(f"Prediction: {top_label} @ {top_confidence}%")

    inference_time_ms = float(round((time.time() - start_time) * 1000, 1))

    return {
        "predicted_class": top_label,
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
