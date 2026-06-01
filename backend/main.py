import os
import re
import uuid
import shutil
import pickle
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Robust import pattern to support running from root or from backend folder
try:
    from backend import ml_engine
except ImportError:
    import ml_engine

# ─── Structured Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("teachable_machine.api")

# ─── Thread Pool for Non-Blocking Training ───────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(
    title="Teachable Machine Clone API",
    description="Decoupled backend for custom image classification using Transfer Learning.",
    version="1.0.0"
)

# ─── CORS — restricted to known frontend origins ────────────────────────────
# In production replace with your deployed domain e.g. "https://yourdomain.com"
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://frontend:8501",   # Docker service name
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Upload Size Limit Middleware (max 20 MB per request) ─────────────────────
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Reject any POST request whose Content-Length exceeds 20 MB."""
    if request.method == "POST":
        content_length = int(request.headers.get("content-length", 0))
        if content_length > MAX_UPLOAD_BYTES:
            logger.warning(f"Rejected oversized upload: {content_length} bytes from {request.client.host}")
            return JSONResponse(
                status_code=413,
                content={"detail": f"Upload too large. Maximum allowed size is 20 MB."}
            )
    return await call_next(request)

# Setup dataset and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
os.makedirs(DATASET_DIR, exist_ok=True)


# ─── Magic Bytes for MIME-type validation ────────────────────────────────────
# Checks the real file header, not just the filename extension.
IMAGE_MAGIC_BYTES: dict[bytes, str] = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG": ".png",
    b"RIFF": ".webp",   # WEBP starts with RIFF
    b"BM": ".bmp",
}

def is_valid_image_bytes(header: bytes) -> bool:
    """Returns True if the file header matches a known image magic-byte signature."""
    for magic in IMAGE_MAGIC_BYTES:
        if header[:len(magic)] == magic:
            return True
    return False


def sanitize_class_name(name: str) -> str:
    """Sanitizes the class name to be safe for a directory name."""
    cleaned = name.strip()
    cleaned = re.sub(r'[^a-zA-Z0-9_\-]', '_', cleaned)
    if not cleaned:
        cleaned = "unnamed_class"
    return cleaned

@app.get("/")
def read_root():
    """Simple API status health check."""
    return {
        "status": "Teachable Machine API is running",
        "dataset_directory": DATASET_DIR,
        "model_trained": os.path.exists(MODEL_PATH),
        "active_classes": [
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != ".gitkeep"
        ]
    }

@app.post("/upload-sample")
async def upload_sample(
    class_name: str = Form(..., description="The custom label/class name for the images"),
    files: List[UploadFile] = File(..., description="A list of sample images to upload")
):
    """
    Ingest custom training images for a specific class folder.
    Uses Python's os module to manage directories and uuid to prevent collisions.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    safe_class_name = sanitize_class_name(class_name)
    class_dir = os.path.join(DATASET_DIR, safe_class_name)
    logger.info(f"Upload request for class='{safe_class_name}' | file_count={len(files)}")

    try:
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, exist_ok=True)
            logger.info(f"Created new class directory: {class_dir}")
    except Exception as e:
        logger.error(f"Failed to create class folder '{class_dir}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create class folder: {str(e)}"
        )

    saved_count = 0
    for file in files:
        filename = file.filename or ""
        ext = os.path.splitext(filename)[1].lower()

        # Read file bytes for MIME validation
        file_bytes = await file.read()
        await file.seek(0)

        # ── Security: validate by magic bytes, not just extension ──
        if not is_valid_image_bytes(file_bytes[:16]):
            logger.warning(f"Rejected file '{filename}': invalid magic bytes (not a real image).")
            await file.close()
            continue

        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            logger.warning(f"Rejected file '{filename}': unsupported extension '{ext}'.")
            await file.close()
            continue

        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = os.path.join(class_dir, unique_name)

        try:
            with open(target_path, "wb") as buffer:
                buffer.write(file_bytes)
            saved_count += 1
            logger.debug(f"Saved image: {target_path}")
        except Exception as e:
            logger.error(f"Failed to write '{filename}' to disk: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write image to disk: {str(e)}"
            )
        finally:
            await file.close()

    if saved_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid image files were successfully uploaded."
        )

    return {
        "message": f"Successfully uploaded {saved_count} image(s) for class '{safe_class_name}'.",
        "class_name": safe_class_name,
        "saved_count": saved_count
    }

@app.post("/train")
async def train(
    backbone_name: str = Form("MobileNetV3", description="The visual backbone to use (MobileNetV3 or ResNet18)"),
    classifier_type: str = Form("LogisticRegression", description="Classifier: LogisticRegression | SVM | RandomForest | KNN"),
    c_value: float = Form(1.0, description="Inverse of regularization strength (positive float)"),
    penalty: str = Form("l2", description="Regularization penalty (l1 or l2)")
):
    """
    Triggers the transfer learning engine asynchronously using a thread pool
    so the server stays responsive to other requests during long training runs.
    """
    # 1. Validation Checks (Graceful Error Controls)
    if not os.path.exists(DATASET_DIR):
        raise HTTPException(status_code=400, detail="No dataset found. Please upload images first.")

    classes = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != ".gitkeep"
    ]

    if len(classes) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 distinct classes with images are required to train a model."
        )

    # Verify each class has at least 1 image
    for cls in classes:
        class_folder = os.path.join(DATASET_DIR, cls)
        images = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ]
        if not images:
            raise HTTPException(
                status_code=400,
                detail=f"Class '{cls}' is empty. All classes must have at least 1 image to train."
            )

    logger.info(f"Training started | backbone={backbone_name} | classes={classes} | C={c_value} | penalty={penalty}")

    # 2. Run training in a thread pool — keeps server non-blocking
    try:
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            _executor,
            lambda: ml_engine.train_model(
                DATASET_DIR, MODEL_PATH,
                backbone_name=backbone_name,
                classifier_type=classifier_type,
                c_value=c_value,
                penalty=penalty
            )
        )
        logger.info(f"Training completed | samples={summary.get('samples_trained')} | classes={summary.get('classes')}")
        return {
            "status": "success",
            "message": "Model trained successfully!",
            "details": summary
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(..., description="The image frame to run inference on")):
    """
    Run inference on a single test image.
    Uses the PyTorch MobileNetV3 + Logistic Regression pipeline.
    """
    # 1. Check if model file exists on disk
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=400,
            detail="No trained model found. Please train the model first."
        )

    # 2. Load file bytes
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image bytes: {str(e)}")
    finally:
        await file.close()

    # 3. Predict using ml_engine
    try:
        prediction = ml_engine.predict_image(image_bytes, MODEL_PATH)
        return prediction
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.delete("/reset")
def reset_dataset():
    """
    DELETE /reset: Clears the entire dataset folder and removes model.pkl.
    Resets the training workspace completely.
    """
    ml_engine._MODEL_CACHE = None
    
    # Remove model weight file if it exists
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete model weights: {str(e)}")

    # Clean up all folders inside dataset/ except .gitkeep
    deleted_folders = []
    if os.path.exists(DATASET_DIR):
        for item in os.listdir(DATASET_DIR):
            item_path = os.path.join(DATASET_DIR, item)
            if os.path.isdir(item_path) and item != ".gitkeep":
                try:
                    shutil.rmtree(item_path)
                    deleted_folders.append(item)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to delete class folder '{item}': {str(e)}")

    return {
        "status": "success",
        "message": "Dataset and trained model have been successfully reset.",
        "deleted_classes": deleted_folders
    }

@app.get("/features-pca")
def features_pca(backbone_name: str = "MobileNetV3"):
    """
    Exposes 2D PCA representation of training data features.
    """
    try:
        results = ml_engine.get_dataset_pca(DATASET_DIR, backbone_name)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PCA clustering calculation failed: {str(e)}")

@app.get("/dataset-info")
def get_dataset_info():
    """
    Get active classes and sample counts from the dataset directory.
    """
    if not os.path.exists(DATASET_DIR):
        return {"classes": {}}
    
    classes_info = {}
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path) and item != ".gitkeep":
            images = [
                f for f in os.listdir(item_path)
                if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
            ]
            classes_info[item] = len(images)
    return {"classes": classes_info}

@app.get("/model-info")
def get_model_info():
    """
    Load and return metadata of the trained model from model.pkl.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Please train the model first."
        )
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        
        clf = model_data.get("classifier")
        return {
            "backbone_name": model_data.get("backbone_name", "MobileNetV3"),
            "features_dim": model_data.get("features_dim", 576),
            "label_map": model_data.get("label_map", {}),
            "classifier_type": type(clf).__name__ if clf else None,
            "penalty": getattr(clf, "penalty", "l2") if clf else None,
            "c_value": getattr(clf, "C", 1.0) if clf else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read model metadata: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NEW ENDPOINTS — Week 1 & 2 Next-Level Features
# ─────────────────────────────────────────────────────────────────────────────

import json
import base64
import hashlib
from datetime import datetime

LOGS_DIR = os.path.join(BASE_DIR, "logs")
PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")
os.makedirs(LOGS_DIR, exist_ok=True)

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@app.get("/export-model")
def export_model():
    """
    Download the trained model.pkl file directly.
    Returns the binary file as an attachment.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="No trained model found. Please train the model first."
        )
    logger.info("Model export requested.")
    return FileResponse(
        MODEL_PATH,
        filename="teachable_machine_model.pkl",
        media_type="application/octet-stream"
    )


@app.get("/class-images/{class_name}")
def get_class_images(class_name: str):
    """
    Returns a list of base64-encoded thumbnail images for the given class.
    Used by the frontend image gallery.
    """
    safe_name = sanitize_class_name(class_name)
    class_dir = os.path.join(DATASET_DIR, safe_name)

    if not os.path.exists(class_dir):
        raise HTTPException(status_code=404, detail=f"Class '{safe_name}' not found.")

    images = []
    for fname in os.listdir(class_dir):
        if os.path.splitext(fname)[1].lower() not in VALID_IMAGE_EXTS:
            continue
        fpath = os.path.join(class_dir, fname)
        try:
            with open(fpath, "rb") as f:
                raw = f.read()
            # Resize to thumbnail using PIL to keep response small
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img.thumbnail((120, 120))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = base64.b64encode(buf.getvalue()).decode()
            images.append({"filename": fname, "thumbnail": b64})
        except Exception as e:
            logger.warning(f"Could not read image {fname}: {e}")
            continue

    logger.info(f"Gallery requested for class='{safe_name}': {len(images)} images returned.")
    return {"class_name": safe_name, "count": len(images), "images": images}


@app.delete("/class-image/{class_name}/{filename}")
def delete_class_image(class_name: str, filename: str):
    """
    Delete a single training image from a class folder.
    Invalidates the in-memory model cache so next prediction reloads.
    """
    safe_name = sanitize_class_name(class_name)
    # Prevent path traversal attacks
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    target = os.path.join(DATASET_DIR, safe_name, filename)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in class '{safe_name}'.")

    try:
        os.remove(target)
        ml_engine._MODEL_CACHE = None  # Invalidate cache — dataset changed
        logger.info(f"Deleted image: {target}")
        return {"status": "success", "message": f"Image '{filename}' deleted from class '{safe_name}'."}
    except Exception as e:
        logger.error(f"Failed to delete image '{target}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")


@app.post("/log-prediction")
async def log_prediction(request: Request):
    """
    Append a prediction result to the JSONL predictions log file.
    Called by the frontend after every inference result.
    """
    try:
        data = await request.json()
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicted_class": data.get("predicted_class"),
            "confidence": data.get("confidence"),
            "probabilities": data.get("probabilities", {}),
            "backbone": data.get("backbone_used"),
        }
        with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics")
def get_analytics():
    """
    Aggregate the predictions log and return summary statistics.
    Returns: total predictions, class distribution, average confidence per class.
    """
    if not os.path.exists(PREDICTIONS_LOG):
        return {"total_predictions": 0, "class_distribution": {}, "avg_confidence": {}}

    entries = []
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue

    if not entries:
        return {"total_predictions": 0, "class_distribution": {}, "avg_confidence": {}}

    class_counts: dict = {}
    class_confidences: dict = {}
    for entry in entries:
        cls = entry.get("predicted_class", "Unknown")
        conf = entry.get("confidence") or 0.0
        class_counts[cls] = class_counts.get(cls, 0) + 1
        class_confidences.setdefault(cls, []).append(conf)

    avg_confidence = {
        cls: round(sum(confs) / len(confs), 2)
        for cls, confs in class_confidences.items()
    }

    logger.info(f"Analytics served: {len(entries)} total predictions.")
    return {
        "total_predictions": len(entries),
        "class_distribution": class_counts,
        "avg_confidence": avg_confidence,
        "recent_predictions": entries[-10:]  # last 10 entries
    }


@app.delete("/clear-cache")
def clear_feature_cache():
    """
    Wipe all cached feature vectors from the feature_cache directory.
    Use when you want to force full re-extraction on next training run.
    """
    cache_dir = os.path.join(BASE_DIR, "feature_cache")
    if not os.path.exists(cache_dir):
        return {"status": "success", "message": "No cache directory found (nothing to clear)."}
    deleted = 0
    for fname in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            deleted += 1
    logger.info(f"Feature cache cleared: {deleted} files deleted.")
    return {"status": "success", "deleted_files": deleted}
