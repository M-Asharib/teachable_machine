import os
import re
import uuid
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Robust import pattern to support running from root or from backend folder
try:
    from backend import ml_engine
except ImportError:
    import ml_engine

app = FastAPI(
    title="Teachable Machine Clone API",
    description="Decoupled backend for custom image classification using Transfer Learning.",
    version="1.0.0"
)

# Enable CORS for all origins to facilitate frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup dataset and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
os.makedirs(DATASET_DIR, exist_ok=True)

# Memory Caching: Cache the model in memory to avoid reloading from disk on every single /predict call
MODEL_CACHE = None

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

    try:
        if not os.path.exists(class_dir):
            os.makedirs(class_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create class folder: {str(e)}"
        )

    saved_count = 0
    for file in files:
        filename = file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            continue

        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = os.path.join(class_dir, unique_name)

        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_count += 1
        except Exception as e:
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
def train():
    """
    Triggers the transfer learning engine.
    Scans dataset directories, extracts MobileNetV3 features, fits Scikit-Learn Logistic Regression, and writes model.pkl.
    """
    global MODEL_CACHE
    
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
    for class_name in classes:
        class_folder = os.path.join(DATASET_DIR, class_name)
        images = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        ]
        if not images:
            raise HTTPException(
                status_code=400,
                detail=f"Class '{class_name}' is empty. All classes must have at least 1 image to train."
            )

    # 2. Trigger the training pipeline
    try:
        summary = ml_engine.train_model(DATASET_DIR, MODEL_PATH)
        # Invalidate the cache to force reloading the newly trained model on the next prediction
        MODEL_CACHE = None
        return {
            "status": "success",
            "message": "Model trained successfully!",
            "details": summary
        }
    except Exception as e:
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
    global MODEL_CACHE
    MODEL_CACHE = None
    
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
