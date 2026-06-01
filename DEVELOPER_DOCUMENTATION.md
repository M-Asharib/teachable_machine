# Teachable Machine Pro: Developer Documentation & Reference Guide

Welcome to the technical reference manual for **Teachable Machine Pro**. This guide is designed for developers, systems engineers, and machine learning practitioners who want to understand, extend, or maintain the codebase of this decoupled, full-stack custom image classification system.

---

## 📖 Table of Contents
1. [System Architecture & Decoupled Strategy](#1-system-architecture--decoupled-strategy)
2. [Directory Layout & File Inventory](#2-directory-layout--file-inventory)
3. [FastAPI Backend Service API Reference](#3-fastapi-backend-service-api-reference)
4. [Machine Learning Engine Deep Dive](#4-machine-learning-engine-deep-dive)
5. [Streamlit Frontend Dashboard](#5-streamlit-frontend-dashboard)
6. [Multi-Container Deployment (Docker)](#6-multi-container-deployment-docker)
7. [Verification & Testing Suite](#7-verification--testing-suite)

---

## 1. System Architecture & Decoupled Strategy

Teachable Machine Pro is structured around a **decoupled client-server architecture**. Rather than running model training and feature extraction directly inside the frontend thread (which is prone to blocking, memory leaks, and browser crashes), the user interface is completely decoupled from the computational backend.

### High-Level Architecture Diagram
The services interact using asynchronous HTTP web protocols over the local network or containerized bridge network:

```mermaid
graph TD
    subgraph Streamlit_Client [Streamlit Frontend (Port 8501)]
        UI[Glassmorphic UI Controls]
        Webcam[Webcam Capture / Upload Uploader]
        AnalyticsUI[Analytics & PCA Charts]
    end

    subgraph FastAPI_Server [FastAPI Backend (Port 8000)]
        API[Uvicorn Server / FastAPI Endpoints]
        MW[Content-Length Size Middleware]
        Ingest[Image Ingestion & Sanitization]
        ML[PyTorch Feature Extraction & Scikit-Learn fitting]
        Cache[MD5-hashed Disk Feature Cache]
        Log[JSONL Prediction Logging]
    end

    subgraph Storage [Persistent Storage]
        Dir[backend/dataset/class_folders]
        Model[backend/model.pkl]
        Logs[backend/logs/predictions.jsonl]
    end

    UI -->|1. POST /upload-sample| API
    UI -->|2. POST /train| API
    UI -->|3. POST /predict| API
    UI -->|4. GET /analytics & /features-pca| API

    API --> MW
    MW --> Ingest
    Ingest -->|Save sanitized images| Dir
    
    API --> ML
    ML -->|Read images & check cache| Cache
    ML -->|Extract embeddings & train classifier| Model
    
    API -->|Serve prediction & log entry| Logs
```

### Key Engineering Benefits
* **轻量级前端 (Lightweight Frontend)**: Streamlit only renders the UI elements, progress bars, and image frames. It remains fluid and responsive during heavy computer vision computations.
* **计算后端独立性 (Independent Backend)**: The backend is built using FastAPI with an asynchronous uvicorn server. It runs heavy tensor calculations using PyTorch and Scikit-Learn. In production, the backend can be hosted on a GPU-accelerated cloud instance (e.g. AWS EC2, GCP Compute Engine) while keeping the frontend on a cheap static host.
* **并发隔离 (Non-Blocking execution)**: Model training is delegated to a Python `ThreadPoolExecutor` from `concurrent.futures`. This prevents long-running training requests from blocking the main event loop and keeps the FastAPI server online for other concurrent requests.

---

## 2. Directory Layout & File Inventory

The workspace is organized to enforce the separation of concerns:

```
teachable_machine/
├── backend/                        # FastAPI Machine Learning Backend Service
│   ├── dataset/                    # Local storage for class-wise training images
│   │   └── [class_name]/           # Dynamically created folders for each category
│   ├── logs/                       # Backend logging directory
│   │   └── predictions.jsonl       # Appended JSON telemetry of each inference
│   ├── feature_cache/              # Disk-based .npy cache for MobileNetV3 features
│   ├── Dockerfile                  # Builds backend service container image
│   ├── main.py                     # FastAPI routes, middleware, and request validation
│   ├── ml_engine.py                # PyTorch feature extraction, caching, and classification
│   └── inspect_model.py            # Diagnostic script to read model.pkl configuration
├── frontend/                       # Streamlit UI Dashboard Client Service
│   ├── Dockerfile                  # Builds frontend service container image
│   └── app.py                      # Main Streamlit script (UI, state, API requests)
├── learning_hub/                   # Interactive learning portal assets
│   ├── assets/                     # Diagnostic graphs, slides, or diagrams
│   ├── index.html                  # HTML structure for the AI Theory Hub
│   ├── script.js                   # Interactive interactive simulators
│   └── styles.css                  # Custom styling for the portal
├── sliders/                        # Standalone Capstone Presentation deck
│   ├── index.html                  # Interactive HTML slides player
│   └── slide_01_title.png (etc)   # Slide images (01 to 18)
├── tasks/                          # Project task tracking
│   └── project_tasks.md            # Checklist of development milestones
├── tests/                          # Automated pytest suite
│   ├── __init__.py
│   └── test_main.py                # Unit and integration test assertions
├── docker-compose.yml              # Multi-container service configuration
├── README.md                       # High-level overview and setup guide
├── requirements.txt                # Unified python library dependencies
├── upload_samples.py               # Bulk dataset zip file uploader script
└── student_project_guide_...pdf   # Original project specification blueprint
```

---

## 3. FastAPI Backend Service API Reference

The FastAPI service exposes a REST API running on Port 8000. It includes security, logging, cache clear controls, and diagnostics.

### 🛡️ Security & Size Middlewares
1. **Upload Size Limit Middleware**: Rejects any incoming request whose `Content-Length` header exceeds `20 MB` with an `HTTP 413 Payload Too Large` status.
2. **CORS Middleware**: Restricts communication to trusted origins, specifically allowing Streamlit frontend services (`http://localhost:8501`, `http://127.0.0.1:8501`, and the Docker bridge service `http://frontend:8501`).
3. **MIME Magic-Bytes Validation**: When images are uploaded via `POST /upload-sample`, the server reads the first 16 bytes of the file to verify if the file header matches known magic-byte signatures (`b"\xff\xd8\xff"` for JPG, `b"\x89PNG"` for PNG, `b"RIFF"` for WEBP, `b"BM"` for BMP). Any mismatch (e.g., uploading a text file renamed to `.png`) is rejected.

### 🔌 Endpoint Directory

#### 1. API Health Check
* **Route**: `GET /`
* **Response Payload**:
  ```json
  {
    "status": "Teachable Machine API is running",
    "dataset_directory": "e:\\SMIT\\teachable_machine\\backend\\dataset",
    "model_trained": true,
    "active_classes": ["Mug", "Book"]
  }
  ```

#### 2. Ingest Sample Images
* **Route**: `POST /upload-sample`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `class_name` (Form parameter, string): Target category label.
  * `files` (Uploaded files list): Raw binary files to save.
* **Logic**: 
  1. Sanitizes `class_name` using regex `[^a-zA-Z0-9_\-]` to replace spaces and special characters with underscores.
  2. Ensures the folder `backend/dataset/{sanitized_class_name}/` exists.
  3. Renames each file using `uuid.uuid4()` combined with its lowercase extension to prevent filename collision.
  4. Writes the binary payload to disk.
* **Response Payload**:
  ```json
  {
    "message": "Successfully uploaded 12 image(s) for class 'Mug'.",
    "class_name": "Mug",
    "saved_count": 12
  }
  ```

#### 3. Train Model
* **Route**: `POST /train`
* **Content-Type**: `application/x-www-form-urlencoded`
* **Parameters**:
  * `backbone_name` (string, default: `"MobileNetV3"`): Feature extractor choice (`MobileNetV3` or `ResNet18`).
  * `classifier_type` (string, default: `"LogisticRegression"`): Classification algorithm (`LogisticRegression`, `SVM`, `RandomForest`, or `KNN`).
  * `c_value` (float, default: `1.0`): Regularization parameter.
  * `penalty` (string, default: `"l2"`): Regularization penalty type (`l1` or `l2`).
* **Validation Guards**:
  * Rejects the training run with `HTTP 400 Bad Request` if the dataset directory does not exist or has fewer than 2 distinct category folders.
  * Rejects if any category folder contains zero images.
* **Response Payload**:
  ```json
  {
    "status": "success",
    "message": "Model trained successfully!",
    "details": {
      "classes": ["Book", "Mug"],
      "samples_trained": 28,
      "samples_validated": 6,
      "class_distribution": { "Book": 17, "Mug": 17 },
      "validation_metrics": {
        "split_executed": true,
        "accuracy": 100.0,
        "precision": 100.0,
        "recall": 100.0,
        "f1_score": 100.0,
        "confusion_matrix": {
          "labels": ["Book", "Mug"],
          "matrix": [[3, 0], [0, 3]]
        }
      }
    }
  }
  ```

#### 4. Inference & Real-Time Prediction
* **Route**: `POST /predict`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `file` (Single uploaded file): Image to classify.
* **Validation Guards**:
  * Checks if `model.pkl` exists on disk. If not, returns `HTTP 400 Bad Request`.
* **Response Payload**:
  ```json
  {
    "predicted_class": "Mug",
    "probabilities": { "Book": 5.2, "Mug": 94.8 },
    "bounding_box_image": "base64_encoded_string...",
    "saliency_image": "base64_encoded_string...",
    "inference_time_ms": 28.5,
    "backbone_used": "MobileNetV3"
  }
  ```

#### 5. Reset Workspace
* **Route**: `DELETE /reset`
* **Description**: Removes the trained `model.pkl`, clears the backend dataset folders (keeping only `.gitkeep`), and flushes the in-memory cache.
* **Response Payload**:
  ```json
  {
    "status": "success",
    "message": "Dataset and trained model have been successfully reset.",
    "deleted_classes": ["Book", "Mug"]
  }
  ```

#### 6. Export Model
* **Route**: `GET /export-model`
* **Description**: Downloads the raw trained binary file (`model.pkl`) directly as an attachment.

#### 7. Analytics Dashboard Telemetry
* **Route**: `GET /analytics`
* **Description**: Parses `backend/logs/predictions.jsonl` to calculate counts and averages.
* **Response Payload**:
  ```json
  {
    "total_predictions": 45,
    "class_distribution": { "Mug": 30, "Book": 15 },
    "avg_confidence": { "Mug": 92.4, "Book": 88.7 },
    "recent_predictions": [ ... ]
  }
  ```

#### 8. Latent Space PCA
* **Route**: `GET /features-pca`
* **Description**: Performs a 2D Principal Component Analysis (PCA) projection of all image feature vectors. Used to plot dataset clustering.

---

## 4. Machine Learning Engine Deep Dive

The core logic of the transfer learning workflow is housed in [ml_engine.py](file:///e:/SMIT/teachable_machine/backend/ml_engine.py). It operates strictly on a **CPU device** (`torch.device("cpu")`) to ensure maximum portability, avoiding large GPU package requirements.

```
       [Input Image Bytes]
                │
                ▼
     [PIL Image (RGB Mode)]
                │
                ▼
     [Resize to 224 x 224 px]
                │
                ▼
    [Normalize Color Channels]  <-- ImageNet Mean/Std Statistics
                │
                ▼
  [Pass through MobileNetV3/ResNet18] (Frozen Weights Backbone)
                │
                ▼
   [Flattened Embedding Vector] (576-dim or 512-dim)
                │
         ┌──────┴──────┐
         ▼             ▼
   [Predict Mode]   [Train Mode]
     (Load pkl)       (Scikit-Learn Logistic Regression)
```

### A. Preprocessing Uniformity (ImageNet Normalization)
Machine learning models are highly sensitive to raw pixel configurations. To ensure that prediction accuracy remains intact:
1. Both the training and inference pipelines load raw image bytes into a PIL Image and force it to RGB mode.
2. The image is resized to exactly $224 \times 224$ pixels.
3. The image is normalized using standard ImageNet mean and standard deviations:
   * **Means**: $R = 0.485$, $G = 0.456$, $B = 0.406$
   * **Standard Deviations**: $R = 0.229$, $G = 0.224$, $B = 0.225$

During training, we apply an **Augmentation Pipeline** (`train_augmentation`) to duplicate and extend the small training dataset, adding horizontal flips, color jitter (brightness, contrast, saturation, hue), and small rotations (up to 10 degrees). The original unaugmented image is also preserved.

### B. High-Speed Feature Extraction Caching
Running neural network feature extraction on the CPU for every training button click can make training slow if the dataset is large. 
To address this, the engine uses **disk-based feature caching**:
1. When features for an image are requested, the engine reads the file bytes and generates an MD5 hash.
2. It looks for a cache file named `backend/feature_cache/{md5_hash}_{backbone}.npy`.
3. **Cache Hit**: The 1D vector is loaded instantly using `np.load()`, saving CPU cycles.
4. **Cache Miss**: The image is preprocessed, passed through PyTorch, and the resulting vector is saved to `feature_cache` for next time.
5. In addition to disk caching, an in-memory cached variable `_MODEL_CACHE` is populated when `model.pkl` is loaded during inference. This prevents disk read latencies during real-time webcam loops.

### C. Convex Classifier & Stratified Validation Split
The classification head relies on standard Scikit-Learn classifiers. When training is triggered:
1. The dataset features are loaded.
2. The engine evaluates whether it is eligible to perform a validation split (requires at least 6 total samples and at least 2 samples per class).
3. If eligible, it performs a **Stratified Validation Split** (80% training, 20% validation) using `sklearn.model_selection.train_test_split`. Stratification ensures each subset contains a balanced proportion of target categories.
4. It fits the classifier (e.g., `LogisticRegression` with regularizer `C` and `L2` solver).
5. If a split is executed, it computes accuracy, weighted precision, recall, f1-score, and a confusion matrix to display on the Streamlit dashboard.
6. The classifier, label map, backbone name, and embedding dimensions are saved in a pickle dictionary bundle:
   ```python
   model_bundle = {
       "classifier": classifier,
       "classifier_type": classifier_type,
       "label_map": label_map,
       "backbone_name": backbone_name,
       "features_dim": 512 if backbone_name == "ResNet18" else 576
   }
   ```

### D. Advanced Inference Telemetry: Grad-CAM & Scanning Bounding Boxes
During `POST /predict`, the engine doesn't just calculate class probabilities; it generates advanced visual overlays:

#### 1. Neural Attention Maps (Grad-CAM)
The engine registers forward and backward hooks into the final convolutional layer of the PyTorch backbone (e.g., `features[-1]` on MobileNetV3, or `layer4[-1]` on ResNet18):
1. A forward pass computes features.
2. The predicted class index is determined.
3. The model backpropagates through a class score proxy (calculated using the logistic regression weights).
4. The captured gradients are global average pooled and multiplied by the convolutional activations to yield a grayscale 2D heat-intensity map.
5. Grayscale heat maps are colorized using a Jet Colormap approximation (Blue to Red) and blended transparently over the test image to highlight where the model focused.

#### 2. Scanning Bounding Boxes
A scanning bounding box is drawn automatically over regions of high attention:
1. Grayscale heat intensities are thresholded (`mask = s_np > 0.35`).
2. The coordinates of all pixels exceeding this threshold are retrieved.
3. The minimum and maximum boundaries are calculated to form a bounding rectangle.
4. A high-contrast cyan border (`#00f2fe`) is drawn, capped with thick, neon-purple corners (`#7f00ff`) for a premium sci-fi overlay.

---

## 5. Streamlit Frontend Dashboard

The user interface, located in [app.py](file:///e:/SMIT/teachable_machine/frontend/app.py), serves as the interactive dashboard.

### State Management (`st.session_state`)
Streamlit reruns the script on every user interaction. To persist the application state across runs, we initialize and check keys inside `st.session_state`:
* `classes`: List of active categories entered by the user.
* `is_trained`: Boolean gating key. Hides testing panels until set to `True`.
* `backend_active`: Boolean tracking connection health.
* `last_prediction`: Stores the response payload from the `/predict` endpoint.
* `prediction_history`: Rolling list of the last 15 prediction scores, rendered as a line chart.

### Interface Tabs
The application is structured into three clean workspace tabs:
1. **Model Training Workspace**:
   * **Left Column**: Class creator, image file uploader, live webcam snapshot tool, hyperparameter radio sliders, and the primary "Train Custom Model" action button.
   * **Right Column**: Inference sandbox. Shows the camera feed, active predictions, bounding boxes, attention heatmaps, and progress meters.
2. **Theory & Simulators**: Inserts an interactive HTML/JS portal explaining concepts like gradient descent, training epochs, overfitting, and neural network weights.
3. **Analytics Dashboard**: Tracks predictions, showing historical averages and bar charts of predicted categories.

---

## 6. Multi-Container Deployment (Docker)

The application is fully containerized. Developers can run the entire workspace using Docker Compose.

### Service Configurations (`docker-compose.yml`)
The stack is split into two networked services:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - backend-dataset:/app/backend/dataset
      - ./backend/logs:/app/backend/logs
      - ./backend/feature_cache:/app/backend/feature_cache

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend

volumes:
  backend-dataset:
```

### Volume Persistence
* **`backend-dataset`**: A named Docker volume mapping to `/app/backend/dataset`. This ensures that uploaded image datasets are preserved on the host machine when containers are stopped, rebuilt, or updated.
* **`logs` & `feature_cache`**: Bind mounted folders that sync prediction logs and feature cache arrays directly into the local workspace for convenient diagnostic access.

---

## 7. Verification & Testing Suite

We use **pytest** to verify the core endpoints, helper methods, and error cases in [test_main.py](file:///e:/SMIT/teachable_machine/tests/test_main.py).

### Running Automated Tests
To execute the backend test assertions locally:
```bash
# Install testing dependencies
pip install pytest httpx

# Run the test suite from the project root
pytest tests/ -v
```

### Coverage Assertions
* **Sanitization Check**: Validates `sanitize_class_name()` against spaces, uppercase characters, and special symbol characters (e.g. `Class@Name!` translates to `Class_Name_`).
* **Health API Check**: Asserts `GET /` returns HTTP 200, tracks the status string, and includes the expected keys (`model_trained`, `active_classes`).
* **Dataset Diagnostics**: Verifies `GET /dataset-info` yields class counts as integers.
* **Training Validation**: Asserts that `POST /train` correctly rejects requests with fewer than 2 classes or empty category directories.
* **Ingestion Verification**: Confirms that non-image payloads (e.g. text files) are rejected during sample uploads.
